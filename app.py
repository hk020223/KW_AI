import streamlit as st
import pandas as pd
import os
import glob
import datetime
import time
import base64
import re  # 정규표현식 사용
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

# Firebase 라이브러리 (Admin SDK)
import firebase_admin
from firebase_admin import credentials, firestore

# -----------------------------------------------------------------------------
# [0] 설정 및 데이터 로드
# -----------------------------------------------------------------------------
st.set_page_config(page_title="KW-강의마스터 Pro", page_icon="🎓", layout="wide")

# [모바일 최적화 CSS]
st.markdown("""
    <style>
        footer { visibility: hidden; }
        @media only screen and (max-width: 600px) {
            .main .block-container {
                padding-left: 0.2rem !important;
                padding-right: 0.2rem !important;
                padding-top: 2rem !important;
                max-width: 100% !important;
            }
            div[data-testid="stMarkdownContainer"] table {
                width: 100% !important;
                table-layout: fixed !important;
                display: table !important;
                font-size: 10px !important;
                margin-bottom: 0px !important;
            }
            div[data-testid="stMarkdownContainer"] th, 
            div[data-testid="stMarkdownContainer"] td {
                padding: 1px 1px !important;
                word-wrap: break-word !important;
                word-break: break-all !important;
                white-space: normal !important;
                line-height: 1.1 !important;
                vertical-align: middle !important;
            }
            div[data-testid="stMarkdownContainer"] th:first-child,
            div[data-testid="stMarkdownContainer"] td:first-child {
                width: 35px !important;
                font-size: 8px !important;
                text-align: center !important;
                letter-spacing: -0.5px !important;
            }
            button { min-height: 45px !important; }
            input { font-size: 16px !important; }
        }
    </style>
""", unsafe_allow_html=True)

# API Key 로드
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
else:
    api_key = os.environ.get("GOOGLE_API_KEY", "")

if not api_key:
    st.error("🚨 **Google API Key가 설정되지 않았습니다.**")
    st.stop()

# 세션 상태 초기화 (없으면 생성)
if "global_log" not in st.session_state:
    st.session_state.global_log = [] 
if "timetable_result" not in st.session_state:
    st.session_state.timetable_result = "" 
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] 
if "current_menu" not in st.session_state:
    st.session_state.current_menu = "🤖 AI 학사 지식인"
# 라디오 버튼 위젯 상태 초기화
if "menu_radio" not in st.session_state:
    st.session_state["menu_radio"] = "🤖 AI 학사 지식인"

if "timetable_chat_history" not in st.session_state:
    st.session_state.timetable_chat_history = []
if "graduation_analysis_result" not in st.session_state:
    st.session_state.graduation_analysis_result = ""
if "graduation_chat_history" not in st.session_state:
    st.session_state.graduation_chat_history = []
if "user" not in st.session_state:
    st.session_state.user = None

# 현재 불러온 시간표 메타데이터 (ID, 이름, 즐겨찾기 여부 등) 관리용
if "current_timetable_meta" not in st.session_state:
    st.session_state.current_timetable_meta = {}

# [추가] 선택된 강의계획서 뷰어 상태 관리
if "selected_syllabus" not in st.session_state:
    st.session_state.selected_syllabus = None

def add_log(role, content, menu_context=None):
    timestamp = datetime.datetime.now().strftime("%H:%M")
    st.session_state.global_log.append({
        "role": role,
        "content": content,
        "time": timestamp,
        "menu": menu_context
    })

# HTML 코드 정제 함수
def clean_html_output(text):
    cleaned = text.strip()
    if cleaned.startswith("```html"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.replace("```html", "").replace("```", "").strip()

# ★ 재시도(Retry) 로직 ★
def run_with_retry(func, *args, **kwargs):
    max_retries = 5
    delays = [1, 2, 4, 8, 16]
    for i in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "503" in error_msg:
                if i < max_retries - 1:
                    time.sleep(delays[i])
                    continue
            raise e

# -----------------------------------------------------------------------------
# [Firebase Manager] Firestore 기반 자체 인증 및 DB 관리
# -----------------------------------------------------------------------------
class FirebaseManager:
    def __init__(self):
        self.db = None
        self.is_initialized = False
        self.init_firestore()

    def init_firestore(self):
        """Firestore DB 초기화 (Service Account 사용)"""
        if "firebase_service_account" in st.secrets:
            try:
                if not firebase_admin._apps:
                    cred_info = dict(st.secrets["firebase_service_account"])
                    cred = credentials.Certificate(cred_info)
                    firebase_admin.initialize_app(cred)
                self.db = firestore.client()
                self.is_initialized = True
            except Exception:
                pass

    def login(self, email, password):
        """Firestore에서 이메일/비번 매칭 검사"""
        if not self.is_initialized:
            return None, "Firebase 연결 실패"
        
        try:
            # users 컬렉션에서 email과 password가 일치하는 문서 검색
            users_ref = self.db.collection('users')
            query = users_ref.where('email', '==', email).where('password', '==', password).stream()
            
            for doc in query:
                user_data = doc.to_dict()
                user_data['localId'] = doc.id
                return user_data, None
            
            return None, "이메일 또는 비밀번호가 일치하지 않습니다."
        except Exception as e:
            return None, f"로그인 오류: {str(e)}"

    def signup(self, email, password):
        """Firestore에 신규 유저 정보 저장"""
        if not self.is_initialized:
            return None, "Firebase 연결 실패"

        try:
            users_ref = self.db.collection('users')
            existing_user = list(users_ref.where('email', '==', email).stream())
            if len(existing_user) > 0:
                return None, "이미 가입된 이메일입니다."
            
            new_user_ref = users_ref.document()
            user_data = {
                "email": email,
                "password": password,
                "created_at": firestore.SERVER_TIMESTAMP
            }
            new_user_ref.set(user_data)
            
            user_data['localId'] = new_user_ref.id
            return user_data, None
        except Exception as e:
            return None, f"회원가입 오류: {str(e)}"

    def save_data(self, collection, doc_id, data):
        """데이터 저장 (덮어쓰기)"""
        if not self.is_initialized or not st.session_state.user:
            return False
        try:
            user_id = st.session_state.user['localId']
            doc_ref = self.db.collection('users').document(user_id).collection(collection).document(doc_id)
            data['updated_at'] = firestore.SERVER_TIMESTAMP
            doc_ref.set(data)
            return True
        except:
            return False

    def update_data(self, collection, doc_id, data):
        """데이터 부분 업데이트 (이름 변경, 즐겨찾기 등)"""
        if not self.is_initialized or not st.session_state.user:
            return False
        try:
            user_id = st.session_state.user['localId']
            doc_ref = self.db.collection('users').document(user_id).collection(collection).document(doc_id)
            data['updated_at'] = firestore.SERVER_TIMESTAMP
            doc_ref.update(data)
            return True
        except:
            return False

    def load_collection(self, collection):
        """데이터 목록 불러오기"""
        if not self.is_initialized or not st.session_state.user:
            return []
        try:
            user_id = st.session_state.user['localId']
            docs = self.db.collection('users').document(user_id).collection(collection).order_by('updated_at', direction=firestore.Query.DESCENDING).stream()
            return [{"id": doc.id, **doc.to_dict()} for doc in docs]
        except:
            return []

fb_manager = FirebaseManager()

# PDF 데이터 로드
@st.cache_resource(show_spinner="PDF 문서를 분석 중입니다...")
def load_knowledge_base():
    if not os.path.exists("data"):
        return ""
    pdf_files = glob.glob("data/*.pdf")
    if not pdf_files:
        return ""
    all_content = ""
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(pdf_file)
            pages = loader.load_and_split()
            filename = os.path.basename(pdf_file)
            all_content += f"\n\n--- [문서: {filename}] ---\n"
            for page in pages:
                all_content += page.page_content
        except Exception as e:
            print(f"Error loading {pdf_file}: {e}")
            continue
    return all_content

PRE_LEARNED_DATA = load_knowledge_base()

# -----------------------------------------------------------------------------
# [1] AI 엔진 (gemini-2.5-flash-preview-09-2025)
# -----------------------------------------------------------------------------
def get_llm():
    if not api_key: return None
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-09-2025", temperature=0)

def get_pro_llm():
    if not api_key: return None
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-09-2025", temperature=0)

def ask_ai(question):
    llm = get_llm()
    if not llm: return "⚠️ API Key 오류"
    def _execute():
        chain = PromptTemplate.from_template(
            "문서 내용: {context}\n질문: {question}\n문서에 기반해 답변해줘. 답변할 때 근거가 되는 문서의 원문 내용을 반드시 \" \" (쌍따옴표) 안에 인용해서 포함해줘."
        ) | llm
        return chain.invoke({"context": PRE_LEARNED_DATA, "question": question}).content
    try:
        return run_with_retry(_execute)
    except Exception as e:
        if "RESOURCE_EXHAUSTED" in str(e):
            return "⚠️ **잠시만요!** 사용량이 많아 AI가 숨을 고르고 있습니다. 1분 뒤에 다시 시도해주세요."
        return f"❌ AI 오류: {str(e)}"

# 공통 프롬프트 지시사항
COMMON_TIMETABLE_INSTRUCTION = """
[★★★ 핵심 알고리즘: 3단계 검증 및 필터링 (Strict Verification) ★★★]
1. **Step 1: 요람(Curriculum) 기반 '수강 대상' 리스트 확정**:
   - PDF 요람 문서에서 **'{major} {grade} {semester}'**에 배정된 **'표준 이수 과목' 목록**을 추출.
2. **Step 2: 학년 정합성 검사 (Grade Validation)**:
   - 사용자가 선택한 학년({grade})과 시간표의 대상 학년이 일치하지 않으면 과감히 제외.
3. **Step 3: 시간표 데이터와 정밀 대조 (Exact Match)**:
   - 위 단계를 통과한 과목만 시간표에 배치. 과목명 완전 일치 필수.
   - **[핵심 규칙] 요일별 교시 분리 배정**: 만약 강의 시간이 **'월3, 수4'**로 되어 있다면, **월요일은 3교시만, 수요일은 4교시만** 채워야 합니다.
   - **절대** '월3,4' 혹은 '수3,4'처럼 연강으로 임의 확장하거나 빈 시간을 채워넣지 마세요.
4. **출력 형식 (세로형 HTML Table)**:
   - `table` 태그, `width="100%"`.
   - 행: 1~9교시 (시간 포함), 열: 월~일.
   - 같은 과목 같은 배경색, 공강은 흰색.
   - 셀 내용: `<b>과목명</b><br><small>교수명 (대상학년)</small>`
5. **온라인 및 원격 강의 처리**:
   - 표 맨 마지막 행에 `<b>온라인/기타</b>` 행 추가하여 포함.
6. **출력 순서**: HTML 표 -> 필수 과목 검증 -> 제외 목록
"""

# [수정] 진단 결과를 입력받아 우선순위를 배정하는 로직 추가
def generate_timetable_ai(major, grade, semester, target_credits, blocked_times_desc, requirements, diagnosis_context=None):
    llm = get_llm()
    if not llm: return "⚠️ API Key 오류"
    def _execute():
        # 기본 템플릿
        base_template = """
        너는 대학교 수강신청 전문가야. 오직 제공된 [학습된 문서]의 텍스트 데이터에 기반해서만 시간표를 짜줘.
        [학생 정보]
        - 소속: {major}
        - 학년/학기: {grade} {semester}
        - 목표: {target_credits}학점
        - 공강 필수: {blocked_times}
        - 추가요구: {requirements}
        """

        # 진단 결과가 있을 경우 우선순위 로직 추가
        if diagnosis_context:
            base_template += f"""
            [성적 및 진로 진단 결과 (반영 필수)]
            {diagnosis_context}

            [★★★ 개인화 우선순위 배정 규칙 (Priority Logic) ★★★]
            1. **1순위 (Must):** 해당 학년/학기에 지정된 표준 이수 필수 과목 (졸업 필수).
            2. **2순위 (Should):** 위 [성적 및 진로 진단 결과]에서 **'재수강 필요'** 혹은 **'미이수 필수'**로 지적된 과목.
            3. **3순위 (Could):** 위 [성적 및 진로 진단 결과]의 **'직무 솔루션'**에서 추천한 과목.
            4. **제외 (Exclude):** 위 진단 결과에서 **이미 이수한 것**으로 확인된 과목 (단, 재수강 대상은 제외).
            """
        
        # 공통 지시사항 및 문서 연결
        base_template += COMMON_TIMETABLE_INSTRUCTION + """
        [추가 지시사항]
        - **HTML 코드를 마크다운 코드 블록(```html)으로 감싸지 마라.** 그냥 Raw HTML 텍스트로 출력해라.
        [학습된 문서]
        {context}
        """
        
        prompt = PromptTemplate(template=base_template, input_variables=["context", "major", "grade", "semester", "target_credits", "blocked_times", "requirements"])
        chain = prompt | llm
        
        input_data = {
            "context": PRE_LEARNED_DATA,
            "major": major,
            "grade": grade,
            "semester": semester,
            "target_credits": target_credits,
            "blocked_times": blocked_times_desc,
            "requirements": requirements
        }
        return chain.invoke(input_data).content

    try:
        response_content = run_with_retry(_execute)
        return clean_html_output(response_content)
    except Exception as e:
        if "RESOURCE_EXHAUSTED" in str(e):
            return "⚠️ **사용량 초과**: 잠시 후 다시 시도해주세요."
        return f"❌ AI 오류: {str(e)}"

def chat_with_timetable_ai(current_timetable, user_input, major, grade, semester):
    llm = get_llm()
    def _execute():
        template = """
        너는 현재 시간표에 대한 상담을 해주는 AI 조교야.
        [현재 시간표 상태]
        {current_timetable}
        [사용자 입력]
        "{user_input}"
        [학생 정보]
        - 소속: {major}
        - 학년/학기: {grade} {semester}
        [지시사항]
        사용자의 입력 의도를 파악해서 아래 두 가지 중 하나로 반응해.
        **Case 1. 시간표 수정 요청 (예: "1교시 빼줘"):**
        - 시간표를 **재작성**.
        """ + COMMON_TIMETABLE_INSTRUCTION + """
        - **HTML 코드를 마크다운 코드 블록(```html)으로 감싸지 마라.** Raw HTML로 출력해.
        **Case 2. 단순 질문 (예: "이거 선수과목 뭐야?"):**
        - **시간표 재출력 X**, 텍스트 답변만.
        - **근거가 되는 문서 원문 내용을 반드시 " " (쌍따옴표) 안에 인용.**
        답변 시작에 [수정] 또는 [답변] 태그를 붙여서 구분.
        [학습된 문서]
        {context}
        """
        prompt = PromptTemplate(template=template, input_variables=["current_timetable", "user_input", "major", "grade", "semester", "context"])
        chain = prompt | llm
        return chain.invoke({
            "current_timetable": current_timetable, 
            "user_input": user_input,
            "major": major,
            "grade": grade,
            "semester": semester,
            "context": PRE_LEARNED_DATA
        }).content
    try:
        response_content = run_with_retry(_execute)
        if "[수정]" in response_content:
            parts = response_content.split("[수정]", 1)
            if len(parts) > 1:
                return "[수정]" + clean_html_output(parts[1])
            else:
                return clean_html_output(response_content)
        return response_content
    except Exception as e:
        if "RESOURCE_EXHAUSTED" in str(e):
            return "⚠️ **사용량 초과**: 잠시 후 다시 시도해주세요."
        return f"❌ AI 오류: {str(e)}"

# =============================================================================
# [섹션] 성적 및 진로 진단 분석 함수
# =============================================================================
def analyze_graduation_requirements(uploaded_images):
    llm = get_pro_llm()
    if not llm: return "⚠️ API Key 오류"

    def encode_image(image_file):
        image_file.seek(0)
        return base64.b64encode(image_file.read()).decode("utf-8")

    image_messages = []
    for img_file in uploaded_images:
        base64_image = encode_image(img_file)
        image_messages.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        })

    def _execute():
        prompt = """
        당신은 [냉철하고 현실적인 대기업 인사담당자 출신의 취업 컨설턴트]입니다.
        제공된 학생의 [성적표 이미지]와 [학습된 학사 문서]를 바탕으로 3가지 측면에서 분석 결과를 작성해주세요.
        
        **[핵심 지시사항 - 중요]**
        - 단순히 "열심히 하세요" 같은 뜬구름 잡는 조언은 하지 마십시오.
        - **반드시** 삼성전자, SK하이닉스, 현대자동차, 네이버, 카카오 등 **실제 한국 주요 대기업의 실명과 구체적인 직무명(JD)**을 언급하며 조언하세요.
        - 예: "삼성전자 DS부문 메모리사업부의 공정기술 직무에서는 반도체공학 A학점 이상을 선호하지만, 현재 학생의 성적은 B+이므로..." 와 같이 구체적으로 비교하세요.

        **[출력 형식]**
        반드시 아래의 구분자(`[[SECTION: ...]]`)를 사용하여 답변을 3개의 구역으로 명확히 나누세요.

        [[SECTION:GRADUATION]]
        ### 🎓 1. 졸업 요건 정밀 진단
        - [학습된 학사 문서]의 규정과 비교하여 졸업 가능 여부를 판정하세요.
        - 부족한 학점(전공, 교양 등)과 미이수 필수 과목을 표나 리스트로 정리하세요.
        - **종합 판정:** [졸업 가능 / 위험 / 불가]

        [[SECTION:GRADES]]
        ### 📊 2. 성적 정밀 분석
        - **전체 평점 vs 전공 평점 비교:** 전공 학점이 전체보다 낮은지 확인하고 질책하세요. (직무 전문성 결여 지적)
        - **재수강 권고:** C+ 이하의 전공 핵심 과목이 있다면 구체적으로 지적하며 재수강을 강력히 권고하세요.
        - **수강 패턴 분석:** 꿀강(학점 따기 쉬운 교양) 위주로 들었는지, 기피 과목(어려운 전공)을 피했는지 간파하고 지적하세요.

        [[SECTION:CAREER]]
        ### 💼 3. AI 커리어 솔루션 (대기업 JD 매칭)
        - **직무 추천:** 학생의 수강 내역(회로 위주, SW 위주 등)을 분석하여 가장 적합한 **구체적인 대기업 직무**를 2~3개 추천하세요. (예: 삼성전자 회로설계, 현대모비스 임베디드SW 등)
        - **Skill Gap 분석:** 해당 직무의 시장 요구사항(대기업 채용 기준) 대비 현재 부족한 점을 냉정하게 꼬집으세요.
        - **Action Plan:** 남은 학기에 반드시 수강해야 할 과목이나, 학교 밖에서 채워야 할 경험(프로젝트, 기사 자격증 등)을 구체적으로 지시하세요.

        [학습된 학사 문서]
        """
        
        content_list = [{"type": "text", "text": prompt}]
        content_list.extend(image_messages)
        content_list.append({"type": "text", "text": f"\n\n{PRE_LEARNED_DATA}"})

        message = HumanMessage(content=content_list)
        response = llm.invoke([message])
        return response.content

    try:
        return run_with_retry(_execute)
    except Exception as e:
         if "RESOURCE_EXHAUSTED" in str(e):
            return "⚠️ **사용량 초과**: 잠시 후 다시 시도해주세요."
         return f"❌ AI 오류: {str(e)}"

# 성적/진로 상담 및 수정 함수 (페르소나 유지)
def chat_with_graduation_ai(current_analysis, user_input):
    llm = get_llm()
    def _execute():
        template = """
        당신은 냉철하고 독설적인 'AI 취업 컨설턴트'입니다.
        학생의 성적 및 진로 진단 결과는 다음과 같습니다:
        
        [현재 진단 결과]
        {current_analysis}

        [사용자 입력]
        "{user_input}"

        [지시사항]
        - 사용자의 질문에 대해 현실적이고 직설적으로 답변하세요. 위로는 필요 없습니다.
        - 정보 수정 요청(예: "나 이 과목 들었어")이 들어오면 `[수정]` 태그를 붙이고 전체 진단 결과를 업데이트하세요.
        - **기업 채용 관점**에서 답변하세요. "이 과목은 삼성전자가 좋아합니다/신경 안 씁니다" 식으로 설명하세요.
        
        [참고 문헌]
        {context}
        """
        prompt = PromptTemplate(template=template, input_variables=["current_analysis", "user_input", "context"])
        chain = prompt | llm
        return chain.invoke({
            "current_analysis": current_analysis,
            "user_input": user_input,
            "context": PRE_LEARNED_DATA
        }).content

    try:
        return run_with_retry(_execute)
    except Exception as e:
        if "RESOURCE_EXHAUSTED" in str(e):
            return "⚠️ **사용량 초과**: 잠시 후 다시 시도해주세요."
        return f"❌ AI 오류: {str(e)}"

# -----------------------------------------------------------------------------
# [2] UI 구성
# -----------------------------------------------------------------------------
def change_menu(menu_name):
    st.session_state.current_menu = menu_name

with st.sidebar:
    st.title("🗂️ 활동 로그")
    # [로그인 UI]
    if st.session_state.user is None:
        with st.expander("🔐 로그인 / 회원가입", expanded=True):
            auth_mode = st.radio("모드 선택", ["로그인", "회원가입"], horizontal=True)
            email = st.text_input("이메일")
            password = st.text_input("비밀번호", type="password")
            
            if st.button(auth_mode):
                if not email or not password:
                    st.error("이메일과 비밀번호를 입력하세요.")
                else:
                    if not fb_manager.is_initialized:
                        st.error("Firebase 연결 실패 (Secrets를 확인하세요)")
                    else:
                        with st.spinner(f"{auth_mode} 중..."):
                            if auth_mode == "로그인":
                                user, err = fb_manager.login(email, password)
                            else:
                                user, err = fb_manager.signup(email, password)
                            
                            # [로그인 성공 시] clear() 호출 안 함 -> 화면 상태 유지
                            if user:
                                st.session_state.user = user
                                st.success(f"환영합니다! ({user['email']})")
                                st.rerun()
                            else:
                                st.error(f"오류: {err}")
    else:
        st.info(f"👤 **{st.session_state.user['email']}**님")
        # [로그아웃 시] clear() 호출 -> 화면/데이터 완전 초기화
        if st.button("로그아웃"):
            st.session_state.clear()
            st.session_state["menu_radio"] = "🤖 AI 학사 지식인" 
            st.rerun()
            
    st.divider()
    st.caption("클릭하면 해당 화면으로 이동합니다.")
    log_container = st.container(height=300)
    with log_container:
        if not st.session_state.global_log:
            st.info("기록 없음")
        else:
            for i, log in enumerate(reversed(st.session_state.global_log)):
                label = f"[{log['time']}] {log['content'][:15]}..."
                if st.button(label, key=f"log_btn_{i}", use_container_width=True):
                    if log['menu']:
                        st.session_state.current_menu = log['menu']
                        st.session_state["menu_radio"] = log['menu'] 
                        st.rerun()
    st.divider()
    if PRE_LEARNED_DATA:
         st.success(f"✅ PDF 문서 학습 완료")
    else:
        st.error("⚠️ 데이터 폴더에 PDF 파일이 없습니다.")

# 메뉴 구성
menu = st.radio("기능 선택", ["🤖 AI 학사 지식인", "📅 스마트 시간표(수정가능)", "📈 성적 및 진로 진단"], 
                horizontal=True, key="menu_radio")

if menu != st.session_state.current_menu:
    st.session_state.current_menu = menu
    st.rerun()

st.divider()

if st.session_state.current_menu == "🤖 AI 학사 지식인":
    st.subheader("🤖 무엇이든 물어보세요")
    if st.session_state.user and fb_manager.is_initialized:
        with st.expander("💾 대화 내용 관리"):
            col_s1, col_s2 = st.columns(2)
            if col_s1.button("현재 대화 저장"):
                doc_id = str(int(time.time()))
                data = {"history": [msg for msg in st.session_state.chat_history]}
                if fb_manager.save_data('chat_history', doc_id, data):
                    st.toast("대화 내용이 저장되었습니다.")
            
            saved_chats = fb_manager.load_collection('chat_history')
            if saved_chats:
                selected_chat = col_s2.selectbox("불러오기", saved_chats, format_func=lambda x: datetime.datetime.fromtimestamp(int(x['id'])).strftime('%Y-%m-%d %H:%M'), label_visibility="collapsed")
                if col_s2.button("로드"):
                    st.session_state.chat_history = selected_chat['history']
                    st.rerun()

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    if user_input := st.chat_input("질문 입력"):
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        add_log("user", f"[지식인] {user_input}", "🤖 AI 학사 지식인")
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                response = ask_ai(user_input)
                st.markdown(response)
        st.session_state.chat_history.append({"role": "assistant", "content": response})

elif st.session_state.current_menu == "📅 스마트 시간표(수정가능)":
    st.subheader("📅 AI 맞춤형 시간표 설계")
    
    # [시간표 불러오기 및 관리 섹션 (UI 개편)]
    if st.session_state.user and fb_manager.is_initialized:
        saved_tables = fb_manager.load_collection('timetables')
        
        # 데이터 전처리: 필드가 없는 경우 기본값 할당
        fav_tables = []
        archive_tables = []
        
        for t in saved_tables:
            # 기본 이름과 즐겨찾기 상태 설정
            if 'name' not in t:
                t['name'] = t['created_at'].strftime('%Y-%m-%d 시간표')
            if 'is_favorite' not in t:
                t['is_favorite'] = False
            
            if t['is_favorite']:
                fav_tables.append(t)
            else:
                archive_tables.append(t)
        
        # [1] 즐겨찾기 (Quick Access)
        if fav_tables:
            st.markdown("##### ⭐ 즐겨찾기 (Quick Access)")
            cols = st.columns(4) # 한 줄에 4개씩
            for idx, table in enumerate(fav_tables):
                with cols[idx % 4]:
                    if st.button(f"📄 {table['name']}", key=f"fav_{table['id']}", use_container_width=True):
                        st.session_state.timetable_result = table['result']
                        st.session_state.current_timetable_meta = {
                            "id": table['id'],
                            "name": table['name'],
                            "is_favorite": table['is_favorite']
                        }
                        st.toast(f"'{table['name']}'을(를) 불러왔습니다.")
                        st.rerun()

        # [2] 보관함 (Archive) - Expander 안에 Grid 배치
        with st.expander("📂 내 시간표 보관함 (클릭하여 열기)", expanded=False):
            if not archive_tables:
                st.info("보관된 시간표가 없습니다.")
            else:
                cols = st.columns(4)
                for idx, table in enumerate(archive_tables):
                    with cols[idx % 4]:
                        if st.button(f"📄 {table['name']}", key=f"arc_{table['id']}", use_container_width=True):
                            st.session_state.timetable_result = table['result']
                            st.session_state.current_timetable_meta = {
                                "id": table['id'],
                                "name": table['name'],
                                "is_favorite": table['is_favorite']
                            }
                            st.toast(f"'{table['name']}'을(를) 불러왔습니다.")
                            st.rerun()

    # [메인 시간표 영역]
    timetable_area = st.empty()
    if st.session_state.timetable_result:
        with timetable_area.container():
            st.markdown("### 🗓️ 내 시간표")

            # [시간표 관리자 툴바] - 불러온 시간표가 있을 때만 표시
            current_meta = st.session_state.get("current_timetable_meta", {})
            if current_meta and st.session_state.user and fb_manager.is_initialized:
                with st.container(border=True):
                    c1, c2, c3 = st.columns([2, 1, 0.8])
                    new_name = c1.text_input("시간표 이름", value=current_meta.get('name', ''), label_visibility="collapsed", placeholder="시간표 이름 입력")
                    is_fav = c2.checkbox("⭐ 즐겨찾기 고정", value=current_meta.get('is_favorite', False))
                    
                    if c3.button("정보 수정 저장", use_container_width=True):
                        if fb_manager.update_data('timetables', current_meta['id'], {'name': new_name, 'is_favorite': is_fav}):
                            st.session_state.current_timetable_meta['name'] = new_name
                            st.session_state.current_timetable_meta['is_favorite'] = is_fav
                            st.toast("정보가 수정되었습니다. (즐겨찾기 이동 등은 새로고침 후 반영됩니다)", icon="✅")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("저장 실패")

            # --------------------------------------------------------------------------------
            # [수정] 표와 설명을 분리하여 중간에 강의계획서 뷰어 삽입
            # --------------------------------------------------------------------------------
            
            # HTML Table과 Description 분리
            full_result = st.session_state.timetable_result
            if "</table>" in full_result:
                parts = full_result.split("</table>", 1)
                table_part = parts[0] + "</table>"
                desc_part = parts[1]
            else:
                table_part = full_result
                desc_part = ""

            # 1. HTML 표 출력
            st.markdown(table_part, unsafe_allow_html=True)

            # 2. 강의계획서 감지 및 인페이지 뷰어 (중간 삽입)
            # 2-1. HTML에서 과목명 및 교수명 추출 (table_part에서만 추출)
            def extract_course_info(html_code):
                if not html_code: return []
                # Pattern: <b>Subject</b><br><small>Professor (Grade)</small>
                matches = re.findall(r"<b>(.*?)</b><br><small>(.*?)</small>", html_code)
                courses = []
                for subj, small_content in matches:
                    if "(" in small_content:
                        prof = small_content.split("(")[0].strip()
                    else:
                        prof = small_content.strip()
                    courses.append({"subject": subj.strip(), "professor": prof})
                return courses

            # 2-2. 파일 매칭 확인
            def match_syllabus_files(courses):
                matched_list = []
                if not os.path.exists("data/syllabus"):
                    return []
                seen = set()
                for c in courses:
                    subj = c['subject']
                    prof = c['professor']
                    key = f"{subj}_{prof}"
                    if key in seen: continue
                    seen.add(key)
                    
                    file_v1 = f"data/syllabus/{subj}_{prof}.txt"
                    file_v2 = f"data/syllabus/{subj}.txt"
                    
                    final_file = None
                    display_label = ""
                    
                    if os.path.exists(file_v1):
                        final_file = file_v1
                        display_label = f"{subj} ({prof})"
                    elif os.path.exists(file_v2):
                        final_file = file_v2
                        display_label = f"{subj}"
                        
                    if final_file:
                        matched_list.append({
                            "subject": subj,
                            "file_path": final_file,
                            "display_label": display_label
                        })
                return matched_list

            # 2-3. 뷰어 선택 콜백 함수
            def set_syllabus_viewer(file_path, display_label):
                st.session_state.selected_syllabus = {"path": file_path, "label": display_label}

            # 2-4. UI 렌더링 (버튼 및 인페이지 뷰어)
            extracted_courses = extract_course_info(table_part)
            matched_courses = match_syllabus_files(extracted_courses)

            if matched_courses:
                st.divider()
                st.markdown("##### 📚 강의계획서 확인")
                # 버튼 나열
                cols = st.columns(len(matched_courses) + 2)
                for i, match in enumerate(matched_courses):
                    cols[i].button(
                        f"📄 {match['display_label']}", 
                        key=f"btn_syl_{i}",
                        on_click=set_syllabus_viewer,
                        args=(match['file_path'], match['display_label'])
                    )
                
                # [인페이지 뷰어] 선택된 강의계획서가 있으면 아래에 바로 표시
                if st.session_state.selected_syllabus:
                    with st.container(border=True):
                        c1, c2 = st.columns([8, 1])
                        c1.subheader(f"📄 {st.session_state.selected_syllabus['label']}")
                        if c2.button("❌ 닫기", key="close_syl_viewer"):
                            st.session_state.selected_syllabus = None
                            st.rerun()

                        try:
                            with open(st.session_state.selected_syllabus['path'], "r", encoding="utf-8") as f:
                                full_text = f.read()
                            st.text_area("강의계획서 원문", full_text, height=400, disabled=True)
                        except Exception as e:
                            st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
                st.divider()

            # 3. 나머지 설명 텍스트 출력
            if desc_part:
                st.markdown(desc_part, unsafe_allow_html=True)
            
            # --------------------------------------------------------------------------------

            # [신규 저장 버튼] - 불러온 게 아니라 새로 만든 경우 or 복사본 저장
            if st.session_state.user and fb_manager.is_initialized:
                st.caption("현재 보고 있는 시간표를 **새로운 항목**으로 저장하려면 아래 버튼을 누르세요.")
                if st.button("☁️ 현재 시간표를 새 이름으로 저장"):
                    current_major = st.session_state.get("tt_major", "학과미정")
                    current_grade = st.session_state.get("tt_grade", "")
                    
                    # 저장할 데이터
                    doc_data = {
                        "result": st.session_state.timetable_result,
                        "major": current_major,
                        "grade": current_grade,
                        "name": f"{current_major} {current_grade} (새 시간표)", # 기본 이름
                        "is_favorite": False,
                        "created_at": datetime.datetime.now()
                    }
                    doc_id = str(int(time.time()))
                    if fb_manager.save_data('timetables', doc_id, doc_data):
                        # 저장 후 메타데이터 업데이트 (바로 관리 가능하도록)
                        st.session_state.current_timetable_meta = {
                            "id": doc_id,
                            "name": doc_data["name"],
                            "is_favorite": False
                        }
                        st.toast("시간표가 저장되었습니다!", icon="✅")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.toast("저장 실패", icon="❌")
            st.divider()

    with st.expander("시간표 설정 열기/닫기", expanded=not bool(st.session_state.timetable_result)):
        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.markdown("#### 1️⃣ 기본 정보")
            # [기존 학과 리스트 유지]
            kw_departments = [
    # 전자정보공과대학
    "전자공학과", "전자통신공학과", "전자융합공학과", "전기공학과", "전자재료공학과", "반도체시스템공학부", "로봇학부",
    # 인공지능융합대학
    "컴퓨터정보공학부", "소프트웨어학부", "정보융합학부", "지능형로봇학과",
    # 공과대학
    "건축학과", "건축공학과", "화학공학과", "환경공학과",
    # 자연과학대학
    "수학과", "전자바이오물리학과", "화학과", "스포츠융합과학과", "정보콘텐츠학과",
    # 인문사회과학대학
    "국어국문학과", "영어산업학과", "미디어커뮤니케이션학부", "산업심리학과", "동북아문화산업학부",
    # 정책법학대학
    "행정학과", "법학부", "국제학부", "자산관리학과",
    # 경영대학
    "경영학부", "국제통상학부",
    # 참빛인재대학 (재직자)
    "금융부동산법무학과", "게임콘텐츠학과", "스마트전기전자학과", "스포츠상담재활학과",
    # 자율전공 및 기타
    "자율전공학부(자연)", "자율전공학부(인문)", "인제니움학부대학"
]
            major = st.selectbox("학과", kw_departments, key="tt_major")
            c1, c2 = st.columns(2)
            grade = c1.selectbox("학년", ["1학년", "2학년", "3학년", "4학년"], key="tt_grade")
            semester = c2.selectbox("학기", ["1학기", "2학기"], key="tt_semester")
            target_credit = st.number_input("목표 학점", 9, 24, 18, key="tt_credit")
            
            # [수정] 성적/진단 결과 반영 체크박스 추가
            use_diagnosis = st.checkbox("☑️ 성적/진로 진단 결과 반영하기 (재수강, 직무 추천 등)", value=True, key="tt_use_diag")
            requirements = st.text_area("추가 요구사항", placeholder="예: 전공 필수 챙겨줘", key="tt_req")

        with col2:
            st.markdown("#### 2️⃣ 공강 시간 설정")
            st.info("✅ **체크된 시간**: 수업 가능 (기본)  \n⬜ **체크 해제**: 공강 (수업 배정 안 함)")
            kw_times = {
                "1교시": "09:00~10:15", "2교시": "10:30~11:45", "3교시": "12:00~13:15",
                "4교시": "13:30~14:45", "5교시": "15:00~16:15", "6교시": "16:30~17:45",
                "7교시": "18:00~19:15", "8교시": "19:25~20:40", "9교시": "20:50~22:05"
            }
            schedule_index = [f"{k} ({v})" for k, v in kw_times.items()]
            if "init_schedule_df" not in st.session_state:
                st.session_state.init_schedule_df = pd.DataFrame(True, index=schedule_index, columns=["월", "화", "수", "목", "금"])
            edited_schedule = st.data_editor(
                st.session_state.init_schedule_df,
                column_config={
                    "월": st.column_config.CheckboxColumn("월", default=True),
                    "화": st.column_config.CheckboxColumn("화", default=True),
                    "수": st.column_config.CheckboxColumn("수", default=True),
                    "목": st.column_config.CheckboxColumn("목", default=True),
                    "금": st.column_config.CheckboxColumn("금", default=True),
                },
                height=360,
                use_container_width=True,
                key="tt_editor"
            )

        if st.button("시간표 생성하기 ✨", type="primary", use_container_width=True):
            blocked_times = []
            for day in ["월", "화", "수", "목", "금"]:
                for idx, period_label in enumerate(edited_schedule.index):
                    if not edited_schedule.iloc[idx][day]:
                        blocked_times.append(f"{day}요일 {period_label}")
            blocked_desc = ", ".join(blocked_times) if blocked_times else "없음"
            
            # [수정] 데이터 파이프라인 (Silent Fetch 포함)
            diagnosis_context = ""
            if use_diagnosis:
                # 1. 현재 세션에 진단 결과가 있으면 사용
                if st.session_state.graduation_analysis_result:
                    diagnosis_context = st.session_state.graduation_analysis_result
                # 2. 없지만 로그인 유저라면 DB에서 Silent Fetch
                elif st.session_state.user and fb_manager.is_initialized:
                    saved_diags = fb_manager.load_collection('graduation_diagnosis')
                    if saved_diags:
                        diagnosis_context = saved_diags[0]['result'] # 가장 최근 결과
                        st.toast("최근 저장된 진단 결과를 불러와 반영했습니다.", icon="✅")

            with st.spinner("선수과목 확인 및 시간표 조합 중... (최대 1분 소요될 수 있습니다)"):
                # [수정] diagnosis_context 전달
                result = generate_timetable_ai(major, grade, semester, target_credit, blocked_desc, requirements, diagnosis_context)
                st.session_state.timetable_result = result
                st.session_state.timetable_chat_history = []
                # 새로 생성했으므로 메타데이터 초기화 (저장 전)
                st.session_state.current_timetable_meta = {} 
                add_log("user", f"[시간표] {major} {grade} 생성", "📅 스마트 시간표(수정가능)")
                st.rerun()

    if st.session_state.timetable_result:
        st.subheader("💬 시간표 상담소")
        st.caption("시간표에 대해 질문하거나(Q&A), 수정을 요청(Refine)하세요.")
        for msg in st.session_state.timetable_chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"], unsafe_allow_html=True)

        if chat_input := st.chat_input("예: 1교시 빼줘, 또는 대학수학1 꼭 들어야 해?"):
            st.session_state.timetable_chat_history.append({"role": "user", "content": chat_input})
            add_log("user", f"[상담] {chat_input}", "📅 스마트 시간표(수정가능)")
            with st.chat_message("user"):
                st.write(chat_input)
            with st.chat_message("assistant"):
                with st.spinner("분석 중..."):
                    response = chat_with_timetable_ai(st.session_state.timetable_result, chat_input, major, grade, semester)
                    if "[수정]" in response:
                        new_timetable = response.replace("[수정]", "").strip()
                        new_timetable = clean_html_output(new_timetable) 
                        st.session_state.timetable_result = new_timetable
                        with timetable_area.container():
                            st.markdown("### 🗓️ 내 시간표")
                            # 수정 시 관리자 도구 유지
                            current_meta = st.session_state.get("current_timetable_meta", {})
                            if current_meta and st.session_state.user and fb_manager.is_initialized:
                                with st.container(border=True):
                                    c1, c2, c3 = st.columns([2, 1, 0.8])
                                    new_name = c1.text_input("시간표 이름", value=current_meta.get('name', ''), label_visibility="collapsed")
                                    is_fav = c2.checkbox("⭐ 즐겨찾기 고정", value=current_meta.get('is_favorite', False))
                                    if c3.button("정보 수정 저장", use_container_width=True):
                                         if fb_manager.update_data('timetables', current_meta['id'], {'name': new_name, 'is_favorite': is_fav}):
                                            st.session_state.current_timetable_meta['name'] = new_name
                                            st.session_state.current_timetable_meta['is_favorite'] = is_fav
                                            st.rerun()

                            st.markdown(new_timetable, unsafe_allow_html=True)
                            st.divider()
                        success_msg = "시간표를 수정했습니다. 위쪽 표가 업데이트 되었습니다."
                        st.write(success_msg)
                        st.session_state.timetable_chat_history.append({"role": "assistant", "content": success_msg})
                    else:
                        clean_response = response.replace("[답변]", "").strip()
                        st.markdown(clean_response)
                        st.session_state.timetable_chat_history.append({"role": "assistant", "content": clean_response})

elif st.session_state.current_menu == "📈 성적 및 진로 진단":
    st.subheader("📈 성적 및 진로 정밀 진단")
    st.markdown("""
    **취득 학점 내역을 캡처해서 업로드하세요!** AI 취업 컨설턴트가 당신의 성적표를 냉철하게 분석하여 **졸업 요건**, **성적 상태**, **커리어 방향성**을 진단해 드립니다.
    - KLAS 또는 학교 포털의 성적/학점 조회 화면을 캡처해주세요.
    """)

    if st.session_state.user and fb_manager.is_initialized:
        with st.expander("📂 저장된 진단 결과 불러오기"):
            saved_diags = fb_manager.load_collection('graduation_diagnosis')
            if saved_diags:
                selected_diag = st.selectbox("불러올 진단 선택", 
                                             saved_diags, 
                                             format_func=lambda x: datetime.datetime.fromtimestamp(int(x['id'])).strftime('%Y-%m-%d %H:%M'))
                if st.button("진단 결과 불러오기"):
                    st.session_state.graduation_analysis_result = selected_diag['result']
                    st.success("진단 결과를 불러왔습니다!")
                    st.rerun()

    uploaded_files = st.file_uploader("캡처 이미지 업로드 (여러 장 가능)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        if st.button("진단 시작 🚀", type="primary"):
            with st.spinner("성적표를 독해하고 분석 중입니다... (냉철한 평가가 준비되고 있습니다)"):
                analysis_result = analyze_graduation_requirements(uploaded_files)
                st.session_state.graduation_analysis_result = analysis_result
                st.session_state.graduation_chat_history = []
                add_log("user", "[진단] 이미지 분석 요청", "📈 성적 및 진로 진단")
                st.rerun()

    if st.session_state.graduation_analysis_result:
        st.divider()
        
        result_text = st.session_state.graduation_analysis_result
        
        # 섹션 파싱
        sec_grad = ""
        sec_grade = ""
        sec_career = ""
        
        try:
            if "[[SECTION:GRADUATION]]" in result_text:
                parts = result_text.split("[[[SECTION:GRADUATION]]")
                if len(parts) > 1:
                    temp = parts[1]
                else:
                    # [[SECTION:GRADUATION]] 태그가 맨 앞에 있거나 split이 제대로 안된 경우
                    # 혹시 모르니 그냥 result_text에서 찾기 시도
                    temp = result_text.split("[[SECTION:GRADUATION]]")[-1]

                if "[[SECTION:GRADES]]" in temp:
                    sec_grad, remaining = temp.split("[[SECTION:GRADES]]")
                    if "[[SECTION:CAREER]]" in remaining:
                        sec_grade, sec_career = remaining.split("[[SECTION:CAREER]]")
                    else:
                        sec_grade = remaining
                else:
                    sec_grad = temp
            else:
                sec_grad = result_text
        except:
            sec_grad = result_text

        tab1, tab2, tab3 = st.tabs(["🎓 졸업 요건 확인", "📊 성적 정밀 분석", "💼 AI 커리어 솔루션"])
        
        with tab1:
            st.markdown(sec_grad)
        with tab2:
            st.markdown(sec_grade if sec_grade else "성적 분석 결과가 없습니다.")
        with tab3:
            st.markdown(sec_career if sec_career else "커리어 솔루션 결과가 없습니다.")
        
        st.divider()

        if st.session_state.user and fb_manager.is_initialized:
            if st.button("☁️ 진단 결과 저장하기"):
                doc_data = {
                    "result": st.session_state.graduation_analysis_result,
                    "created_at": datetime.datetime.now()
                }
                doc_id = str(int(time.time()))
                if fb_manager.save_data('graduation_diagnosis', doc_id, doc_data):
                    st.toast("진단 결과가 저장되었습니다!", icon="✅")
        
        st.subheader("💬 컨설턴트와의 대화")
        st.caption("결과에 대해 추가 질문을 하거나, 누락된 정보를 알려주세요.")

        for msg in st.session_state.graduation_chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if chat_input := st.chat_input("질문이나 추가 정보를 입력하세요"):
            st.session_state.graduation_chat_history.append({"role": "user", "content": chat_input})
            add_log("user", f"[진단상담] {chat_input}", "📈 성적 및 진로 진단")
            with st.chat_message("user"):
                st.write(chat_input)
            
            with st.chat_message("assistant"):
                with st.spinner("분석 중..."):
                    response = chat_with_graduation_ai(st.session_state.graduation_analysis_result, chat_input)
                    if "[수정]" in response:
                        new_result = response.replace("[수정]", "").strip()
                        st.session_state.graduation_analysis_result = new_result
                        success_msg = "정보를 반영하여 진단 결과를 업데이트했습니다. 위쪽 탭을 다시 확인해주세요."
                        st.session_state.graduation_chat_history.append({"role": "assistant", "content": success_msg})
                        st.rerun()
                    else:
                        st.markdown(response)
                        st.session_state.graduation_chat_history.append({"role": "assistant", "content": response})

        if st.button("결과 초기화"):
            st.session_state.graduation_analysis_result = ""
            st.session_state.graduation_chat_history = []
            st.rerun()
