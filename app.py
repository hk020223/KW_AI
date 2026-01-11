import streamlit as st
import pandas as pd
import os
import glob
import datetime
import time
import base64
import re  # [추가] HTML 파싱을 위한 정규표현식 라이브러리
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

# 세션 상태 초기화
if "global_log" not in st.session_state: st.session_state.global_log = [] 
if "timetable_result" not in st.session_state: st.session_state.timetable_result = "" 
if "chat_history" not in st.session_state: st.session_state.chat_history = [] 
if "current_menu" not in st.session_state: st.session_state.current_menu = "🤖 AI 학사 지식인"
if "menu_radio" not in st.session_state: st.session_state["menu_radio"] = "🤖 AI 학사 지식인"
if "timetable_chat_history" not in st.session_state: st.session_state.timetable_chat_history = []
if "graduation_analysis_result" not in st.session_state: st.session_state.graduation_analysis_result = ""
if "graduation_chat_history" not in st.session_state: st.session_state.graduation_chat_history = []
if "user" not in st.session_state: st.session_state.user = None

def add_log(role, content, menu_context=None):
    timestamp = datetime.datetime.now().strftime("%H:%M")
    st.session_state.global_log.append({
        "role": role, "content": content, "time": timestamp, "menu": menu_context
    })

def clean_html_output(text):
    cleaned = text.strip()
    if cleaned.startswith("```html"): cleaned = cleaned[7:]
    elif cleaned.startswith("```"): cleaned = cleaned[3:]
    if cleaned.endswith("```"): cleaned = cleaned[:-3]
    return cleaned.replace("```html", "").replace("```", "").strip()

def run_with_retry(func, *args, **kwargs):
    max_retries = 5
    delays = [1, 2, 4, 8, 16]
    for i in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                if i < max_retries - 1:
                    time.sleep(delays[i])
                    continue
            raise e

# -----------------------------------------------------------------------------
# [Firebase Manager]
# -----------------------------------------------------------------------------
class FirebaseManager:
    def __init__(self):
        self.db = None
        self.is_initialized = False
        self.init_firestore()

    def init_firestore(self):
        if "firebase_service_account" in st.secrets:
            try:
                if not firebase_admin._apps:
                    cred_info = dict(st.secrets["firebase_service_account"])
                    cred = credentials.Certificate(cred_info)
                    firebase_admin.initialize_app(cred)
                self.db = firestore.client()
                self.is_initialized = True
            except Exception: pass

    def login(self, email, password):
        if not self.is_initialized: return None, "Firebase 연결 실패"
        try:
            users_ref = self.db.collection('users')
            query = users_ref.where('email', '==', email).where('password', '==', password).stream()
            for doc in query:
                user_data = doc.to_dict()
                user_data['localId'] = doc.id
                return user_data, None
            return None, "이메일/비번 불일치"
        except Exception as e: return None, str(e)

    def signup(self, email, password):
        if not self.is_initialized: return None, "Firebase 연결 실패"
        try:
            users_ref = self.db.collection('users')
            if list(users_ref.where('email', '==', email).stream()):
                return None, "이미 가입된 이메일"
            new_user_ref = users_ref.document()
            user_data = {"email": email, "password": password, "created_at": firestore.SERVER_TIMESTAMP}
            new_user_ref.set(user_data)
            user_data['localId'] = new_user_ref.id
            return user_data, None
        except Exception as e: return None, str(e)

    def save_data(self, collection, doc_id, data):
        if not self.is_initialized or not st.session_state.user: return False
        try:
            uid = st.session_state.user['localId']
            self.db.collection('users').document(uid).collection(collection).document(doc_id).set(data)
            return True
        except: return False

    def load_collection(self, collection):
        if not self.is_initialized or not st.session_state.user: return []
        try:
            uid = st.session_state.user['localId']
            docs = self.db.collection('users').document(uid).collection(collection).order_by('created_at', direction=firestore.Query.DESCENDING).stream()
            return [{"id": d.id, **d.to_dict()} for d in docs]
        except: return []

fb_manager = FirebaseManager()

# 전처리된 TXT 파일 로드 (전체 지식 로드용)
@st.cache_resource(show_spinner="강의 데이터를 메모리에 적재 중...")
def load_knowledge_base():
    if not os.path.exists("data/processed"): return ""
    txt_files = glob.glob("data/processed/*.txt")
    if not txt_files: return ""
    
    all_content = ""
    for txt_file in txt_files:
        try:
            with open(txt_file, "r", encoding="utf-8") as f:
                content = f.read()
            filename = os.path.basename(txt_file)
            all_content += f"\n\n--- [문서: {filename}] ---\n{content}"
        except Exception as e:
            print(f"Error loading {txt_file}: {e}")
    return all_content

PRE_LEARNED_DATA = load_knowledge_base()

# -----------------------------------------------------------------------------
# [AI 엔진]
# -----------------------------------------------------------------------------
def get_llm():
    if not api_key: return None
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-09-2025", temperature=0)

def ask_ai(question):
    llm = get_llm()
    if not llm: return "⚠️ API Key 오류"
    def _execute():
        chain = PromptTemplate.from_template(
            "문서 내용: {context}\n질문: {question}\n문서에 기반해 답변해줘."
        ) | llm
        return chain.invoke({"context": PRE_LEARNED_DATA, "question": question}).content
    return run_with_retry(_execute)

# 시간표 생성 AI
def generate_timetable_ai(major, grade, semester, target_credits, blocked_times, requirements, diagnosis_context=None):
    llm = get_llm()
    def _execute():
        base_template = """
        너는 대학교 수강신청 전문가야. [학습된 문서]에 기반하여 시간표를 짜줘.
        [학생 정보]
        - 소속: {major} / {grade} {semester}
        - 목표: {target_credits}학점
        - 공강 필수: {blocked_times}
        - 요구사항: {requirements}
        """
        if diagnosis_context:
            base_template += f"\n[성적 진단 결과]\n{diagnosis_context}\n[우선순위] 1.필수과목 2.재수강/미이수 3.직무추천"
        
        base_template += """
        [지시사항]
        - 1교시~9교시, 월~금 형식의 **HTML Table**로 출력해.
        - 요일별 교시를 정확히 분리해 (월3,수4 -> 월요일3교시, 수요일4교시).
        - **HTML 코드만 출력해 (```html 태그 없이).**
        [학습된 문서]
        {context}
        """
        prompt = PromptTemplate(template=base_template, input_variables=["context","major","grade","semester","target_credits","blocked_times","requirements"])
        chain = prompt | llm
        return chain.invoke({
            "context": PRE_LEARNED_DATA, "major": major, "grade": grade, 
            "semester": semester, "target_credits": target_credits, 
            "blocked_times": blocked_times, "requirements": requirements
        }).content
    return clean_html_output(run_with_retry(_execute))

# -----------------------------------------------------------------------------
# [UI 구성]
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🗂️ 활동 로그")
    if st.session_state.user:
        st.info(f"👤 {st.session_state.user['email']}")
        if st.button("로그아웃"):
            st.session_state.clear()
            st.rerun()
    else:
        with st.expander("🔐 로그인", expanded=True):
            email = st.text_input("이메일")
            pw = st.text_input("비번", type="password")
            if st.button("로그인"):
                u, e = fb_manager.login(email, pw)
                if u: st.session_state.user = u; st.rerun()
            if st.button("회원가입"):
                u, e = fb_manager.signup(email, pw)
                if u: st.session_state.user = u; st.rerun()

    if PRE_LEARNED_DATA: st.success("✅ 학습 데이터 로드 완료")
    else: st.error("⚠️ 데이터 없음 (전처리 필요)")

menu = st.radio("메뉴", ["🤖 AI 지식인", "📅 스마트 시간표", "📈 성적 진단"], horizontal=True, key="menu_radio")
if menu != st.session_state.current_menu:
    st.session_state.current_menu = menu
    st.rerun()

st.divider()

# 1. 지식인
if st.session_state.current_menu == "🤖 AI 지식인":
    st.subheader("🤖 무엇이든 물어보세요 (강의계획서/규정 기반)")
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])
    if q := st.chat_input("질문 입력"):
        st.session_state.chat_history.append({"role":"user","content":q})
        with st.chat_message("user"): st.markdown(q)
        with st.chat_message("assistant"):
            ans = ask_ai(q)
            st.markdown(ans)
        st.session_state.chat_history.append({"role":"assistant","content":ans})

# 2. 스마트 시간표
elif st.session_state.current_menu == "📅 스마트 시간표":
    st.subheader("📅 AI 맞춤형 시간표")
    
    # [1] 시간표 결과 표시 (최상단)
    if st.session_state.timetable_result:
        st.markdown("### 🗓️ 생성된 시간표")
        st.markdown(st.session_state.timetable_result, unsafe_allow_html=True)
        
        # [★핵심 기능] 스마트 강의계획서 매칭 (자동 인식)
        # HTML에서 과목명 추출 (<b>과목명</b> 패턴)
        extracted_courses = re.findall(r"<b>(.*?)</b>", st.session_state.timetable_result)
        # 중복 제거
        extracted_courses = list(set(extracted_courses))

        if extracted_courses and os.path.exists("data/processed"):
            matched_files = {}
            processed_files = glob.glob("data/processed/*.txt")
            
            # 파일 매칭 로직 (과목명이 파일명에 포함되면 매칭)
            for course in extracted_courses:
                for f in processed_files:
                    if course.replace(" ", "") in os.path.basename(f).replace(" ", ""): # 공백 무시 비교
                        matched_files[course] = f
                        break
            
            if matched_files:
                st.divider()
                st.markdown("#### 📄 내 시간표 강의계획서 (자동 매칭)")
                st.caption("시간표에 포함된 과목 중, 상세 정보가 확인된 강의입니다. 클릭해서 펼쳐보세요.")
                
                # 매칭된 과목별로 Expander 생성
                for course_name, file_path in matched_files.items():
                    with st.expander(f"📘 **{course_name}** 강의계획서 상세 보기"):
                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                content = f.read()
                            st.markdown(content) # 전처리된 마크다운을 그대로 렌더링 (즉시 로딩)
                        except Exception as e:
                            st.error(f"파일을 읽는 중 오류 발생: {e}")
            else:
                st.divider()
                st.info("ℹ️ 시간표에 포함된 과목의 상세 강의계획서 정보가 없습니다.")
        
        st.divider()

    # [2] 설정 패널
    with st.expander("⚙️ 시간표 설정", expanded=not bool(st.session_state.timetable_result)):
        c1, c2 = st.columns(2)
        major = c1.selectbox("학과", ["전자융합공학과", "컴퓨터정보공학부", "소프트웨어학부", "정보융합학부"], key="tt_major")
        grade = c2.selectbox("학년", ["1학년", "2학년", "3학년", "4학년"], key="tt_grade")
        use_diag = st.checkbox("☑️ 성적 진단 결과 반영 (재수강/직무 추천)", value=True)
        
        if st.button("시간표 생성 🚀", type="primary", use_container_width=True):
            diag_ctx = ""
            if use_diag and st.session_state.user:
                saved = fb_manager.load_collection('graduation_diagnosis')
                if saved: diag_ctx = saved[0]['result']
            
            with st.spinner("AI가 최적의 시간표를 설계 중입니다..."):
                res = generate_timetable_ai(major, grade, "1학기", 18, "없음", "전공필수 위주", diag_ctx)
                st.session_state.timetable_result = res
                st.rerun()

    # [3] 시간표 상담 채팅 (하단 배치)
    if st.session_state.timetable_result:
        st.markdown("#### 💬 시간표 상담소")
        for msg in st.session_state.timetable_chat_history:
            with st.chat_message(msg["role"]): st.markdown(msg["content"], unsafe_allow_html=True)
        if chat_input := st.chat_input("예: 1교시 빼줘"):
            st.session_state.timetable_chat_history.append({"role":"user","content":chat_input})
            # (상담 로직은 단순화를 위해 ask_ai 사용, 필요 시 chat_with_timetable_ai 복원 가능)
            with st.chat_message("user"): st.write(chat_input)
            with st.chat_message("assistant"):
                ans = ask_ai(f"현재 시간표: {st.session_state.timetable_result}\n사용자 요청: {chat_input}\n시간표를 수정하거나 질문에 답해줘.")
                st.markdown(ans)
                st.session_state.timetable_chat_history.append({"role":"assistant","content":ans})

# 3. 성적 진단 (기존 유지)
elif st.session_state.current_menu == "📈 성적 진단":
    st.subheader("📈 성적 및 진로 진단")
    files = st.file_uploader("성적표 업로드", accept_multiple_files=True)
    if files and st.button("진단 시작"):
        # (이미지 분석 로직 생략)
        st.session_state.graduation_analysis_result = "진단 결과 예시입니다." 
        st.rerun()
    
    if st.session_state.graduation_analysis_result:
        st.markdown(st.session_state.graduation_analysis_result)
