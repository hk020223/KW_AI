import streamlit as st
import pandas as pd
import os
import glob
import datetime
import time
import base64
import json
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# LangChain & AI
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun

# Firebase
import firebase_admin
from firebase_admin import credentials, firestore

# -----------------------------------------------------------------------------
# [0] 설정 및 상수 정의
# -----------------------------------------------------------------------------
st.set_page_config(page_title="KW-강의마스터 Pro", page_icon="🎓", layout="wide")

# 광운대학교 전체 학과 리스트 (상수)
ALL_DEPARTMENTS = [
    "전자융합공학과", "전자공학과", "전자통신공학과", "전기공학과", "전자재료공학과", "로봇학부",
    "소프트웨어학부", "컴퓨터정보공학부", "정보융합학부",
    "건축학과", "건축공학과", "화학공학과", "환경공학과",
    "수학과", "전자바이오물리학과", "화학과", "스포츠융합과학과",
    "국어국문학과", "영어산업학과", "미디어커뮤니케이션학부", "산업심리학과", "동북아문화산업학부",
    "행정학과", "법학부", "국제학부", "경영학부", "국제통상학부"
]
ALL_DEPARTMENTS.sort()

# [복구] 상세 시간표 생성 지침 (정확도 향상)
COMMON_TIMETABLE_INSTRUCTION = """
[★★★ 핵심 알고리즘: 3단계 검증 및 필터링 (Strict Verification) ★★★]

1. **Step 1: 요람(Curriculum) 기반 '수강 대상' 리스트 확정**:
   - 먼저 PDF 요람 문서에서 **'{major} {grade} {semester}'**에 배정된 **'표준 이수 과목' 목록**을 추출하세요.
   - **주의:** 'MSC 필수', '공학인증 필수'라고 적혀 있어도, 이 학기(예: 1학년 1학기) 표에 없으면 리스트에 넣지 마세요.

2. **Step 2: 학년 정합성 검사 (Grade Validation)**:
   - 추출된 과목이 실제 시간표 데이터에서 몇 학년 대상으로 개설되었는지 확인하세요.
   - **사용자가 선택한 학년({grade})과 시간표의 대상 학년이 일치하지 않으면 과감히 제외하세요.**
   - (예: 사용자가 1학년인데, 시간표에 '2학년' 대상이라고 적혀있으면 배치 금지)

3. **Step 3: 시간표 데이터와 정밀 대조 (Exact Match)**:
   - 위 단계를 통과한 과목만 시간표에 배치하세요.
   - **과목명 완전 일치 필수**: 예: '대학물리학1' vs '대학물리및실험1' 구분.

4. **출력 형식 (세로형 HTML Table)**:
   - 반드시 **HTML `<table>` 태그**를 사용해라.
   - **행(Row): 1교시 ~ 9교시** (행 머리글에 시간 포함: 1교시 (09:00~10:15) 등)
   - **열(Column): 월, 화, 수, 목, 금, 토, 일** (7일 모두 표시)
   - **스타일 규칙**:
     - `table` 태그에 `width="100%"` 속성을 주어라.
     - **같은 과목은 반드시 같은 배경색**을 사용해라. (파스텔톤 권장)
     - **수업이 없는 빈 시간(공강)은 반드시 흰색 배경**으로 둬라.
     - 셀 내용: `<b>과목명</b><br><small>교수명 (대상학년)</small>`

5. **온라인 및 원격 강의 처리 (필수 - 표 내부에 포함)**:
   - 강의 시간이 **'온라인', '원격', 'Cyber', '시간 미지정'** 등이면 **시간표 표(Table)의 맨 마지막 행에 추가**하세요.
   - **행 제목:** `<b>온라인/기타</b>`
   - **내용:** 해당되는 모든 과목을 `<b>과목명</b>(교수명)` 형식으로 나열하세요. (요일 열은 합치거나(colspan) 적절히 분배하여 표시)
   - **절대 표 밖으로 빼지 말고, 테이블의 일부로 포함시키세요.**

6. **출력 순서 고정**:
   - 1순위: HTML 시간표 표 (온라인 강의 포함)
   - 2순위: "### ✅ 필수 과목 검증 및 학년 일치 확인" (각 과목별로 '대상 학년'이 맞는지 명시)
   - 3순위: "### ⚠️ 배치 실패/제외 목록" (학년 불일치로 제외된 과목 포함)
"""

# CSS 스타일
st.markdown("""
    <style>
        footer { visibility: hidden; }
        div.row-widget.stRadio > div { flex-direction: row; align-items: stretch; }
        div.row-widget.stRadio > div[role="radiogroup"] > label {
            background-color: #f0f2f6; padding: 10px 20px; border-radius: 10px; margin-right: 10px; border: 1px solid #e0e0e0; cursor: pointer; transition: all 0.3s;
        }
        div.row-widget.stRadio > div[role="radiogroup"] > label[data-checked="true"] {
            background-color: #ff4b4b; color: white; border-color: #ff4b4b;
        }
        .stTabs [data-baseweb="tab-list"] { gap: 10px; }
        .stTabs [data-baseweb="tab"] { height: 40px; white-space: pre-wrap; border-radius: 4px; gap: 1px; padding-top: 5px; padding-bottom: 5px; }
        @media only screen and (max-width: 600px) {
            .main .block-container { padding-top: 2rem !important; }
            div[data-testid="stMarkdownContainer"] table { font-size: 10px !important; }
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
if "user" not in st.session_state: st.session_state.user = None
if "global_log" not in st.session_state: st.session_state.global_log = []
if "shared_context" not in st.session_state: st.session_state.shared_context = "" 
if "grade_json_data" not in st.session_state: st.session_state.grade_json_data = None
if "graduation_json_data" not in st.session_state: st.session_state.graduation_json_data = None 
if "graduation_analysis_result" not in st.session_state: st.session_state.graduation_analysis_result = "" # 텍스트 리포트용
if "timetable_result" not in st.session_state: st.session_state.timetable_result = ""
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "timetable_chat_history" not in st.session_state: st.session_state.timetable_chat_history = []
if "graduation_chat_history" not in st.session_state: st.session_state.graduation_chat_history = []
if "bookmarks" not in st.session_state: st.session_state.bookmarks = [] 
if "current_menu" not in st.session_state: st.session_state.current_menu = "📈 성적 및 진로 진단"

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

    def auth_user(self, email, password, mode="login"):
        if "FIREBASE_WEB_API_KEY" not in st.secrets: return None, "API Key 설정 필요"
        api_key_fb = st.secrets["FIREBASE_WEB_API_KEY"].strip()
        endpoint = "signInWithPassword" if mode == "login" else "signUp"
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:{endpoint}?key={api_key_fb}"
        payload = {"email": email, "password": password, "returnSecureToken": True}
        try:
            res = requests.post(url, json=payload)
            data = res.json()
            if "error" in data: return None, data["error"]["message"]
            return data, None
        except Exception as e: return None, str(e)

    def save_user_data(self, collection, doc_id, data):
        if not self.is_initialized or not st.session_state.user: return False
        try:
            user_id = st.session_state.user['localId']
            data['updated_at'] = firestore.SERVER_TIMESTAMP
            self.db.collection('users').document(user_id).collection(collection).document(doc_id).set(data)
            return True
        except: return False
    
    def load_user_data(self, collection, doc_id):
        if not self.is_initialized or not st.session_state.user: return None
        try:
            user_id = st.session_state.user['localId']
            doc = self.db.collection('users').document(user_id).collection(collection).document(doc_id).get()
            return doc.to_dict() if doc.exists else None
        except: return None

    def add_bookmark(self, question, answer, tag):
        if not self.is_initialized or not st.session_state.user: return False
        try:
            user_id = st.session_state.user['localId']
            data = {"question": question, "answer": answer, "tag": tag, "created_at": firestore.SERVER_TIMESTAMP}
            self.db.collection('users').document(user_id).collection('bookmarks').add(data)
            return True
        except: return False

    def load_bookmarks(self):
        if not self.is_initialized or not st.session_state.user: return []
        try:
            user_id = st.session_state.user['localId']
            docs = self.db.collection('users').document(user_id).collection('bookmarks').order_by('created_at', direction=firestore.Query.DESCENDING).stream()
            return [{"id": doc.id, **doc.to_dict()} for doc in docs]
        except: return []

fb_manager = FirebaseManager()

# -----------------------------------------------------------------------------
# [AI 엔진] - 모델명 수정 (gemini-1.5-flash)
# -----------------------------------------------------------------------------
def get_llm(): 
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=api_key)

def get_pro_llm(): 
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=api_key)

@st.cache_resource
def load_knowledge_base():
    if not os.path.exists("data"): return ""
    pdf_files = glob.glob("data/*.pdf")
    content = ""
    for f in pdf_files:
        try: content += f"\n\n--- [{os.path.basename(f)}] ---\n" + "".join([p.page_content for p in PyPDFLoader(f).load()])
        except: pass
    return content

PRE_LEARNED_DATA = load_knowledge_base()

def clean_json_output(text):
    text = text.strip()
    if text.startswith("```json"): 
        text = text[7:]
    elif text.startswith("```"): 
        text = text[3:]
    if text.endswith("```"): 
        text = text[:-3]
    return text.strip()

def clean_html_output(text):
    cleaned = text.strip()
    if cleaned.startswith("```html"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.replace("```html", "").replace("```", "").strip()

# -----------------------------------------------------------------------------
# [기능 1] 성적표 분석 (JSON)
# -----------------------------------------------------------------------------
def analyze_grades_structure(uploaded_images):
    llm = get_pro_llm()
    image_messages = []
    for img_file in uploaded_images:
        img_file.seek(0)
        b64 = base64.b64encode(img_file.read()).decode("utf-8")
        image_messages.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    
    prompt = """
    성적표 이미지를 분석하여 **반드시 유효한 JSON 형식**으로만 출력하세요. 마크다운 금지.
    {
        "student_info": {"admission_year": "2024", "major": "전자공학과"},
        "courses": [{"year": "2024", "semester": "1", "type": "전필", "name": "회로이론1", "grade": "A+", "score": 4.5}, ...],
        "strength_keywords": ["회로설계", "임베디드"],
        "weakness_analysis": "전공 기초는 튼튼하나 SW 관련 프로젝트 경험이 부족함."
    }
    """
    msg = HumanMessage(content=[{"type": "text", "text": prompt}] + image_messages)
    try:
        res = llm.invoke([msg]).content
        return json.loads(clean_json_output(res))
    except: return None

# -----------------------------------------------------------------------------
# [기능 2] 졸업 요건 분석 (JSON + 리포트 + 상담)
# -----------------------------------------------------------------------------
def analyze_graduation_json(uploaded_images):
    llm = get_pro_llm()
    image_messages = []
    for img_file in uploaded_images:
        img_file.seek(0)
        b64 = base64.b64encode(img_file.read()).decode("utf-8")
        image_messages.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    
    prompt = """
    졸업 요건을 진단하여 **JSON 데이터**와 **분석 리포트(Text)** 두 가지를 모두 포함한 JSON으로 출력하세요.
    [학사 문서]를 참고하여 정확히 계산하세요.
    출력 형식:
    {
        "chart_data": {
            "total": {"earned": 100, "required": 130},
            "major_req": {"earned": 15, "required": 21},
            "major_sel": {"earned": 30, "required": 54},
            "liberal": {"earned": 20, "required": 30}
        },
        "report_text": "### 🎓 졸업 요건 진단 결과\n\n..."
    }
    """
    msg = HumanMessage(content=[{"type": "text", "text": prompt}] + image_messages + [{"type": "text", "text": f"\n[학사 문서]\n{PRE_LEARNED_DATA}"}])
    try:
        res = llm.invoke([msg]).content
        return json.loads(clean_json_output(res))
    except: return None

# [복구] 졸업 요건 상담 함수
def chat_with_graduation_ai(current_analysis, user_input):
    llm = get_llm()
    template = """
    당신은 학사 전문 AI 상담사입니다.
    [현재 진단 결과] {current_analysis}
    [사용자 입력] "{user_input}"
    
    지시사항:
    1. 사용자가 정보를 수정/추가(예: "나 캡스톤 들었어")하면, 진단 결과를 수정해서 다시 작성하고 맨 앞에 **[수정]** 태그를 붙이세요.
    2. 단순 질문(예: "MSC가 뭐야?")이면 친절하게 답변만 하세요.
    
    [참고 문헌] {context}
    """
    prompt = PromptTemplate(template=template, input_variables=["current_analysis", "user_input", "context"])
    return (prompt | llm).invoke({"current_analysis": current_analysis, "user_input": user_input, "context": PRE_LEARNED_DATA}).content

# -----------------------------------------------------------------------------
# [기능 3] AI 도구 (커리어, 시간표)
# -----------------------------------------------------------------------------
def consult_career_path(job_role, grade_json, context):
    llm = get_llm()
    search = DuckDuckGoSearchRun()
    try: search_res = search.invoke(f"{job_role} 신입 채용 기술 스택 자격요건")
    except: search_res = "검색 불가"
    
    template = """
    당신은 냉철한 채용 담당자입니다.
    [지원자 스펙] {student_data}
    [시장 요구사항] {search_result}
    [학교 커리큘럼] {context}
    지원자의 부족한 점(Skill Gap)을 지적하고, 학교 강의 중 무엇을 들어야 할지 구체적으로 추천하세요.
    """
    prompt = PromptTemplate(template=template, input_variables=["student_data", "search_result", "context"])
    return (prompt | llm).invoke({"student_data": json.dumps(grade_json), "search_result": search_res, "context": context}).content

# [복구] 상세 프롬프트 적용된 시간표 생성
def generate_timetable_ai(major, grade, semester, target, blocked, req, shared_ctx):
    llm = get_llm()
    template = """
    수강신청 전문가로서 시간표를 작성하세요.
    [학생 정보] {major} {grade} {semester}, 목표 {target}학점
    [공강 시간] {blocked}
    [추가 요구] {req}
    
    ★★★ [이전 상담 맥락 반영] ★★★
    "{shared_ctx}"
    
    """ + COMMON_TIMETABLE_INSTRUCTION + """
    
    [학습 문서] {context}
    """
    prompt = PromptTemplate(template=template, input_variables=["major", "grade", "semester", "target", "blocked", "req", "shared_ctx", "context"])
    res = (prompt | llm).invoke({
        "major": major, "grade": grade, "semester": semester, "target": target, 
        "blocked": blocked, "req": req, "shared_ctx": shared_ctx, "context": PRE_LEARNED_DATA
    }).content
    return clean_html_output(res)

# [복구] 시간표 상담 함수
def chat_with_timetable_ai(current_timetable, user_input, major, grade, semester):
    llm = get_llm()
    template = """
    시간표 상담 AI입니다.
    [현재 시간표] {current_timetable}
    [사용자 입력] "{user_input}"
    [학생 정보] {major} {grade} {semester}
    
    지시사항:
    1. 시간표 수정 요청이면 **[수정]** 태그를 붙이고 HTML Table을 다시 작성하세요.
    """ + COMMON_TIMETABLE_INSTRUCTION + """
    2. 단순 질문이면 답변만 하세요.
    
    [학습 문서] {context}
    """
    prompt = PromptTemplate(template=template, input_variables=["current_timetable", "user_input", "major", "grade", "semester", "context"])
    res = (prompt | llm).invoke({
        "current_timetable": current_timetable, "user_input": user_input, 
        "major": major, "grade": grade, "semester": semester, "context": PRE_LEARNED_DATA
    }).content
    return res

# -----------------------------------------------------------------------------
# [UI] 메인 앱
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🗂️ 내비게이션")
    
    if st.session_state.user is None:
        with st.expander("🔐 로그인 / 회원가입", expanded=True):
            mode = st.radio("모드", ["로그인", "회원가입"], horizontal=True, label_visibility="collapsed")
            email = st.text_input("이메일")
            pw = st.text_input("비밀번호", type="password")
            if st.button("실행"):
                u, e = fb_manager.auth_user(email, pw, "login" if mode == "로그인" else "signup")
                if u:
                    st.session_state.user = u
                    # 데이터 로드
                    grade_data = fb_manager.load_user_data('grade_data', 'latest')
                    if grade_data: st.session_state.grade_json_data = grade_data
                    
                    grad_data = fb_manager.load_user_data('graduation_data', 'latest')
                    if grad_data: 
                        st.session_state.graduation_json_data = grad_data
                        if 'report_text' in grad_data: st.session_state.graduation_analysis_result = grad_data['report_text']
                    
                    st.success("로그인 성공!")
                    time.sleep(1)
                    st.rerun()
                else: st.error(e)
    else:
        st.info(f"👋 {st.session_state.user['email']}님")
        if st.button("로그아웃"):
            st.session_state.user = None
            st.session_state.grade_json_data = None
            st.session_state.graduation_json_data = None
            st.session_state.graduation_analysis_result = ""
            st.rerun()

    if st.session_state.user:
        st.divider()
        st.subheader("📂 Q&A 보관함")
        bookmarks = fb_manager.load_bookmarks()
        if not bookmarks: st.caption("저장된 내용이 없습니다.")
        for bm in bookmarks:
            with st.expander(f"📌 {bm['question'][:15]}..."):
                st.write(f"**Q:** {bm['question']}")
                st.write(f"**A:** {bm['answer']}")
                st.caption(f"Tag: {bm['tag']}")

# 메인 페이지
st.title("🎓 KW-강의마스터 Pro")
menu_options = ["📈 성적 및 진로 진단", "📅 스마트 시간표", "🤖 AI 학사 지식인"]
menu = st.radio("기능 선택", menu_options, horizontal=True, label_visibility="collapsed")

# -----------------------------------------------------------------------------
# MENU 1: 성적 및 진로 진단
# -----------------------------------------------------------------------------
if menu == "📈 성적 및 진로 진단":
    st.header("📈 성적 분석 및 진로 설계")
    sub_tabs = st.tabs(["📊 성적 분석", "🎓 졸업 요건 확인", "🚀 AI 커리어 솔루션"])
    
    # 1. 성적 분석
    with sub_tabs[0]:
        st.markdown("##### 📄 성적표 업로드")
        uploaded_grades = st.file_uploader("성적표 이미지", accept_multiple_files=True, key="grade_upl")
        if uploaded_grades and st.button("분석 시작", key="anlz_btn"):
            with st.spinner("분석 중..."):
                data = analyze_grades_structure(uploaded_grades)
                if data:
                    st.session_state.grade_json_data = data
                    if "weakness_analysis" in data: st.session_state.shared_context = data["weakness_analysis"]
                    fb_manager.save_user_data('grade_data', 'latest', data)
                    st.rerun()

        if st.session_state.grade_json_data:
            d = st.session_state.grade_json_data
            st.success(f"학번: {d.get('student_info',{}).get('admission_year')} | 전공: {d.get('student_info',{}).get('major')}")
            if st.session_state.shared_context: st.info(f"💡 **AI 진단(맥락):** {st.session_state.shared_context}")
            st.write("🔥 **나의 강점:** " + " ".join([f"`{k}`" for k in d.get("strength_keywords", [])]))
            df = pd.DataFrame(d.get("courses", []))
            if not df.empty:
                df['score'] = pd.to_numeric(df['score'], errors='coerce')
                st.line_chart(df.groupby('year')['score'].mean())
                with st.expander("데이터 원본"): st.json(d)

    # 2. 졸업 요건 (도넛 차트 + 상담 복구)
    with sub_tabs[1]:
        st.markdown("##### 🎓 졸업 요건 달성률")
        grad_files = st.file_uploader("졸업 요건용 성적표", accept_multiple_files=True, key="grad_upl")
        if grad_files and st.button("졸업 요건 진단", key="grad_btn"):
            with st.spinner("분석 중..."):
                res = analyze_graduation_json(grad_files)
                if res:
                    st.session_state.graduation_json_data = res
                    st.session_state.graduation_analysis_result = res.get("report_text", "")
                    st.session_state.graduation_chat_history = []
                    fb_manager.save_user_data('graduation_data', 'latest', res)
                    st.rerun()
        
        # 차트 표시
        if st.session_state.graduation_json_data:
            data = st.session_state.graduation_json_data.get("chart_data", {})
            if data:
                fig = make_subplots(rows=1, cols=4, specs=[[{'type':'domain'}]*4], 
                                    subplot_titles=['총 학점', '전공 필수', '전공 선택', '교양'])
                keys = ['total', 'major_req', 'major_sel', 'liberal']
                for i, key in enumerate(keys):
                    curr = data.get(key, {}).get('earned', 0)
                    req = data.get(key, {}).get('required', 100)
                    fig.add_trace(go.Pie(labels=["이수", "미이수"], values=[curr, max(0, req-curr)], hole=.6, marker_colors=['#4CAF50', '#E0E0E0'], textinfo='none'), 1, i+1)
                    fig.add_annotation(text=f"<b>{int((curr/req)*100 if req>0 else 0)}%</b>", x=[0.11, 0.37, 0.63, 0.89][i], y=0.5, showarrow=False, font_size=20)
                fig.update_layout(height=250, margin=dict(t=30, b=0, l=0, r=0), showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        # 리포트 및 상담
        if st.session_state.graduation_analysis_result:
            st.markdown(st.session_state.graduation_analysis_result)
            st.divider()
            st.subheader("💬 졸업 요건 상담소")
            for msg in st.session_state.graduation_chat_history:
                with st.chat_message(msg["role"]): st.markdown(msg["content"])
            
            if chat_input := st.chat_input("예: 캡스톤디자인 들었는데 왜 미이수야?", key="grad_chat"):
                st.session_state.graduation_chat_history.append({"role": "user", "content": chat_input})
                with st.chat_message("user"): st.write(chat_input)
                with st.chat_message("assistant"):
                    with st.spinner("분석 중..."):
                        resp = chat_with_graduation_ai(st.session_state.graduation_analysis_result, chat_input)
                        if "[수정]" in resp:
                            new_res = resp.replace("[수정]", "").strip()
                            st.session_state.graduation_analysis_result = new_res
                            st.write("리포트를 수정했습니다. 위 내용을 확인해주세요.")
                            st.session_state.graduation_chat_history.append({"role": "assistant", "content": "리포트 수정 완료."})
                            st.rerun()
                        else:
                            st.markdown(resp)
                            st.session_state.graduation_chat_history.append({"role": "assistant", "content": resp})

    # 3. 커리어
    with sub_tabs[2]:
        st.markdown("##### 🚀 AI 채용 담당자 컨설팅")
        job = st.text_input("희망 직무")
        if st.button("분석"):
            if not st.session_state.grade_json_data: st.error("성적 분석 먼저 진행하세요.")
            else:
                with st.spinner("검색 및 분석 중..."):
                    res = consult_career_path(job, st.session_state.grade_json_data, PRE_LEARNED_DATA)
                    st.markdown(res)
                    st.session_state.shared_context += f"\n(진로 조언: {job} 관련 역량 보강 필요)"

# -----------------------------------------------------------------------------
# MENU 2: 스마트 시간표 (상세 프롬프트 + 상담 복구)
# -----------------------------------------------------------------------------
elif menu == "📅 스마트 시간표":
    st.header("📅 맥락 기반 AI 시간표")
    if st.session_state.shared_context: st.info(f"💡 **반영된 맥락:** {st.session_state.shared_context}")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        major = st.selectbox("학과 선택", ALL_DEPARTMENTS)
        grade = st.selectbox("학년", ["1학년", "2학년", "3학년", "4학년"])
        semester = st.selectbox("학기", ["1학기", "2학기"])
        target = st.number_input("목표 학점", 9, 24, 18)
        req = st.text_area("추가 요구사항")
    with col2:
        st.caption("공강 시간 선택 (체크 해제 시 공강)")
        times = ["1교시", "2교시", "3교시", "4교시", "5교시", "6교시", "7교시", "8교시", "9교시"]
        if "sched_df" not in st.session_state:
            st.session_state.sched_df = pd.DataFrame(True, index=times, columns=["월", "화", "수", "목", "금"])
        edited_df = st.data_editor(st.session_state.sched_df, height=300, use_container_width=True)

    if st.button("시간표 생성", type="primary"):
        blocked = [f"{d} {t}" for d in edited_df.columns for t in times if not edited_df.loc[t, d]]
        with st.spinner("AI가 시간표 작성 중..."):
            res = generate_timetable_ai(major, grade, semester, target, ", ".join(blocked), req, st.session_state.shared_context)
            st.session_state.timetable_result = res
            st.rerun()

    if st.session_state.timetable_result:
        st.markdown(st.session_state.timetable_result, unsafe_allow_html=True)
        st.divider()
        st.subheader("💬 시간표 상담소")
        for msg in st.session_state.timetable_chat_history:
            with st.chat_message(msg["role"]): st.markdown(msg["content"], unsafe_allow_html=True)
        
        if chat_input := st.chat_input("예: 1교시 빼줘"):
            st.session_state.timetable_chat_history.append({"role": "user", "content": chat_input})
            with st.chat_message("user"): st.write(chat_input)
            with st.chat_message("assistant"):
                with st.spinner("수정 중..."):
                    resp = chat_with_timetable_ai(st.session_state.timetable_result, chat_input, major, grade, semester)
                    if "[수정]" in resp:
                        new_tt = clean_html_output(resp.replace("[수정]", ""))
                        st.session_state.timetable_result = new_tt
                        st.write("시간표를 업데이트했습니다.")
                        st.session_state.timetable_chat_history.append({"role": "assistant", "content": "수정 완료."})
                        st.rerun()
                    else:
                        st.markdown(resp)
                        st.session_state.timetable_chat_history.append({"role": "assistant", "content": resp})

# -----------------------------------------------------------------------------
# MENU 3: AI 학사 지식인 (보관함 기능)
# -----------------------------------------------------------------------------
elif menu == "🤖 AI 학사 지식인":
    st.subheader("🤖 무엇이든 물어보세요")
    for i, msg in enumerate(st.session_state.chat_history):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and i > 0 and st.session_state.chat_history[i-1]["role"] == "user":
                if st.button("💾 보관함 저장", key=f"save_{i}"):
                    if fb_manager.add_bookmark(st.session_state.chat_history[i-1]["content"], msg["content"], "지식인"):
                        st.toast("저장 완료!", icon="✅")
                    else: st.toast("로그인 필요", icon="⚠️")

    if user_input := st.chat_input("질문 입력"):
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"): st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("생성 중..."):
                q_ctx = user_input
                if st.session_state.shared_context: q_ctx = f"[상황: {st.session_state.shared_context}] \n{user_input}"
                chain = PromptTemplate.from_template("문서: {ctx}\n질문: {q}") | get_llm()
                response = chain.invoke({"ctx": PRE_LEARNED_DATA, "q": q_ctx}).content
                st.markdown(response)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()
