import streamlit as st
import pandas as pd
import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# -----------------------------------------------------------------------------
# [1] 서버 설정 및 데이터 로드
# -----------------------------------------------------------------------------
st.set_page_config(page_title="KW-강의마스터", page_icon="🎓", layout="wide")
api_key = os.environ.get("GOOGLE_API_KEY", "")

# 지식 베이스 로딩 함수 (data 폴더의 모든 PDF 읽기)
@st.cache_resource(show_spinner="학교 정보를 학습하는 중입니다... (약 1분 소요)")
def load_knowledge_base():
    all_content = ""
    
    if not os.path.exists("data"):
        os.makedirs("data")
        return ""

    pdf_files = glob.glob("data/*.pdf")
    
    if not pdf_files:
        return ""

    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(pdf_file)
            pages = loader.load_and_split()
            
            filename = os.path.basename(pdf_file)
            all_content += f"\n\n--- [문서 시작: {filename}] ---\n"
            
            for page in pages:
                all_content += page.page_content
                
        except Exception as e:
            print(f"Error loading {pdf_file}: {e}")
            continue
            
    return all_content

PRE_LEARNED_DATA = load_knowledge_base()

# -----------------------------------------------------------------------------
# [2] AI 엔진 (질의응답 & 고도화된 시간표 생성)
# -----------------------------------------------------------------------------
def get_llm():
    if not api_key:
        return None
    # 404 오류 방지 및 최신 모델 사용
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-09-2025", temperature=0)

def ask_ai(question):
    llm = get_llm()
    if not llm: return "⚠️ 서버에 API Key가 설정되지 않았습니다."
    if not PRE_LEARNED_DATA: return "⚠️ 학습된 데이터가 없습니다."

    try:
        template = """
        너는 광운대학교 학사 전문 상담 비서 'KW-강의마스터'야.
        [지시사항]
        1. 질문에 대한 답변은 오직 제공된 [학습된 PDF 문서들] 내용에 기반해서 작성해.
        2. 출처(문서명)를 언급해줘.
        3. 모르는 내용은 모른다고 답해.

        [학습된 PDF 문서들]
        {context}

        [질문]
        {question}
        """
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        chain = prompt | llm
        response = chain.invoke({"context": PRE_LEARNED_DATA, "question": question})
        return response.content
    except Exception as e:
        return f"❌ AI 오류: {str(e)}"

def generate_timetable_ai(major, grade, semester, target_credits, free_days, requirements):
    llm = get_llm()
    if not llm: return "⚠️ 서버에 API Key가 설정되지 않았습니다."
    if not PRE_LEARNED_DATA: return "⚠️ 학습된 데이터가 없습니다."

    try:
        # 시간표 생성 전용 고도화 프롬프트
        template = """
        너는 대학교 수강신청 및 커리큘럼 전문가야. 
        제공된 [학습된 PDF 문서들](학사요람, 강의시간표 등)을 철저히 분석하여 학생에게 최적화된 시간표를 작성해줘.

        [학생 정보]
        - 소속 학과: {major}
        - 학년/학기: {grade} {semester}
        - 목표 학점: {target_credits}학점
        - 공강 희망 요일: {free_days} (이 요일 수업 배제)
        - 추가 요구사항: {requirements}

        [필수 지시사항 - 단계별로 생각할 것]
        1. **필수 과목 식별**: {major} {grade}학년 {semester} 커리큘럼상 **반드시 들어야 하는 과목(전공필수, 교양필수, 학문기초 등)**을 PDF에서 찾아내라. 
           (예: 1학년 1학기라면 대학수학, 대학물리, 프로그래밍 기초 등)
        2. **선택적 필수 고려**: "1학년 중 택1" 또는 "1학기/2학기 중 선택 수강"인 과목(예: 공학설계입문)은 현재 학점 상황과 시간표 밸런스를 고려해 넣을지 말지 결정해라.
        3. **선수 과목 체크**: 해당 학년에 듣기에 부적절하거나 선수과목이 필요한 수업인지 확인해라.
        4. **시간표 배치**: 
           - 필수 과목을 최우선으로 배치한다.
           - 남는 학점은 전공선택이나 균형교양으로 채운다.
           - 실제 PDF에 있는 '강의 시간'과 '교수님 성함'을 매칭한다.
           - 공강 희망 요일을 최대한 지킨다.
        
        [출력 형식 - 에브리타임 스타일]
        1. 결과는 반드시 **마크다운 표(Table)**로 작성한다.
           - 열: 시간, 월, 화, 수, 목, 금
           - 행: 1교시(09:00) ~ 9교시(17:00)
        2. 셀 내용: **과목명<br>(교수명)** (HTML 줄바꿈 태그 사용)
        3. 표 아래에 **"상세 분석 리포트"**를 작성해라.
           - **필수 과목 포함 여부**: 왜 이 과목들을 넣었는지(커리큘럼 근거).
           - **학점 구성**: 전공 O학점, 교양 O학점.
           - **주의사항**: 선수과목 경고나 수강신청 팁.

        [학습된 PDF 문서들]
        {context}
        """
        prompt = PromptTemplate(template=template, input_variables=["context", "major", "grade", "semester", "target_credits", "free_days", "requirements"])
        chain = prompt | llm
        
        input_data = {
            "context": PRE_LEARNED_DATA,
            "major": major,
            "grade": grade,
            "semester": semester,
            "target_credits": target_credits,
            "free_days": ", ".join(free_days) if free_days else "없음",
            "requirements": requirements if requirements else "없음"
        }
        
        response = chain.invoke(input_data)
        return response.content
    except Exception as e:
        return f"❌ AI 오류: {str(e)}"

# -----------------------------------------------------------------------------
# [3] UI 구성
# -----------------------------------------------------------------------------
st.sidebar.title("🎓 KW-강의마스터")
try:
    pdf_count = len(glob.glob("data/*.pdf"))
except:
    pdf_count = 0
st.sidebar.info(f"📚 학습된 문서: {pdf_count}개")

menu = st.sidebar.radio("메뉴", ["AI 학사 지식인", "이수학점 진단", "스마트 시간표"])

if menu == "AI 학사 지식인":
    st.header("🤖 AI 학사 지식인")
    st.caption("궁금한 학사 규정이나 커리큘럼을 물어보세요.")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input := st.chat_input("질문 입력 (예: 전자융합공학과 졸업 요건이 뭐야?)"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("문서를 분석 중입니다..."):
                answer = ask_ai(user_input)
                st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

elif menu == "이수학점 진단":
    st.header("📊 졸업 이수 현황 (간편)")
    col1, col2 = st.columns(2)
    with col1:
        major_score = st.number_input("전공 이수 학점", 0, 150, 45)
        ge_score = st.number_input("교양 이수 학점", 0, 150, 20)
    with col2:
        total = major_score + ge_score
        st.metric("총 이수 학점", f"{total} / 130")
        st.progress(min(total/130, 1.0))

elif menu == "스마트 시간표":
    st.header("📅 AI 맞춤형 시간표 설계")
    st.info("학과 요람과 강의 시간표 PDF를 분석하여, 필수 과목을 포함한 최적의 시간표를 제안합니다.")

    # 입력 폼 고도화
    with st.form("timetable_form"):
        col1, col2 = st.columns(2)
        with col1:
            major_input = st.text_input("소속 학과 (정확히 입력)", value="전자융합공학과")
            grade_input = st.selectbox("학년", ["1학년", "2학년", "3학년", "4학년"])
            semester_input = st.selectbox("학기", ["1학기", "2학기"])
        
        with col2:
            target_credit = st.number_input("목표 학점", 9, 24, 19)
            free_days = st.multiselect("공강 희망 요일", ["월", "화", "수", "목", "금"])
            requirements = st.text_input("추가 요구사항 (예: 오전 수업 선호, 영어강의 제외 등)")
        
        submitted = st.form_submit_button("시간표 생성하기 ✨")

    if submitted:
        with st.spinner(f"{major_input} {grade_input} {semester_input} 커리큘럼 분석 및 시간표 생성 중..."):
            result = generate_timetable_ai(major_input, grade_input, semester_input, target_credit, free_days, requirements)
            st.markdown("### 🗓️ 추천 시간표")
            st.markdown(result, unsafe_allow_html=True)
