import streamlit as st
import pandas as pd
import os
import glob
import datetime
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# -----------------------------------------------------------------------------
# [0] 설정 및 데이터 로드
# -----------------------------------------------------------------------------
st.set_page_config(page_title="KW-강의마스터 Pro", page_icon="🎓", layout="wide")
api_key = os.environ.get("GOOGLE_API_KEY", "")

# 세션 상태 초기화
if "global_log" not in st.session_state:
    st.session_state.global_log = [] 
if "timetable_result" not in st.session_state:
    st.session_state.timetable_result = "" 
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] 
if "current_menu" not in st.session_state:
    st.session_state.current_menu = "🤖 AI 학사 지식인"
if "timetable_chat_history" not in st.session_state:
    st.session_state.timetable_chat_history = []

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
    return cleaned.strip()

# ★ 재시도(Retry) 로직 ★
def run_with_retry(func, *args, **kwargs):
    """API 호출 실패 시 지수 백오프로 재시도"""
    max_retries = 5
    delays = [1, 2, 4, 8, 16]
    
    for i in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                if i < max_retries - 1:
                    wait_time = delays[i]
                    time.sleep(wait_time) 
                    continue
            raise e

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
# [1] AI 엔진
# -----------------------------------------------------------------------------
def get_llm():
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

# 시간표 생성 함수 (강화된 프롬프트 적용)
def generate_timetable_ai(major, grade, semester, target_credits, blocked_times_desc, requirements):
    llm = get_llm()
    if not llm: return "⚠️ API Key 오류"
    
    def _execute():
        template = """
        너는 대학교 수강신청 전문가야. 오직 제공된 [학습된 문서]의 텍스트 데이터에 기반해서만 시간표를 짜줘.

        [학생 정보]
        - 소속: {major}
        - 학년/학기: {grade} {semester}
        - 목표: {target_credits}학점
        - 공강 필수 시간: {blocked_times} (이 시간은 수업 배치 절대 금지)
        - 추가요구: {requirements}

        [★★★ 초강력 데이터 검증 규칙 - 위반 시 치명적 오류로 간주 ★★★]
        
        1. **교과목명 100% 일치 필수 (유사어 금지)**:
           - 요람(Curriculum)에 적힌 과목명과 시간표(Schedule)에 적힌 과목명이 **글자 하나까지 정확히 일치**해야 합니다.
           - 예: 요람에 '대학물리학1'이라 되어 있다면, 시간표의 '대학물리및실험1'을 가져오면 **안 됩니다**. '대학물리학1'을 찾아야 합니다.
           - 예: 'C프로그래밍'과 '고급C프로그래밍'은 다른 과목입니다.
           - **만약 정확히 일치하는 과목명이 시간표에 없다면, 절대 표에 넣지 말고 아래 '배치 실패 목록'에 적으세요.**

        2. **강의 시간 변조 및 확장 금지**:
           - PDF 시간표에 적힌 요일과 교시를 **절대로** 변경하거나 늘리지 마세요.
           - 예: PDF에 "월1, 수2"라고 적혀있다면, **그대로 "월1, 수2"**에만 배치해야 합니다.
           - 절대로 "월1,2, 수1,2" 처럼 시간을 임의로 늘려서 잡지 마세요.
           - 시간이 "미정"이거나 비어있다면 표에 넣지 마세요.

        3. **학년 및 이수구분 엄격 준수**:
           - {grade} {semester} 커리큘럼상 **'필수(Required)'**로 지정된 과목(전공필수, 교양필수)을 최우선으로 찾으세요.
           - 해당 학년의 과목이 아닌데 임의로 넣지 마세요. (예: 1학년 시간표에 3학년 전공선택을 넣지 마세요.)
           - 전공, 교필, 교선, 기교 등의 이수 구분을 문서에 적힌 그대로 따르세요.

        [출력 형식 (세로형 HTML Table)]
        - 반드시 **HTML `<table>` 태그** 사용.
        - **행(Row): 1교시 ~ 9교시**
        - **열(Column): 월, 화, 수, 목, 금, 토, 일** (7일 모두 표시)
        - 각 수업 셀마다 **서로 다른 파스텔톤 배경색** 적용.
        - 셀 내용: `<b>과목명</b><br><small>교수명</small>`
        - 빈 시간(공강)은 비워둘 것.

        [출력 순서]
        1. 시간표 HTML 표 (가장 먼저 출력)
        2. **배치된 필수 과목 검증**: (예: "대학수학1: 요람의 1-1 필수 과목과 일치하여 배치함")
        3. **⚠️ 배치 실패 목록**: (필수인데 시간표 데이터에서 이름을 못 찾거나 시간이 없는 경우 여기에 명시)

        [학습된 문서]
        {context}
        """
        prompt = PromptTemplate(template=template, input_variables=["context", "major", "grade", "semester", "target_credits", "blocked_times", "requirements"])
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

def chat_with_timetable_ai(current_timetable, user_input):
    llm = get_llm()
    
    def _execute():
        template = """
        너는 현재 시간표에 대한 상담을 해주는 AI 조교야.
        
        [현재 시간표 상태]
        {current_timetable}

        [사용자 입력]
        "{user_input}"

        [지시사항]
        사용자의 입력 의도를 파악해서 아래 두 가지 중 하나로 반응해.
        
        **Case 1. 시간표 수정 요청인 경우 (예: "1교시 빼줘", "교수 바꿔줘"):**
        - 시간표를 **재작성(HTML Table 형식 유지 - 세로형, 월~일 7일 표시)**해줘.
        - **HTML 코드를 마크다운 코드 블록(```html)으로 감싸지 마라.** Raw HTML로 출력해.
        - **중요**: 수정 시에도 없는 과목을 만들거나, 시간을 임의로 늘리지 마. (원래 1시간짜리면 1시간만 배치)
        
        **Case 2. 과목에 대한 단순 질문인 경우 (예: "이거 선수과목 뭐야?"):**
        - **시간표를 다시 출력하지 말고**, 질문에 대한 **텍스트 답변**만 해.
        - **답변할 때 근거가 되는 문서의 원문 내용을 반드시 " " (쌍따옴표) 안에 인용해서 포함해줘.**
        
        답변 시작에 [수정] 또는 [답변] 태그를 붙여서 구분해줘.
        """
        prompt = PromptTemplate(template=template, input_variables=["current_timetable", "user_input"])
        chain = prompt | llm
        return chain.invoke({"current_timetable": current_timetable, "user_input": user_input}).content
    
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

# -----------------------------------------------------------------------------
# [2] UI 구성
# -----------------------------------------------------------------------------
def change_menu(menu_name):
    st.session_state.current_menu = menu_name

with st.sidebar:
    st.title("🗂️ 활동 로그")
    st.caption("클릭하면 해당 화면으로 이동합니다.")
    
    log_container = st.container(height=400)
    with log_container:
        if not st.session_state.global_log:
            st.info("기록 없음")
        else:
            for i, log in enumerate(reversed(st.session_state.global_log)):
                label = f"[{log['time']}] {log['content'][:15]}..."
                if st.button(label, key=f"log_btn_{i}", use_container_width=True):
                    if log['menu']:
                        change_menu(log['menu'])
                        st.rerun()

    st.divider()
    # 학습 상태 표시
    if PRE_LEARNED_DATA:
         st.success(f"✅ PDF 문서 학습 완료")
    else:
        st.error("⚠️ 데이터 폴더에 PDF 파일이 없습니다.")

menu = st.radio("기능 선택", ["🤖 AI 학사 지식인", "📅 스마트 시간표(수정가능)"], 
                horizontal=True, key="menu_radio", 
                index=["🤖 AI 학사 지식인", "📅 스마트 시간표(수정가능)"].index(st.session_state.current_menu))

if menu != st.session_state.current_menu:
    st.session_state.current_menu = menu
    st.rerun()

st.divider()

if st.session_state.current_menu == "🤖 AI 학사 지식인":
    st.subheader("🤖 무엇이든 물어보세요")
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

    # 시간표 표시 영역을 위한 빈 컨테이너 생성 (Placeholder)
    timetable_area = st.empty()

    # 현재 시간표가 있으면 표시
    if st.session_state.timetable_result:
        with timetable_area.container():
            st.markdown("### 🗓️ 내 시간표")
            st.markdown(st.session_state.timetable_result, unsafe_allow_html=True)
            st.divider()

    with st.expander("시간표 설정 열기/닫기", expanded=not bool(st.session_state.timetable_result)):
        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.markdown("#### 1️⃣ 기본 정보")
            # 광운대학교 주요 학과 리스트
            kw_departments = [
                "전자융합공학과", "전자공학과", "전자통신공학과", "전기공학과", 
                "전자재료공학과", "로봇학부", "컴퓨터정보공학부", "소프트웨어학부", 
                "정보융합학부", "건축학과", "건축공학과", "화학공학과", "환경공학과"
            ]
            major = st.selectbox("학과", kw_departments)
            
            c1, c2 = st.columns(2)
            grade = c1.selectbox("학년", ["1학년", "2학년", "3학년", "4학년"])
            semester = c2.selectbox("학기", ["1학기", "2학기"])
            target_credit = st.number_input("목표 학점", 9, 24, 18)
            requirements = st.text_area("추가 요구사항", placeholder="예: 전공 필수 챙겨줘")

        with col2:
            st.markdown("#### 2️⃣ 공강 시간 설정")
            kw_times = {
                "1교시": "09:00~10:15", "2교시": "10:30~11:45", "3교시": "12:00~13:15",
                "4교시": "13:30~14:45", "5교시": "15:00~16:15", "6교시": "16:30~17:45",
                "7교시": "18:00~19:15", "8교시": "19:25~20:40", "9교시": "20:50~22:05"
            }
            schedule_index = [f"{k} ({v})" for k, v in kw_times.items()]
            schedule_data = pd.DataFrame(True, index=schedule_index, columns=["월", "화", "수", "목", "금"])
            edited_schedule = st.data_editor(
                schedule_data,
                column_config={
                    "월": st.column_config.CheckboxColumn("월", default=True),
                    "화": st.column_config.CheckboxColumn("화", default=True),
                    "수": st.column_config.CheckboxColumn("수", default=True),
                    "목": st.column_config.CheckboxColumn("목", default=True),
                    "금": st.column_config.CheckboxColumn("금", default=True),
                },
                height=360,
                use_container_width=True
            )

        if st.button("시간표 생성하기 ✨", type="primary", use_container_width=True):
            blocked_times = []
            for day in ["월", "화", "수", "목", "금"]:
                for idx, period_label in enumerate(edited_schedule.index):
                    if not edited_schedule.iloc[idx][day]:
                        blocked_times.append(f"{day}요일 {period_label}")
            blocked_desc = ", ".join(blocked_times) if blocked_times else "없음"
            with st.spinner("선수과목 확인 및 시간표 조합 중... (최대 1분 소요될 수 있습니다)"):
                result = generate_timetable_ai(major, grade, semester, target_credit, blocked_desc, requirements)
                st.session_state.timetable_result = result
                st.session_state.timetable_chat_history = []
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
                    response = chat_with_timetable_ai(st.session_state.timetable_result, chat_input)
                    
                    if "[수정]" in response:
                        new_timetable = response.replace("[수정]", "").strip()
                        new_timetable = clean_html_output(new_timetable) 
                        st.session_state.timetable_result = new_timetable
                        
                        with timetable_area.container():
                            st.markdown("### 🗓️ 내 시간표")
                            st.markdown(new_timetable, unsafe_allow_html=True)
                            st.divider()
                        
                        success_msg = "시간표를 수정했습니다. 위쪽 표가 업데이트 되었습니다."
                        st.write(success_msg)
                        st.session_state.timetable_chat_history.append({"role": "assistant", "content": success_msg})
                    else:
                        clean_response = response.replace("[답변]", "").strip()
                        st.markdown(clean_response)
                        st.session_state.timetable_chat_history.append({"role": "assistant", "content": clean_response})
