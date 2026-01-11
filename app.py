import streamlit as st
import pandas as pd
import os
import glob
import datetime
import time
import base64
import re  # 정규표현식 추가
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

# Firebase 라이브러리 (Admin SDK)
import firebase_admin
from firebase_admin import credentials, firestore

# ... [기존 설정, CSS, FirebaseManager 클래스 등은 수정 없이 그대로 유지] ...

# -----------------------------------------------------------------------------
# [수정 1] 데이터 로드 함수: PDF 대신 .txt 파일 로드 및 강의계획서 파싱
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="학사 데이터 및 강의계획서를 분석 중입니다...")
def load_knowledge_base():
    if not os.path.exists("data"):
        return ""
    
    # .txt 파일 로드
    txt_files = glob.glob("data/*.txt")
    if not txt_files:
        return ""
        
    all_content = ""
    syllabus_list = [] # 강의계획서가 있는 과목명 저장

    for txt_file in txt_files:
        try:
            filename = os.path.basename(txt_file)
            with open(txt_file, "r", encoding="utf-8") as f:
                content = f.read()

            # 1. 강의계획서 파일 식별 및 파싱
            # (파일 내용에 '# 강의계획서 구조화'가 있거나 파일명에 포함된 경우)
            if "강의계획서" in content or "강의계획서" in filename:
                # 교과목명 추출 시도 (정규식: | **교과목명** | 과목명 |)
                match = re.search(r"\|\s*\*\*교과목명\*\*\s*\|\s*(.*?)\s*\|", content)
                subject_name = "알수없음"
                if match:
                    # '대학물리학1-General Physics 1' -> '대학물리학1'만 추출
                    raw_name = match.group(1).strip()
                    subject_name = raw_name.split("-")[0].strip()
                    syllabus_list.append(subject_name)
                
                # AI에게 명확히 구분해주기 위해 태그 추가
                all_content += f"\n\n--- [강의계획서 데이터: {subject_name}] ---\n{content}"
            
            # 2. 일반 수강신청 자료집 (규정 등)
            else:
                all_content += f"\n\n--- [학사 규정 및 가이드: {filename}] ---\n{content}"

        except Exception as e:
            print(f"Error loading {txt_file}: {e}")
            continue
            
    # 강의계획서 목록을 전역 변수처럼 프롬프트에 활용하기 위해 텍스트 상단에 요약 추가
    if syllabus_list:
        summary_header = f"--- [보유한 강의계획서 목록] ---\n{', '.join(syllabus_list)}\n\n"
        all_content = summary_header + all_content

    return all_content

PRE_LEARNED_DATA = load_knowledge_base()

# ... [AI 엔진 설정 함수 get_llm, get_pro_llm, ask_ai 등은 그대로 유지] ...

# -----------------------------------------------------------------------------
# [수정 2] 공통 프롬프트: <details> 태그 사용 규칙 추가
# -----------------------------------------------------------------------------
COMMON_TIMETABLE_INSTRUCTION = """
[★★★ 핵심 알고리즘: 3단계 검증 및 필터링 (Strict Verification) ★★★]
1. **Step 1: 요람(Curriculum) 기반 '수강 대상' 리스트 확정**:
   - [학사 규정] 문서에서 **'{major} {grade} {semester}'**에 배정된 **'표준 이수 과목' 목록**을 추출.
2. **Step 2: 학년 정합성 검사 (Grade Validation)**:
   - 사용자가 선택한 학년({grade})과 시간표의 대상 학년이 일치하지 않으면 과감히 제외.
3. **Step 3: 시간표 데이터와 정밀 대조 (Exact Match)**:
   - 위 단계를 통과한 과목만 시간표에 배치. 과목명 완전 일치 필수.
   - **[핵심 규칙] 요일별 교시 분리 배정**: 만약 강의 시간이 **'월3, 수4'**로 되어 있다면, **월요일은 3교시만, 수요일은 4교시만** 채워야 합니다.
   - **절대** '월3,4' 혹은 '수3,4'처럼 연강으로 임의 확장하거나 빈 시간을 채워넣지 마세요.

[★★★ UI 렌더링 규칙: 강의계획서 요약 연동 (<details> 태그) ★★★]
- 만약 배정하려는 과목이 **[보유한 강의계획서 목록]**에 포함되어 있거나, **[강의계획서 데이터: 과목명]** 섹션이 존재한다면,
- 해당 과목의 셀 안에 반드시 **`<details>` 태그**를 사용하여 **핵심 요약(3줄 이내)**을 포함시켜라.
- **요약 내용:** 핵심 목표, 평가 비율(중간/기말/과제), 수업 방식(대면/비대면).
- **HTML 출력 형식 예시 (엄격 준수):**
  ```html
  <b>과목명</b><br><small>교수명 (대상학년)</small>
  <details style="margin-top:5px; cursor:pointer; border-top:1px dashed #eee;">
      <summary style="color:#0056b3; font-size:0.8em; font-weight:bold;">📄 계획서 요약</summary>
      <div style="background:#f8f9fa; padding:5px; border-radius:4px; font-size:0.75em; text-align:left;">
          • <b>목표:</b> 역학 및 파동 원리 이해<br>
          • <b>평가:</b> 시험 70%, 과제 20%<br>
          • <b>방식:</b> 100% 녹화 강의
      </div>
  </details>
