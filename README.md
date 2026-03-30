# 🛡️ AutoGuard
**LLM 기반 지능형 보안 사고 분석 및 대응 자동화 시스템**

> 본 프로젝트는 급변하는 사이버 위협에 대응하기 위해 LLM과 머신러닝(ML)을 결합, URL/이메일/PE 파일의 위협 탐지부터 대응 가이드 생성까지의 전 과정을 자동화합니다.

---

## 👨‍💻 Team: 오토가드 (AutoGuard)
* **역할 분담**
    * **팀장(TL)**: 유현상(https://github.com/runhy) - 시스템 아키텍처 설계, 멀티 에이전트 오케스트레이션 및 백엔드 통합
    * **ML (URL)**: 김효은(https://github.com/k1h2e3) - Random Forest 기반 악성 URL 탐지 모델 개발
    * **ML (Mail)**: 윤현식(https://github.com/YHs1k) - XGBoost 기반 스팸/피싱 메일 탐지 모델 개발
    * **ML (File)**: 이한별(https://github.com/hanbyeol325) - Random Forest 기반 PE 파일 정적 분석 모델 개발
    * **BE**: 김태현(https://github.com/bird8696) - FastAPI 서버 구축 및 외부 인텔리전스 API 연동
    * **FE/GRC 1**: 이재윤(https://github.com/leeyxon) - Streamlit 시각화 및 KISA 가이드 기반 RAG 환경 구축
    * **FE/GRC 2**: 신승준(https://github.com/seungjuuun01) - Streamlit 시각화 및 KISA 가이드 기반 RAG 환경 구축


---

## 📅 Project Timeline
* **2026.03.25**: 프로젝트 계획 수립 및 보안 데이터셋 확보
* **2026.03.25 - 03.27**: 분야별 ML 모델링 및 에이전트 핵심 로직 구현
* **2026.03.27 - 03.30**: 시스템 통합(FastAPI-Streamlit) 및 인터페이스 연동 테스트
* **2026.03.30 - 03.31**: RAG 성능 튜닝, 보고서 자동화 검증 및 최종 평가

---

## 🏗️ System Architecture
AutoGuard는 사용자의 입력부터 최종 리포트 도출까지 **멀티 에이전트(Multi-Agent)** 아키텍처를 기반으로 유기적으로 동작합니다.

### **핵심 구성 요소**
AutoGuard의 핵심 역량은 각 단계별로 특화된 4종의 지능형 에이전트 협업 체계에 있습니다.

1. **Dispatcher Agent (라우터)**
   - 사용자 입력을 실시간으로 식별하여 URL, 메일, 파일 중 최적의 분석 도구로 업무를 할당합니다.
   - 분석의 우선순위를 결정하고 전체 워크플로우를 트리거하는 관제탑 역할을 수행합니다.
2. **Analyzer Agent (기술 분석 엔진) - [추가됨]**
   - **기능**: 실질적인 머신러닝 추론 및 정적 분석을 수행합니다.
   - **세부 모듈**: 
     - **URL Analyzer**: 38개의 구조적 피처와 도메인 특성을 분석하여 악성 여부 판별.
     - **Mail Analyzer**: NLP(TF-IDF) 기반 텍스트 분석 및 피싱 징후 탐지.
     - **File Analyzer**: PE 구조 분석, 엔트로피 추출 및 섹션 변조 확인.
   - **역할**: 로우 데이터를 기술적 지표(Feature)와 확률값(Confidence Score)으로 변환하여 Advisor에게 전달합니다.
3. **Intel Agent (외부 위협 첩보)**
   - 내부 분석 결과의 신뢰도를 보완하기 위해 VirusTotal, Google Safe Browsing, Tavily AI(실시간 웹 검색) 등을 연동합니다.
   - 최신 위협 트렌드 및 해시 기반의 평판 데이터를 수집합니다.
4. **Advisor Agent (보안 전략가)**
   - Analyzer와 Intel Agent가 도출한 파편화된 기술 데이터를 수집합니다.
   - **KISA 침해사고 대응 매뉴얼**을 근거로, 일반 사용자가 즉시 실행 가능한 형태의 '최종 보안 권고 리포트'를 생성합니다.

---

### 🏗️ 데이터 분석 및 흐름 (Workflow)

1. **Input**: 사용자가 의심스러운 파일이나 URL을 시스템에 업로드합니다.
2. **Dispatch**: Dispatcher가 데이터의 유형을 판단하고 적절한 분석 에이전트를 호출합니다.
3. **Analysis**: **Analyzer Agent**가 도메인별 ML 모델을 가동하여 기술적 특징점과 악성 신뢰도를 산출합니다.
4. **Enrichment**: Intel Agent가 외부 DB 및 웹 검색을 통해 해당 위협의 실시간 평판 정보를 결합합니다.
5. **Reporting**: Advisor Agent가 모든 데이터를 종합하여 마크다운 형식의 전문 리포트를 출력합니다.

---

## 🚀 Key Technologies & Performance
각 분석 모듈은 도메인 특성에 최적화된 피처 엔지니어링과 알고리즘이 적용되었습니다.

### **1. Machine Learning Models**
| 분석 모듈 | 알고리즘 | 주요 피처 | 정확도(Accuracy) |
| :--- | :--- | :--- | :---: |
| **URL** | Random Forest | 구조적 특징, 문자 엔트로피, TLD 정보 | **92.0%** |
| **Mail** | XGBoost | TF-IDF(2~4gram), 한글/숫자 비중, URL 개수 | **99.43%** |
| **File** | Random Forest | PE 헤더 정보, DLL/함수 개수, 섹션 분석 | **99.6%** |

### **2. 핵심 최적화 기술**
* **하이브리드 분석 엔진**: 내부 ML 모델의 기술적 수치와 외부 인텔리전스의 평판 정보를 결합하여 판단 신뢰도 향상.
* **지능형 연쇄 호출(CoT)**: 이메일 본문 내 의심 URL이나 첨부파일 감지 시 하위 에이전트를 자동으로 가동하는 로직 구현.
* **사용자 편의 기능**: Enter 키 분석 실행, 분석 시작 시 입력창 자동 비우기(Auto-clear) 등 세밀한 UX 제공.

---

## 🛠️ Tech Stack
* **Frontend**: Streamlit
* **Backend**: FastAPI, Uvicorn
* **AI/LLM**: OpenAI GPT-4o, Tavily AI (Web Search)
* **ML Libraries**: Scikit-learn, XGBoost, Optuna, Pefile
* **Database/RAG**: Vector Store (JSON Cache), SentenceTransformer

---

## 📝 Technical Challenges & Resolutions (Tech Lead's Note)
* **Windows 파일 점유 문제**: 분석 직후 파일 핸들을 명시적으로 해제(`pe.close()`)하고 `safe_delete` 재시도 로직을 구현하여 `WinError 32` 해결.
* **데이터 정합성 확보**: LLM이 생성한 비정형 텍스트에서 위험 수치를 정규표현식(`XX/100`)으로 추출하여 대시보드와 실시간 동기화.
* **미탐율(FN) 최소화**: 패킹된 악성코드(UPX 등) 탐지력을 높이기 위해 임계값(Threshold)을 **0.5에서 0.2로 하향 조정**하여 보안 가용성 확보.

---

## 📈 Conclusion & Self-Evaluation
AutoGuard는 단순 탐지를 넘어 **"Actionable Advice(실행 가능한 조언)"**를 제공하는 지능형 보안 자동화 솔루션입니다.

* **성과**: 각 분야 92% 이상의 고성능 모델 확보 및 KISA 가이드 기반 대응 프로세스 자동화 성공.
* **한계 및 향후 계획**: 현재 정적 분석 위주의 파일 탐지를 샌드박스 기반의 동적 분석으로 확장하여 제로데이 공격 대응력을 강화할 예정입니다.