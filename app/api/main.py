# app/api/main.py

from fastapi import FastAPI
<<<<<<< Updated upstream
from app.api import safebrowsing, virustotal, websearch
=======
from fastapi.middleware.cors import CORSMiddleware 
from app.api import safebrowsing, virustotal, websearch, analyze, file_scan
import os
import logging

# 서버 시작 시 API 키 유효성 체크
if not os.getenv("VIRUSTOTAL_API_KEY"):
    raise RuntimeError("❌ VIRUSTOTAL_API_KEY 가 .env 에 설정되지 않았습니다")
if not os.getenv("GOOGLE_SAFE_BROWSING_API_KEY"):
    raise RuntimeError("❌ GOOGLE_SAFE_BROWSING_API_KEY 가 .env 에 설정되지 않았습니다")

# 로깅 설정 (서버 시작 시 한 번만 설정)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
>>>>>>> Stashed changes

# FastAPI 앱 인스턴스 생성 (서버의 중심)
app = FastAPI(
    title = "AutoGard API",
    description = "LLM 기반 보안 사고 분석 및 대응 자동화 시스템",
    version = "1.0.0"
)

<<<<<<< Updated upstream
# 라우터 등록
app.include_router(virustotal.router, prefix = "/virustodtal", tags = ["virustotal"])
app.include_router(safebrowsing.router, prefix = "/safebrowsing", tags = ["safebrowsing"])
app.include_router(websearch.router, prefix = "/websearch", tags = ["websearch"])
=======
# CORS 설정 추가 (Streamlit에서 FastAPI 호출 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# 라우터 등록(내부)
app.include_router(virustotal.router, prefix = "/virustotal", tags = ["virustotal"], include_in_schema=False)
app.include_router(safebrowsing.router, prefix = "/safebrowsing", tags = ["safebrowsing"], include_in_schema=False)
app.include_router(websearch.router, prefix = "/websearch", tags = ["websearch"], include_in_schema=False)
app.include_router(file_scan.router, prefix="/file", tags=["File"], include_in_schema=False)

# 라우터 등록(외부)
app.include_router(analyze.router, prefix="/analyze", tags=["Analyze"])
>>>>>>> Stashed changes

# 서버 상태 확인용 기본 엔드포인트
@app.get("/")
def root():
    return {
        "status" : "ok",
        "message" : "AutoGuard API 실행 중"
    }
    