# app/api/main.py

from fastapi import FastAPI
from app.api import safebrowsing, virustotal, websearch

# FastAPI 앱 인스턴스 생성 (서버의 중심)
app = FastAPI(
    title = "AutoGard API",
    description = "LLM 기반 보안 사고 분석 및 대응 자동화 시스템",
    version = "1.0.0"
)

# 라우터 등록
app.include_router(virustotal.router, prefix = "/virustodtal", tags = ["virustotal"])
app.include_router(safebrowsing.router, prefix = "/safebrowsing", tags = ["safebrowsing"])
app.include_router(websearch.router, prefix = "/websearch", tags = ["websearch"])

# 서버 상태 확인용 기본 엔드포인트
@app.get("/")
def root():
    return {
        "status" : "ok",
        "message" : "AutoGuard API 실행 중"
    }
    
