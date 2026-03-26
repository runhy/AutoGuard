# app/api/main.py

from fastapi import FastAPI
from app.api import safebrowsing, virustotal, websearch, analyze

# FastAPI 앱 인스턴스 생성 (서버의 중심)
app = FastAPI(
    title = "AutoGuard API",
    description = "LLM 기반 보안 사고 분석 및 대응 자동화 시스템",
    version = "1.0.0"
)

# 라우터 등록(내부)
app.include_router(virustotal.router, prefix = "/virustotal", tags = ["virustotal"], include_in_schema=False)
app.include_router(safebrowsing.router, prefix = "/safebrowsing", tags = ["safebrowsing"], include_in_schema=False)
app.include_router(websearch.router, prefix = "/websearch", tags = ["websearch"], include_in_schema=False)

# 라우터 등록(외부)
app.include_router(analyze.router, prefix="/analyze", tags=["Analyze"])

# 서버 상태 확인용 기본 엔드포인트
@app.get("/")
def root():
    return {
        "status" : "ok",
        "message" : "AutoGuard API 실행 중"
    }
    
