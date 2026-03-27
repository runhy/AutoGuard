# app/api/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.api import safebrowsing, virustotal, websearch, analyze, file_scan
import os
import logging
# ====== [BE1 김태현] 에이전트 연동 전까지 주석 처리 ======
# import sys
# from pydantic import BaseModel
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from agents.tools.dispatcher_agent import AutoGuardAgent
# from agents.tools.intel_agent import IntelAgent
# ======================================================

# API 키 체크
if not os.getenv("VIRUSTOTAL_API_KEY"):
    raise RuntimeError("❌ VIRUSTOTAL_API_KEY 설정 필요")
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("❌ OPENAI_API_KEY 설정 필요")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("AutoGuard-Main")

app = FastAPI(title="AutoGuard API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ====== [BE1 김태현] 에이전트 연동 전까지 주석 처리 ======
# intel_engine = None
# dispatcher = None
#
# @app.on_event("startup")
# async def startup_event():
#     global intel_engine, dispatcher
#     logger.info("[*] 에이전트 시스템 가동 중...")
#     intel_engine = IntelAgent()
#     dispatcher = AutoGuardAgent(intel_agent=intel_engine)
#     await dispatcher.create_inspector()
#     logger.info("[*] 에이전트 생성 완료!")
#
# class ChatRequest(BaseModel):
#     message: str
#
# @app.post("/agent/chat", tags=["Agent"])
# async def agent_chat(request: ChatRequest):
#     try:
#         response = await dispatcher.run_agent(request.message)
#         return {"status": "success", "reply": response}
#     except Exception as e:
#         logger.error(f"[-] 에이전트 처리 오류: {e}")
#         raise HTTPException(status_code=500, detail=str(e))
# ======================================================

# 라우터 등록
app.include_router(virustotal.router, prefix="/virustotal", tags=["virustotal"], include_in_schema=False)
app.include_router(safebrowsing.router, prefix="/safebrowsing", tags=["safebrowsing"], include_in_schema=False)
app.include_router(websearch.router, prefix="/websearch", tags=["websearch"], include_in_schema=False)
app.include_router(file_scan.router, prefix="/file", tags=["File"], include_in_schema=False)
app.include_router(analyze.router, prefix="/analyze", tags=["Analyze"])

@app.get("/")
def root():
    return {"status": "ok", "message": "AutoGuard API 실행 중"}