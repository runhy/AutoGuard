# app/api/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.api import safebrowsing, virustotal, websearch, analyze, file_scan
import os
import logging
import sys
from pydantic import BaseModel

# 에이전트 모듈 임포트 (경로 설정 포함)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.agents.tools.dispatcher_agent import AutoGuardAgent
from app.agents.tools.intel_agent import IntelAgent

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

# ====== [에이전트 인스턴스 전역 관리] ======
intel_engine = None
dispatcher = None

@app.on_event("startup")
async def startup_event():
    global intel_engine, dispatcher
    try:
        logger.info("[*] 에이전트 시스템 가동 중...")
        # 1. 인텔 엔진(도구함) 생성
        intel_engine = IntelAgent()
        # 2. 디스패처(두뇌) 생성 및 도구 연결
        dispatcher = AutoGuardAgent(intel_agent=intel_engine)
        # 3. OpenAI Assistant 생성/연결
        await dispatcher.create_inspector()
        logger.info("[*] 에이전트 엔진 준비 완료!")
    except Exception as e:
        logger.error(f"[-] 에이전트 초기화 실패: {e}")

class ChatRequest(BaseModel):
    message: str

@app.post("/agent/chat", tags=["Agent"])
async def agent_chat(request: ChatRequest):
    """
    사용자의 자연어 질문을 받아 에이전트가 분석 후 답변을 반환합니다.
    예: "이 해시 분석해줘: 275a02..."
    """
    if not dispatcher:
        raise HTTPException(status_code=503, detail="에이전트 엔진이 아직 준비되지 않았습니다.")
        
    try:
        logger.info(f"[CHAT REQUEST] {request.message}")
        # 에이전트 실행 (이 과정에서 도구 호출, 서버 내부 통신이 모두 일어남)
        response = await dispatcher.run_agent(request.message)
        return {"status": "success", "reply": response}
    except Exception as e:
        logger.error(f"[-] 에이전트 처리 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ====== [기존 라우터 등록] ======
app.include_router(virustotal.router, prefix="/virustotal", tags=["virustotal"], include_in_schema=False)
app.include_router(safebrowsing.router, prefix="/safebrowsing", tags=["safebrowsing"], include_in_schema=False)
app.include_router(websearch.router, prefix="/websearch", tags=["websearch"], include_in_schema=False)
app.include_router(file_scan.router, prefix="/file", tags=["File"], include_in_schema=False)
app.include_router(analyze.router, prefix="/analyze", tags=["Analyze"])

@app.get("/")
def root():
    return {"status": "ok", "message": "AutoGuard API 실행 중"}