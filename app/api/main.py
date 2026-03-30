# app/api/main.py
# FastAPI 서버의 진입점입니다. 에이전트 시스템과 API 라우터를 초기화하고 관리합니다.

import os
import logging
import sys
import shutil
import tempfile
import time
import hashlib  # [추가] 파일 해시값(SHA-256) 추출을 위한 모듈
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

# [경로 설정] 프로젝트 루트를 path에 추가하여 app 모듈 인식
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.api import safebrowsing, virustotal, websearch, analyze, file_scan
from app.agents.tools.dispatcher_agent import AutoGuardAgent
from app.agents.tools.intel_agent import IntelAgent
from app.agents.tools.advisor_agent import AdvisorAgent  # ✅ [추가] RAG 요약용

# [로깅 설정] 운영 가시성 확보
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("AutoGuard-Main")

# [환경변수 체크] 필수 API 키 확인
if not os.getenv("VIRUSTOTAL_API_KEY"):
    logger.warning("⚠️ VIRUSTOTAL_API_KEY가 설정되지 않았습니다.")
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("❌ OPENAI_API_KEY 설정이 필수적입니다. .env를 확인하세요.")

app = FastAPI(
    title="AutoGuard API", 
    description="LLM 및 전문 ML 엔진 기반 보안 분석 자동화 시스템",
    version="1.2.5"
)

# CORS 설정 (프론트엔드 연동용)
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
    """서버 시작 시 에이전트 엔진을 미리 로드합니다."""
    global intel_engine, dispatcher
    try:
        logger.info("[*] 에이전트 시스템 가동 중...")
        intel_engine = IntelAgent()
        dispatcher = AutoGuardAgent(intel_agent=intel_engine)
        await dispatcher.create_inspector()
        logger.info("[*] 에이전트 엔진(AutoGuard Dispatcher) 준비 완료!")
    except Exception as e:
        logger.error(f"[-] 에이전트 초기화 실패: {e}")

# ====== [API 요청 모델] ======
class ChatRequest(BaseModel):
    message: str

# ✅ [추가] RAG 요약 요청 모델
class RagSummarizeRequest(BaseModel):
    chunks: list[str]               # 벡터 DB에서 검색된 청크 텍스트 목록
    threat_type: str = "보안 위협"   # 분석 유형 (URL / Email / File 등)

# ====== [엔드포인트: 일반 채팅 분석] ======
@app.post("/agent/chat", tags=["Agent"])
async def agent_chat(request: ChatRequest):
    """자연어 질문을 분석하여 리포트를 반환합니다."""
    if not dispatcher:
        raise HTTPException(status_code=503, detail="에이전트 엔진이 아직 준비되지 않았습니다.")
    try:
        logger.info(f"[CHAT REQUEST] {request.message}")
        response = await dispatcher.run_agent(request.message)
        return {"status": "success", "reply": response}
    except Exception as e:
        logger.error(f"[-] 에이전트 처리 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ====== [엔드포인트: 실물 파일 업로드 분석] ======
@app.post("/agent/file-analysis", tags=["Agent"])
async def analyze_file_upload(file: UploadFile = File(...)):
    """
    [업그레이드] 실물 파일 저장 -> SHA-256 해시 추출 -> 에이전트 분석(ML + Intel) -> 자동 삭제
    윈도우 환경의 파일 잠금(WinError 32)을 완벽히 방어합니다.
    """
    if not dispatcher:
        raise HTTPException(status_code=503, detail="에이전트 엔진이 준비되지 않았습니다.")

    # 1. 임시 파일 경로 설정 (윈도우 호환성을 위해 NamedTemporaryFile 사용)
    extension = os.path.splitext(file.filename)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=extension)
    tmp_path = tmp.name
    
    try:
        # 2. 파일 내용 읽기 및 해시값 추출
        content = await file.read()
        
        # SHA-256 해시 계산 (에이전트에게 힌트로 제공)
        sha256_hash = hashlib.sha256(content).hexdigest()
        
        # 3. 임시 파일에 쓰기 및 즉시 닫기
        # 윈도우에서는 파일을 닫아야(close) 에이전트의 ML 도구가 파일을 읽을 수 있음
        tmp.write(content)
        tmp.close() 
        
        logger.info(f"[*] 업로드 완료: {file.filename} | Hash: {sha256_hash} | Path: {tmp_path}")

        # 4. 디스패처 분석 수행 (해시값과 경로를 함께 전달)
        query = (
            f"파일명: {file.filename}\n"
            f"경로: {tmp_path}\n"
            f"해시: {sha256_hash}\n\n"
            f"명령: 외부 검색 결과와 상관없이, **반드시 먼저 'predict_file_malicious' 도구를 실행**하여 "
            f"파일의 엔트로피와 PE 구조 피처 데이터를 직접 추출하라. "
            f"추출된 모델 데이터(확률값 등)를 리포트 상단에 배치하고, 그 다음 외부 인텔리전스와 비교하라."
        )
        
        response = await dispatcher.run_agent(query)
        
        return_data = {
            "status": "success",
            "filename": file.filename,
            "hash": sha256_hash,
            "report": response
        }

    except Exception as e:
        logger.error(f"[-] 파일 분석 프로세스 실패: {e}")
        raise HTTPException(status_code=500, detail=f"분석 중 오류 발생: {str(e)}")
    
    finally:
        # 5. 윈도우의 지연 잠금 해제 대응 재시도 삭제 로직
        def safe_delete(path):
            for i in range(5):
                try:
                    if os.path.exists(path):
                        os.remove(path)
                        logger.info(f"[*] 임시 파일 삭제 성공: {path}")
                    return
                except PermissionError:
                    logger.warning(f"[!] 파일 사용 중... {i+1}차 삭제 재시도 대기 (1s)")
                    time.sleep(1)
            logger.error(f"[!!] 파일 자동 삭제 실패: {path}. 수동 삭제가 필요합니다.")

        safe_delete(tmp_path)
    
    return return_data

# ✅ [추가] RAG 요약 엔드포인트
@app.post("/rag/summarize", tags=["RAG"])
async def summarize_rag(req: RagSummarizeRequest):
    """
    벡터 DB에서 검색된 파편화된 PDF 청크들을
    AdvisorAgent.summarize_rag_chunks()를 통해
    자연스러운 KISA 보안 권고문으로 재가공하여 반환합니다.
    """
    try:
        logger.info(f"[RAG SUMMARIZE] 위협 유형: {req.threat_type} | 청크 수: {len(req.chunks)}")
        advisor = AdvisorAgent()
        summary = await advisor.summarize_rag_chunks(req.chunks, req.threat_type)
        return {"summary": summary}
    except Exception as e:
        logger.error(f"[-] RAG 요약 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ====== [라우터 등록] ======
app.include_router(virustotal.router, prefix="/virustotal", tags=["VirusTotal"])
app.include_router(safebrowsing.router, prefix="/safebrowsing", tags=["SafeBrowsing"])
app.include_router(websearch.router, prefix="/websearch", tags=["WebSearch"])
app.include_router(file_scan.router, prefix="/file", tags=["File"])
app.include_router(analyze.router, prefix="/analyze", tags=["Analyze"])

@app.get("/")
def root():
    return {"status": "ok", "message": "AutoGuard API Server is Running"}