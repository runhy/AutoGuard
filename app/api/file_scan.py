# app/api/file_scan.py

from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel
import httpx
import os
import hashlib
import logging
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()
VIRUSTOTAL_API_KEY = os.getenv("VIRUSTOTAL_API_KEY")

logger = logging.getLogger(__name__)

# [추가] 에이전트의 JSON 요청 본문을 정의하는 모델
class HashRequest(BaseModel):
    hash: str

# [기존] 파일 업로드 스캔 (유지)
@router.post("/scan")
async def scan_file(file: UploadFile = File(...)):
    headers = {"x-apikey": VIRUSTOTAL_API_KEY}
    try:
        file_content = await file.read()
        MAX_FILE_SIZE = 10 * 1024 * 1024
        if len(file_content) > MAX_FILE_SIZE:
            return {"module": "File_Analyzer", "status_code": 413, "detected_features": ["파일 크기 초과"]}

        file_hash = hashlib.sha256(file_content).hexdigest()

        async with httpx.AsyncClient(timeout=10.0) as client:
            result = await client.get(
                f"https://www.virustotal.com/api/v3/files/{file_hash}",
                headers=headers
            )

            if result.status_code == 200:
                stats = result.json()["data"]["attributes"]["last_analysis_stats"]
                return {
                    "module": "File_Analyzer",
                    "file_name": file.filename,
                    "file_hash": file_hash,
                    "is_malicious": 1 if stats["malicious"] > 0 else 0,
                    "confidence_score": round(stats["malicious"] / (sum(stats.values()) + 0.001), 2),
                    "detected_features": [f"악성 엔진 수: {stats['malicious']}"]
                }
            return {"error": "파일 조회 실패", "status_code": result.status_code}
    except Exception as e:
        return {"error": str(e)}

# [수정] 에이전트 전용: POST + JSON 방식 (422 에러 해결 지점)
@router.post("/analyze-agent")
async def analyze_file_by_hash(request: HashRequest):
    """ 에이전트가 보낸 JSON { "hash": "..." } 데이터를 받아 VirusTotal에 조회합니다. """
    headers = {"x-apikey": VIRUSTOTAL_API_KEY}
    file_hash = request.hash
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            result = await client.get(
                f"https://www.virustotal.com/api/v3/files/{file_hash}",
                headers=headers
            )
            if result.status_code == 200:
                data = result.json()["data"]["attributes"]
                stats = data["last_analysis_stats"]
                return {
                    "is_malicious": 1 if stats["malicious"] > 0 else 0,
                    "confidence_score": round(stats["malicious"] / (sum(stats.values()) + 0.001), 2),
                    "details": stats
                }
            return {"is_malicious": 0, "message": "해당 해시 정보가 없습니다."}
    except Exception as e:
        return {"error": str(e)}