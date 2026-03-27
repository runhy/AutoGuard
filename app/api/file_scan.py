# app/api/file_scan.py

from fastapi import APIRouter, UploadFile, File
import httpx                    # 비동기 지원 라이브러리
import os
import hashlib                  # 파일 해시값 변환
import logging
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()
VIRUSTOTAL_API_KEY = os.getenv("VIRUSTOTAL_API_KEY")

logger = logging.getLogger(__name__)

# 파일 해시값으로 VirusTotal DB 조회
@router.post("/scan")
async def scan_file(file: UploadFile = File(...)):      # 파일 업로드 받음
    headers = {
        "x-apikey" : VIRUSTOTAL_API_KEY
    }
    
    try:
        # 업로드된 파일 읽기
        file_content = await file.read()
        
        # 파일 크기 제한 (서버 메모리 보호용, ML팀 기준 넉넉하게 10MB)
        MAX_FILE_SIZE = 10 * 1024 * 1024            # 10MB
        if len(file_content) > MAX_FILE_SIZE:
            logger.warning(f"파일 크기 초과: {file.filename}")
            return {
                "module": "File_Analyzer",
                "file_name": file.filename,
                "is_malicious": 0,
                "confidence_score": 0.0,
                "detected_features": ["파일 크기 초과 (최대 10MB)"],
                "status_code": 413
            }

        # 파일 SHA256 해시값 추출
        file_hash = hashlib.sha256(file_content).hexdigest()

        async with httpx.AsyncClient(timeout=10.0) as client:
            # VirusTotal 파일 DB에 해시값으로 조회
            result = await client.get(
                f"https://www.virustotal.com/api/v3/files/{file_hash}",
                headers=headers
            )

            # 파일 DB에 있으면 즉시 반환
            if result.status_code == 200:
                logger.info(f"파일 분석 완료: {file.filename} | 해시: {file_hash}")
                stats = result.json()["data"]["attributes"]["last_analysis_stats"]
                return {
                    "module": "File_Analyzer",
                    "file_name": file.filename,             # 업로드된 파일명
                    "file_hash": file_hash,                 # SHA256 해시값
                    "is_malicious": 1 if stats["malicious"] > 0 else 0,
                    # 악성 비율 (0~1.0)
                    "confidence_score": round(stats["malicious"] / (stats["malicious"] + stats["harmless"] + stats["undetected"] + 0.001), 2),
                    "detected_features": [
                        f"악성 엔진 수: {stats['malicious']}",
                        f"의심 엔진 수: {stats['suspicious']}",
                        f"안전 엔진 수: {stats['harmless']}",
                        f"미탐지 엔진 수: {stats['undetected']}"
                    ]
                }

            # 파일 DB에 없으면 (한 번도 분석된 적 없는 파일)
            elif result.status_code == 404:
                logger.info(f"VirusTotal DB에 없는 파일: {file.filename}")
                return {
                    "module": "File_Analyzer",
                    "file_name": file.filename,
                    "file_hash": file_hash,
                    "is_malicious": 0,
                    "confidence_score": 0.0,
                    "detected_features": ["VirusTotal DB에 없는 파일 - 분석 이력 없음"]
                }

            # 429: API 요청 한도 초과
            elif result.status_code == 429:
                return {
                    "module": "File_Analyzer",
                    "file_name": file.filename,
                    "file_hash": file_hash,
                    "is_malicious": 0,
                    "confidence_score": 0.0,
                    "detected_features": ["VirusTotal API 요청 한도 초과 - 잠시 후 다시 시도"]
                }

            else:
                return {
                    "error": "파일 조회 실패",
                    "status_code": result.status_code
                }

    except httpx.TimeoutException:                      # API 응답 시간 초과
        return {
            "module": "File_Analyzer",
            "is_malicious": 0,
            "confidence_score": 0.0,
            "detected_features": ["VirusTotal API 응답 시간 초과"]
        }
    except httpx.RequestError:                          # API 연결 실패
        return {
            "module": "File_Analyzer",
            "is_malicious": 0,
            "confidence_score": 0.0,
            "detected_features": ["VirusTotal API 연결 실패"]
        }
                