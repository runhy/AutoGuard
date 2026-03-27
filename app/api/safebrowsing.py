# app/api/safebrowsing.py

from fastapi import APIRouter
import requests
import os
import logging
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

GOOGLE_API_KEY = os.getenv("GOOGLE_SAFE_BROWSING_API_KEY")

logger = logging.getLogger(__name__)

@router.get("/scan")
<<<<<<< Updated upstream
def scan_url(url: str):
=======
async def scan_url(url: str):
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        logger.warning(f"유효하지 않은 URL 형식: {url}")
        return {
            "module": "URL_Analyzer",
            "is_malicious": 0,
            "confidence_score": 0.0,
            "detected_features": ["유효하지 않은 URL 형식 (http:// 또는 https:// 로 시작해야 함)"]
        }
    
>>>>>>> Stashed changes
    # Google Safe Browsing API 요청 body 구성
    payload = {
        "client" : {
            "clientId" : "AutoGuard",                       # 프로젝트 이름
            "clientVersion": "1.0.0"                        # 버전
        },
        "threatInfo" : {
            "threatTypes" : [                               # 검사할 위험 유형
                "MALWARE",                                  # 악성코드
                "SOCIAL_ENGINEERING",                       # 피싱
                "UNWANTED_SOFTWARE",                        # 원치 않는 소프트웨어
                "POTENTIALLY_HARMFUL_APPLICATION"           # 유해 앱
            ],
            "platformTypes": ["ANY_PLATFORM"],              # 모든 플랫폼
            "threatEntryTypes": ["URL"],                    # URL 검사
            "threatEntries": [{"url": url}]                 # 검사할 URL
        }
    }
<<<<<<< Updated upstream
    
    response = requests.post(
        f"https://safebrowsing.googleapis.com/v4/threatMatches:find?key={GOOGLE_API_KEY}",
        json = payload
    )
    
=======

    try:
        # async with 으로 비동기 클라이언트 사용 (timeout 10초 설정)
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"https://safebrowsing.googleapis.com/v4/threatMatches:find?key={GOOGLE_API_KEY}",
                json = payload
            )

    except httpx.TimeoutException:                          # API 응답 시간 초과
        logger.error(f"Safe Browsing API 응답 시간 초과: {url}")
        return {
            "module": "URL_Analyzer",
            "is_malicious": 0,
            "confidence_score": 0.0,
            "detected_features": ["Safe Browsing API 응답 시간 초과"]
        }
    except httpx.RequestError:                              # API 연결 실패
        logger.error(f"Safe Browsing API 연결 실패: {url}")
        return {
            "module": "URL_Analyzer",
            "is_malicious": 0,
            "confidence_score": 0.0,
            "detected_features": ["Safe Browsing API 연결 실패"]
        }

>>>>>>> Stashed changes
    if response.status_code == 200:
        logger.info(f"Safe Browsing 분석 완료: {url} | 악성: {is_malicious}")
        data = response.json()
        is_malicious = 1 if "matches" in data else 0       # matches 있으면 악성
        
        return {
            "module" : "URL_Analyzer",
            "is_malicious" : is_malicious,
<<<<<<< Updated upstream
            # 구글이 탐지하면 확실한 거라 1.0
            "confidence_score" : 1.0 if is_malicious else 0.0,
=======
            # 악성 비율 (0~1.0)
            "confidence_score": 1.0 if is_malicious else 0.0,
>>>>>>> Stashed changes
            "detected_features" : [
                f"위협 유형: {m['threatType']}" for m in data.get("matches", [])
            ] if is_malicious else ["위협 없음"]
        }
    
    else:
        return {
            "error": "Safe Browsing API 호출 실패",
            "status_code": response.status_code
        }