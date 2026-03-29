# app/api/safebrowsing.py

from fastapi import APIRouter
import httpx
import os
import logging
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()
GOOGLE_API_KEY = os.getenv("GOOGLE_SAFE_BROWSING_API_KEY")

logger = logging.getLogger(__name__)

@router.get("/scan")
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
    
    # Google Safe Browsing API 요청 body 구성
    payload = {
        "client" : {
            "clientId" : "AutoGuard",
            "clientVersion": "1.0.0"
        },
        "threatInfo" : {
            "threatTypes" : [
                "MALWARE",
                "SOCIAL_ENGINEERING",
                "UNWANTED_SOFTWARE",
                "POTENTIALLY_HARMFUL_APPLICATION"
            ],
            "platformTypes": ["ANY_PLATFORM"],
            "threatEntryTypes": ["URL"],
            "threatEntries": [{"url": url}]
        }
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"https://safebrowsing.googleapis.com/v4/threatMatches:find?key={GOOGLE_API_KEY}",
                json = payload
            )

    except httpx.TimeoutException:
        logger.error(f"Safe Browsing API 응답 시간 초과: {url}")
        return {
            "module": "URL_Analyzer",
            "is_malicious": 0,
            "confidence_score": 0.0,
            "detected_features": ["Safe Browsing API 응답 시간 초과"]
        }
    except httpx.RequestError:
        logger.error(f"Safe Browsing API 연결 실패: {url}")
        return {
            "module": "URL_Analyzer",
            "is_malicious": 0,
            "confidence_score": 0.0,
            "detected_features": ["Safe Browsing API 연결 실패"]
        }

    # --- [수정 구간 시작] ---
    if response.status_code == 200:
        data = response.json()
        
        # 1. 변수 정의를 로그 출력보다 먼저 수행 (UnboundLocalError 방지)
        is_malicious = 1 if "matches" in data else 0 

        # 2. 이제 변수가 정의되었으므로 안전하게 로그 출력 가능
        logger.info(f"Safe Browsing 분석 완료: {url} | 악성: {is_malicious}")

        return {
            "module" : "URL_Analyzer",
            "is_malicious" : is_malicious,
            "confidence_score": 1.0 if is_malicious else 0.0,
            "detected_features" : [
                f"위협 유형: {m['threatType']}" for m in data.get("matches", [])
            ] if is_malicious else ["위협 없음"]
        }
    # --- [수정 구간 끝] ---

    elif response.status_code == 429:
        return {
            "module": "URL_Analyzer",
            "is_malicious": 0,
            "confidence_score": 0.0,
            "detected_features": ["Safe Browsing API 요청 한도 초과 - 잠시 후 다시 시도"]
        }

    else:
        return {
            "error": "Safe Browsing API 호출 실패",
            "status_code": response.status_code
        }