# app/api/safebrowsing.py

from fastapi import APIRouter
import httpx
import os
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()
GOOGLE_API_KEY = os.getenv("GOOGLE_SAFE_BROWSING_API_KEY")

@router.get("/scan")
async def scan_url(url: str):
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
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"https://safebrowsing.googleapis.com/v4/threatMatches:find?key={GOOGLE_API_KEY}",
            json = payload
        )
    
    
    if response.status_code == 200:
        data = response.json()
        is_malicious = 1 if "matches" in data else 0       # matches 있으면 악성
        
        return {
            "module" : "URL_Analyzer",
            "is_malicious" : is_malicious,
            # 구글이 탐지하면 확실한 거라 1.0
            "confidence_score" : 1.0 if is_malicious else 0.0,
            "detected_features" : [
                f"위협 유형: {m['threatType']}" for m in data.get("matches", [])
            ] if is_malicious else ["위협 없음"]
        }
    
    else:
        return {
            "error": "Safe Browsing API 호출 실패",
            "status_code": response.status_code
        }