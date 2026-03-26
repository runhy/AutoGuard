# app/api/analyze.py

from fastapi import APIRouter
import asyncio
from app.api.virustotal import scan_url as vt_scan        # virustotal 함수 가져오기
from app.api.safebrowsing import scan_url as sb_scan      # safebrowsing 함수 가져오기

router = APIRouter()

# /analyze/scan?url=검사할URL 요청 처리
@router.get("/scan")
async def analyze_url(url: str):
    # VirusTotal + SafeBrowsing 동시에 호출 (병렬 처리)
    vt_result, sb_result = await asyncio.gather(
        vt_scan(url),
        sb_scan(url)
    )
    
    # 둘 중 하나라도 악성이면 악성으로 판단
    is_malicious = 1 if (vt_result["is_malicious"] or sb_result["is_malicious"]) else 0
    
    return {
        "module": "URL_Analyzer",
        "is_malicious": is_malicious,
        "confidence_score": max(vt_result["confidence_score"], sb_result["confidence_score"]),
        "detected_features": vt_result["detected_features"] + sb_result["detected_features"],
        "sources": {
            "virustotal": vt_result,
            "safebrowsing": sb_result
        }
    }