# tools/vt_tool.py

import httpx
import logging

logger = logging.getLogger(__name__)

# FastAPI 주소
FASTAPI_URL = "http://127.0.0.1:8000"

async def analyze_url(url: str) -> dict:
    # 임시 프롬프트
    '''
    URL의 악성 여부를 VirusTotal + SafeBrowsing API로 분석합니다.
    FastAPI /analyze/scan 엔드포인트를 호출합니다.
    '''
    # 로그 확인용
    logger.info(f"URL 분석 요청: {url}")
    
    try:
        # VirusTotal 30초 대기 고려
        async with httpx.AsyncClient(timeout=40.0) as client:
            response = await client.get(
                f"{FASTAPI_URL}/analyze/scan",
                params = {
                    "url" : url
                }
            )
            result = response.json()
            logger.info(f"URL 분석 완료: {url} | 악성: {result.get('is_malicious')}")
            return result
        
    except httpx.TimeoutException:
        logger.error(f"URL 분석 타임아웃: {url}")
        return {
            "module": "File_Analyzer",
            "is_malicious": 0,
            "confidence_score": 0.0,
            "detected_features": ["분석 서버 연결 실패"]
        }
        
    
