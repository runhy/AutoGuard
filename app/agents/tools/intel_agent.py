# tools/intel_agent.py

import httpx
import logging
import asyncio

logger = logging.getLogger(__name__)

# 내부 분석 로직을 구축한 FastAPI 엔드포인트와 연결했습니다.
FASTAPI_URL = "http://127.0.0.1:8000"

class IntelAgent:
    def __init__(self):
        # 정해주신 엔진 명칭 유지
        self.name = "AutoGuard Intel Engine"

    async def search_web(self, query: str) -> dict:
        """
        이름은 search_web이지만, 
        내부적으로는 VirusTotal + SafeBrowsing 통합 분석(FastAPI)을 호출하도록 구현했습니다.
        """
        logger.info(f"[*] 인텔 분석 요청(Query/URL): {query}")
        
        try:
            # VT 분석 대기 시간을 고려하여 40초 타임아웃 적용
            async with httpx.AsyncClient(timeout=40.0) as client:
                response = await client.get(
                    f"{FASTAPI_URL}/analyze/scan",
                    params={"url": query}
                )
                result = response.json()
                logger.info(f"[+] 분석 완료: {query} | 결과: {result.get('is_malicious')}")
                return result
        
        except httpx.TimeoutException:
            logger.error(f"[-] 분석 타임아웃: {query}")
            return {
                "module": "Intel_Agent",
                "is_malicious": 0,
                "confidence_score": 0.0,
                "detected_features": ["분석 서버 응답 지연(Timeout)"]
            }
        except Exception as e:
            logger.error(f"[-] 분석 중 오류 발생: {e}")
            return {"error": str(e)}

    async def predict_file_malicious(self, file_hash: str) -> dict:
        """ [확장용] 파일 해시 기반 악성 여부 조회 (준비 중) """
        return {"module": "File_Analyzer", "status": "pending", "hash": file_hash}