# tools/intel_agent.py

import httpx
import logging
import json

# 로깅 설정 (콘솔에서 분석 흐름 확인용)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 현재 로컬에서 돌아가는 FastAPI 서버 주소
FASTAPI_URL = "http://127.0.0.1:8000"

class IntelAgent:
    def __init__(self):
        self.name = "AutoGuard Intel Engine"

    async def search_web(self, query: str) -> dict:
        """
        [URL 분석] 에이전트가 호출하면 FastAPI의 /analyze/scan 엔드포인트를 찌릅니다.
        (VirusTotal + SafeBrowsing 통합 결과 반환)
        """
        logger.info(f"[*] URL 분석 요청: {query}")
        
        try:
            # 타임아웃 40초 (VT 스캔 대기 시간 고려)
            async with httpx.AsyncClient(timeout=40.0) as client:
                response = await client.get(
                    f"{FASTAPI_URL}/analyze/scan",
                    params={"url": query}
                )
                response.raise_for_status() # 200 아니면 에러 발생
                
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
        """
        [파일 해시 분석] SHA-256 해시값을 기반으로 FastAPI 서버에 조회를 요청합니다.
        """
        logger.info(f"[*] 파일 해시 분석 요청: {file_hash}")
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{FASTAPI_URL}/analyze/file",
                    params={"hash": file_hash}
                )
                response.raise_for_status()
                
                result = response.json()
                is_malicious = result.get('is_malicious', 'unknown')
                logger.info(f"[+] 해시 분석 완료: {file_hash} | 결과: {is_malicious}")
                
                return result
            
        except Exception as e:
            logger.error(f"[!] 파일 분석 중 오류 발생: {e}")
            return {"error": str(e)}