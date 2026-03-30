# tools/intel_agent.py

import os
import httpx
import logging
import json
from tavily import TavilyClient  # [추가] Tavily 클라이언트 임포트

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 현재 로컬에서 돌아가는 FastAPI 서버 주소
FASTAPI_URL = "http://127.0.0.1:8000"

class IntelAgent:
    def __init__(self):
        self.name = "AutoGuard Intel Engine"
        # [추가] Tavily 클라이언트 초기화
        api_key = os.getenv("TAVILY_API_KEY")
        if api_key:
            self.tavily = TavilyClient(api_key=api_key)
        else:
            self.tavily = None
            logger.warning("[!] TAVILY_API_KEY가 설정되지 않았습니다.")

    async def search_web(self, query: str) -> dict:
        """
        [진짜 웹 서치] Tavily AI를 사용하여 최신 보안 위협 정보를 검색합니다.
        """
        if not self.tavily:
            return {"error": "Tavily API key is missing."}

        logger.info(f"[*] Tavily 웹 검색 요청: {query}")
        
        try:
            # [수정] 기존 FastAPI 호출 대신 Tavily API 사용
            # search_depth="smart"는 더 정밀한 검색을 수행합니다.
            response = self.tavily.search(
                query=query, 
                search_depth="advanced", # 'basic'보다 더 깊은 분석을 수행합니다.
                max_results=5
            )
            
            logger.info(f"[+] 검색 완료: {query} (결과 {len(response.get('results', []))}건)")
            return response # 에이전트가 읽을 수 있도록 검색 결과 리스트 반환
            
        except Exception as e:
            logger.error(f"[-] Tavily 검색 중 오류 발생: {e}")
            return {"error": str(e)}

    # predict_file_malicious 함수는 그대로 두시면 됩니다. (해시 분석용)
    async def predict_file_malicious(self, file_hash: str) -> dict:
        """ [파일 해시 분석] 기존 로직 유지 """
        logger.info(f"[*] 파일 해시 분석 요청: {file_hash}")
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{FASTAPI_URL}/analyze/file",
                    params={"hash": file_hash}
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"[!] 파일 분석 중 오류 발생: {e}")
            return {"error": str(e)}
        
        
# 웹 서치 기능을 테스트하기 위한 간단한 실행 코드
if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv
    
    # .env 파일 로드 (TAVILY_API_KEY가 들어있어야 함)
    load_dotenv()

    async def test():
        intel = IntelAgent()
        print("\n[*] Tavily 웹 검색 테스트 시작...")
        # 테스트 질문: 최근 보안 뉴스 검색
        result = await intel.search_web("2026 current cyber security threats ransomware")
        
        if "error" in result:
            print(f"[-] 테스트 실패: {result['error']}")
        else:
            print(f"[+] 테스트 성공! 검색 결과 {len(result.get('results', []))}건 발견")
            for i, res in enumerate(result.get('results', []), 1):
                print(f"  {i}. {res.get('title')} ({res.get('url')})")

    asyncio.run(test())