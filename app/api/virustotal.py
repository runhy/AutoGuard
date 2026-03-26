# app/api/virustotal.py

from fastapi import APIRouter
import httpx                    # requests 대신 비동기 지원 라이브러리
import os
import asyncio                  # 대기 시간 지정
import hashlib                  # URL 해시값 변환
from dotenv import load_dotenv

# .env 파일 로더
load_dotenv()

# 라우터 인스턴스 생성
router = APIRouter()
VIRUSTOTAL_API_KEY = os.getenv("VIRUSTOTAL_API_KEY")

# 검사할URL 요청 처리(비동기 처리)
@router.get("/scan")
async def scan_url(url: str):                     # url은 쿼리 파라미터로 받음
    headers = {
        "x-apikey" : VIRUSTOTAL_API_KEY           # API key을 헤더의 담아 인증
    }

    try:
        # async with 으로 비동기 클라이언트 사용 (timeout 10초 설정)
        async with httpx.AsyncClient(timeout=10.0) as client:

            # 1단계: URL 해시값으로 먼저 조회 (이미 분석된 URL이면 즉시 반환)
            url_hash = hashlib.sha256(url.encode()).hexdigest()  # URL → SHA256 해시 변환
            cached = await client.get(
                f"https://www.virustotal.com/api/v3/urls/{url_hash}",
                headers=headers
            )

            if cached.status_code == 200:                       # 이미 분석된 URL이면 바로 반환
                stats = cached.json()["data"]["attributes"]["last_analysis_stats"]
                return {
                    "module": "URL_Analyzer",
                    "is_malicious": 1 if stats["malicious"] > 0 else 0,
                    # 악성 비율 (0~1.0)
                    "confidence_score": round((stats["malicious"] / (stats["malicious"] + stats["harmless"] + stats["undetected"] + 0.001)), 2),
                    "detected_features": [
                        f"악성 엔진 수: {stats['malicious']}",
                        f"의심 엔진 수: {stats['suspicious']}",
                        f"안전 엔진 수: {stats['harmless']}",
                        f"미탐지 엔진 수: {stats['undetected']}"
                    ]
                }

            # 2단계: 캐시 없으면 URL 제출 (분석 요청)
            response = await client.post(
                "https://www.virustotal.com/api/v3/urls",
                headers=headers,
                data={"url": url}
            )

            # 오류 로그
            if response.status_code == 429:                     # API 요청 한도 초과
                return {
                    "module": "URL_Analyzer",
                    "is_malicious": 0,
                    "confidence_score": 0.0,
                    "detected_features": ["VirusTotal API 요청 한도 초과 - 잠시 후 다시 시도"]
                }

            if response.status_code != 200:
                return {
                    "error": "URL 제출 실패",
                    "status_code": response.status_code
                }

            # 3단계: 분석 ID 추출 후 완료될 때까지 최대 30초 대기
            analysis_id = response.json()["data"]["id"]         # 분석 id 추출

            MAX_WAIT = 30          # 최대 대기 시간 30초
            elapsed = 0            # 경과 시간

            while elapsed < MAX_WAIT:
                result = await client.get(
                    f"https://www.virustotal.com/api/v3/analyses/{analysis_id}",
                    headers=headers
                )
                if result.status_code == 200:
                    status = result.json()["data"]["attributes"]["status"]  # 분석 완료 여부 확인
                    if status == "completed":                               # 완료되면 탈출
                        break
                await asyncio.sleep(2)    # 2초 대기
                elapsed += 2              # 경과시간 누적

            # 30초 넘어도 안 끝나면 에러 반환
            if elapsed >= MAX_WAIT:
                return {
                    "error": "분석 시간 초과",
                    "status_code": 408
                }

    except httpx.TimeoutException:                              # API 응답 시간 초과
        return {
            "module": "URL_Analyzer",
            "is_malicious": 0,
            "confidence_score": 0.0,
            "detected_features": ["VirusTotal API 응답 시간 초과"]
        }
    except httpx.RequestError:                                  # API 연결 실패
        return {
            "module": "URL_Analyzer",
            "is_malicious": 0,
            "confidence_score": 0.0,
            "detected_features": ["VirusTotal API 연결 실패"]
        }

    # 성공
    if result.status_code == 200:
        stats = result.json()["data"]["attributes"]["stats"]    # 분석 통계 추출
        return {
            "module": "URL_Analyzer",
            # 악성 엔진 1개라도 있으면 1
            "is_malicious": 1 if stats["malicious"] > 0 else 0,
            # 악성 비율 (0~1.0)
            "confidence_score": round((stats["malicious"] / (stats["malicious"] + stats["harmless"] + stats["undetected"] + 0.001)), 2),
            "detected_features": [
                f"악성 엔진 수: {stats['malicious']}",
                f"의심 엔진 수: {stats['suspicious']}",
                f"안전 엔진 수: {stats['harmless']}",
                f"미탐지 엔진 수: {stats['undetected']}"
            ]
        }

    # 실패
    else:
        return {
            "error": "결과 조회 실패",
            "status_code": result.status_code
        }