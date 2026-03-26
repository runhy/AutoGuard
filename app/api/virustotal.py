# app/api/virustotal.py

from fastapi import APIRouter
import requests
import os
from dotenv import load_dotenv

# .env 파일 로더
load_dotenv()

# 라우터 인스턴스 생성
router = APIRouter()
VIRUSTOTAL_API_KEY = os.getenv("VIRUSTOTAL_API_KEY")

# 검사할URL 요청 처리
@router.get("/scan")
def scan_url(url: str):                     # url은 쿼리 파라미터로 받음
    headers = {
        "x-apikey" : VIRUSTOTAL_API_KEY     # API key을 헤더의 담아 인증
    }
    
    # 1단계: VirusTotal에 URL 제출 (분석 요청)
    response = requests.post(
        "https://www.virustotal.com/api/v3/urls",
        headers = headers,
        data = {
            "url" : url
        }
    )
    
    # 오류 로그
    if response.status_code != 200:
        return {
            "error" : "URL 제출 실패",
            "status_code" : response.status_code
        }
        
    # 2단계: 제출 후 받은 분석 ID로 결과 조회
    analysis_id = response.json()["data"]["id"]                 # 분석 id 추출
    result = requests.get(                                      # response → requests 로 수정
        f"https://www.virustotal.com/api/v3/analyses/{analysis_id}",
        headers = headers
    )
    
    # 성공
    if result.status_code == 200:
        stats = result.json()["data"]["attributes"]["stats"]    # 분석 통계 추출
        return {
            "module": "URL_Analyzer",
            # 악성 엔진 1개라도 있으면 1
            "is_malicious": 1 if stats["malicious"] > 0 else 0,
            # 악성 비율
            "confidence_score": round(stats["malicious"] / (stats["malicious"] + stats["harmless"] + stats["undetected"] + 0.001), 2),
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
            "error" : "결과 조회 실패",
            "status_code" : result.status_code
        }