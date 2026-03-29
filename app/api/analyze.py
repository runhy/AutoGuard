# app/api/analyze.py

from fastapi import APIRouter
import asyncio
from fastapi import APIRouter, UploadFile, File
from app.api.virustotal import scan_url as vt_scan        # virustotal 함수 가져오기
from app.api.safebrowsing import scan_url as sb_scan      # safebrowsing 함수 가져오기
from app.api.file_scan import scan_file as file_scan
# [추가] 자체 ML 모델 분석 함수
from app.ml.scripts.url import analyze_url as internal_url_inference
from app.agents.tools.analyzer_agent import AnalyzerAgent

router = APIRouter()
agent = AnalyzerAgent() # 파일 분석 등을 위해 에이전트 생성

# /analyze/scan?url=검사할URL 요청 처리
# 하이브리드 URL 분석 (내부 ML + 외부 인텔)
@router.get("/scan")
async def analyze_url(url: str):
    # VirusTotal + SafeBrowsing + 자체 ML 모델 동시에 호출 (병렬 처리)
    # 자체 모델은 동기 함수일 수 있으므로 run_in_executor 등으로 감싸거나 그대로 호출
    loop = asyncio.get_event_loop()
    
    # 세 가지 분석을 동시에 진행 (병렬 처리로 속도 최적화)
    vt_task = vt_scan(url)
    sb_task = sb_scan(url)
    # 자체 모델 호출 (동기 함수인 경우를 대비해 래핑)
    internal_task = loop.run_in_executor(None, internal_url_inference, url)

    vt_result, sb_result, internal_res = await asyncio.gather(
        vt_scan,
        sb_scan,
        internal_task
    )
    
    # 둘 중 하나라도 악성이면 악성으로 판단
    is_malicious = 1 if (vt_result["is_malicious"] or sb_result["is_malicious"]) else 0
    
    return {
        "module": "URL_Analyzer",
        "is_malicious": is_malicious,
        "internal_ml": internal_res,        # [추가] 모델 결과
        "confidence_score": max(vt_result["confidence_score"], sb_result["confidence_score"]),
        "detected_features": vt_result["detected_features"] + sb_result["detected_features"],
        "sources": {
            "virustotal": vt_result,
            "safebrowsing": sb_result
        }
    }
    
# 2. 파일 업로드 분석 엔드포인트 (POST /analyze/file)
@router.post("/file")
async def analyze_file(file: UploadFile = File(...)):
    return await file_scan(file)

# 3. 파일 해시 기반 분석 엔드포인트 (GET /analyze/file?hash=...)
@router.get("/file")
async def analyze_file_hash(hash: str):
    # VirusTotal의 파일 보고서 기능을 호출하도록 연결
    from app.api.virustotal import get_file_report
    return await get_file_report(hash)