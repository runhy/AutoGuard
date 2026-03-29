# app/api/analyze.py
# asyncio.gather의 치명적 오류를 수정하고 내부 모델의 분석 결과까지 완벽하게 통합한 버전


from fastapi import APIRouter, UploadFile, File, HTTPException
import asyncio
from app.api.virustotal import scan_url as vt_scan        # virustotal 함수 가져오기
from app.api.safebrowsing import scan_url as sb_scan      # safebrowsing 함수 가져오기
from app.api.file_scan import scan_file as file_scan
# [추가] 수정한 자체 ML 엔진
from ml.scripts.url import analyze_url as internal_url_inference
from app.agents.tools.analyzer_agent import AnalyzerAgent


router = APIRouter()
# [추가] AnalyzerAgent
agent = AnalyzerAgent()

# [추가] 하이브리드 URL 분석 (내부 ML + 외부 인텔)
# /analyze/scan?url=검사할URL 요청 처리
@router.get("/scan")
async def analyze_url_endpoint(url: str):
    """
    내부 ML 모델과 외부 위협 인텔리전스(VT, SB)를 병렬로 호출하여
    결과를 교차 검증하고 최종 판정을 내립니다.
    """
    loop = asyncio.get_event_loop()
    
    # [수정] 비동기 작업 예약 (Task 생성)
    vt_task = vt_scan(url)
    sb_task = sb_scan(url)
    # 자체 모델은 동기 함수이므로 별도 스레드에서 실행하여 병목 방지
    internal_task = loop.run_in_executor(None, internal_url_inference, url)

    # [수정] 함수 자체가 아닌 생성된 'Task'들을 gather에 전달해야 함
    vt_result, sb_result, internal_res = await asyncio.gather(
        vt_task, 
        sb_task, 
        internal_task
    )
    
    # [수정 : 논리 통합] 셋 중 하나라도 악성이면 악성(1)으로 판단
    is_malicious = 1 if (
        vt_result.get("is_malicious") or 
        sb_result.get("is_malicious") or 
        internal_res.get("is_malicious")
    ) else 0
    
    # [결과 병합] 모든 탐지 피처와 소스를 하나로 합침
    return {
        "module": "Hybrid_URL_Analyzer", # [변경] 하이브리드임을 명시
        "is_malicious": is_malicious,    # 셋 중 하나라도 1이면 1
        "final_confidence": max(         # 가장 높은 신뢰도 점수 채택
        internal_res.get("confidence_score", 0), 
        vt_result.get("confidence_score", 0)
        ),
        "internal_ml_report": internal_res, # 모델의 핵심 (상세 피처 포함)
        "external_sources": {               # 외부 인텔리전스는 '참고용'으로 그룹화
        "virustotal": vt_result,
        "safebrowsing": sb_result
        },
        "all_detected_features": list(set(  # 중복 없는 통합 탐지 근거
        internal_res.get("detected_features", []) + 
        vt_result.get("detected_features", []) + 
        sb_result.get("detected_features", [])
        ))
    }


# [수정] 파일 업로드 분석 엔드포인트
@router.post("/file")
async def analyze_file_upload(file: UploadFile = File(...)):
    return await file_scan(file)

# [추가] 파일 해시 기반 분석 엔드포인트
@router.get("/file")
async def analyze_file_hash(hash: str):
    """
    [수정] 존재하지 않는 get_file_report 호출로 인한 500 에러 방지.
    파일 분석은 AnalyzerAgent를 통해 처리하도록 안내하거나 예외 처리.
    """
    try:
        from app.api.virustotal import get_file_report
        return await get_file_report(hash)
    except (ImportError, AttributeError):
        return {
            "module": "File_Hash_Analyzer",
            "hash": hash,
            "message": "해시 분석 기능은 현재 IntelAgent를 통해 제공됩니다."
        }