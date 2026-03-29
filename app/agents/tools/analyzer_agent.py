# app/agents/tools/analyzer_agent.py
# [수정] 기존 복잡한 로직(엔트로피 계산, 피처 맵 등)을 모두 제거 -> 전문 엔진들에 분석을 위임하는 구조로 간결하게 수정

import os
import sys

# [필수] 프로젝트 루트 및 ml/scripts 경로를 시스템 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../../'))
scripts_path = os.path.join(project_root, 'ml', 'scripts')

if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

# [수정] 전문 추론 엔진 모듈에서 분석 함수 임포트
try:
    from url import analyze_url as url_inference
    from mal import analyze_file as file_inference
    from mail import analyze_mail as email_inference
    print("[*] 모든 전문 분석 엔진(URL, File, Email) 임포트 성공")
except ImportError as e:
    print(f"[!] 엔진 임포트 실패: {e}")
    # 분석 엔진 파일(url.py, mal.py, mail.py)이 ml/scripts 폴더에 있는지 확인 필요
    raise


class AnalyzerAgent:
    """
    DispatcherAgent의 요청을 받아 전문 ML 엔진으로 전달하는 오케스트레이터입니다.
    직접적인 로직을 갖지 않고, 독립된 전문 모듈에 분석을 위임합니다.
    """
    def __init__(self):
        # [수정] 모델 로드 및 피처 추출 로직은 이제 각 전문 스크립트(url.py, mal.py 등)가 담당합니다.
        pass
        # print(f"[*] 모델 탐색 경로: {base_path}") # 디버깅용 출력


    def analyze_url(self, url: str) -> dict:
        """전문 모듈인 url.py로 분석 위임"""
        try:
            return url_inference(url)
        except Exception as e:
            return {"module": "URL_Analyzer", "error": str(e)}

    def analyze_file(self, path: str) -> dict:
        """전문 모듈인 mal.py로 분석 위임"""
        try:
            return file_inference(path)
        except Exception as e:
            return {"module": "File_Analyzer", "error": str(e)}

    def analyze_email(self, text: str, attachments: list = None) -> dict:
        """전문 모듈인 mail.py로 분석 위임"""
        try:
            return email_inference(text, attachments)
        except Exception as e:
            return {"module": "Email_Analyzer", "error": str(e)}