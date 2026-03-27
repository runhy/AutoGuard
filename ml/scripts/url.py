import pickle
import pandas as pd
from pathlib import Path
 
# 모델 로드
_MODEL_PATH = Path(__file__).parent.parent / "models" / "url_model.pkl"
_loaded = pickle.load(open(_MODEL_PATH, "rb"))
 
_pipeline              = _loaded["pipeline"]
_numeric_features      = _loaded["numeric_features"]
_categorical_features  = _loaded["categorical_features"]
_extract_features      = _loaded["extract_features"]
_get_detected_features = _loaded["get_detected_features"]
 
 
# 메인 분석 함수 병합 시
# from url import analyze_url
# analyze_url 함수 사용
# result = analyze_url("https://example.com/login")
def analyze_url(url: str) -> dict:
    # 1. 피처 추출
    feat = _extract_features(url)
 
    # 2. 모델 입력 형식으로 변환
    X_in = pd.DataFrame([feat])[_numeric_features + _categorical_features]
 
    # 3. 추론
    pred  = int(_pipeline.predict(X_in)[0])
    proba = float(_pipeline.predict_proba(X_in)[0][1])
 
    # 4. 탐지 피처 서술
    detected = _get_detected_features(feat, [])
 
    result_json = {
        # 분석 모듈 식별자
        "module":           "URL_Analyzer",
 
        # 악성 여부 판정
        # 0: 정상 / 1: 악성
        "is_malicious":     pred,
 
        # ML 모델의 악성 확률값
        "confidence_score": round(proba, 4),
 
        # 탐지된 특이 패턴 설명 목록
        # ex) ["HTTP 사용 (암호화 없음)", "도메인에 IP 주소 직접 사용"]
        # ex) []  ← 특이 패턴 없음
        "detected_features": detected,
    }
    return result_json