# ml/scripts/url.py
# URL의 구조적 특징 파싱 및 악성 여부 판단

import pickle
import pandas as pd
import re
import math
import sys
import __main__  # [추가] Pickle 로드 시 함수 참조 에러 방지용
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# ==========================================================
# [추가] Pickle 참조 에러 방지 구역
# Pickle 모델 로드 시 학습 때 사용했던 함수 설계도가 필요.
# ==========================================================

# [추가] Pickle이 모델을 로드할 때 참조할 함수 설계도들(엔트로피 계산 함수) : 설계도
def _entropy(s):
    if not s: return 0.0
    freq = [s.count(c) / len(s) for c in set(s)]
    return round(-sum(p * math.log2(p) for p in freq), 2) 

# [추가] url_model.py 코드에서 가져온 특징 추출 함수 : 설계도
def extract_features_from_url(url: str) -> dict:
    url = str(url).strip()
    try:
        parsed = urlparse(url)
    except Exception:
        parsed = urlparse("")

    domain   = parsed.netloc or ""
    path     = parsed.path or ""
    query    = parsed.query or ""
    fragment = parsed.fragment or ""

    lowercase = sum(c.islower() for c in url)
    uppercase = sum(c.isupper() for c in url)
    digits    = sum(c.isdigit() for c in url)
    url_len   = len(url)

    tld_match = re.search(r"\.([a-zA-Z]{2,})$", domain)
    tld       = tld_match.group(1).lower() if tld_match else "unknown"

    IANA_TLDS = {"com","net","org","edu","gov","mil","int","io","co",
                 "uk","de","jp","fr","au","us","ru","cn","br","se","kr"}

    special_chars = sum(
        1 for c in url if not c.isalnum() and c not in "/:.-_?=&#%~@"
    )


    return {
        "dots":           url.count("."),
        "at":             url.count("@"),
        "equals":         url.count("="),
        "slashes":        url.count("/"),
        "hyphens":        url.count("-"),
        "colons":         url.count(":"),
        "question_marks": url.count("?"),
        "digits":         digits,
        "and":            url.count("&"),
        "underscore":     url.count("_"),
        "tilde":          url.count("~"),
        "percent":        url.count("%"),
        "lowercase":      lowercase,
        "uppercase":      uppercase,
        "upper_to_lower_ratio": round(uppercase / lowercase, 4) if lowercase else 0.0,
        "is_https":       int(url.startswith("https")),
        "url_length":     url_len,
        "domain_length":  len(domain),
        "path_length":    len(path),
        "path_depth":     path.count("/"),
        "query_length":   len(query),
        "query_count":    len(parse_qs(query)),
        "fragment_length": len(fragment),
        "se_url":         _entropy(url),
        "se_domain":      _entropy(domain),
        "se_path":        _entropy(path),
        "se_query":       _entropy(query),
        "se_fragment":    _entropy(fragment),
        "cte_domain":     _entropy(domain),
        "is_domain_ip":   int(bool(re.fullmatch(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", domain))),
        "is_tld_iana_reg": int(tld in IANA_TLDS),
        "is_mtld":        int(tld in {"co.uk","com.au","co.jp"}),
        "subdomains":     max(0, len(domain.split(".")) - 2),
        "special_chars":  special_chars,
        "digit_to_length_ratio":       round(digits / url_len, 4) if url_len else 0.0,
        "char_to_length_ratio":        round(lowercase / url_len, 4) if url_len else 0.0,
        "specialchar_to_length_ratio": round(special_chars / url_len, 4) if url_len else 0.0,
        "tld": tld,   # 범주형 — 파이프라인에서 OneHotEncoding 처리
    }


# [추가] url_model.py 코드에서 가져온 의심 패턴 탐지 함수 : 설계도
def get_detected_features(features: dict, top_features: list) -> list:
    detected = []
    if features.get("is_https") == 0:
        detected.append("HTTP 사용 (암호화 없음)")
    if features.get("is_domain_ip") == 1:
        detected.append("도메인에 IP 주소 직접 사용")
    if features.get("subdomains", 0) >= 3:
        detected.append(f"과다 서브도메인 ({features['subdomains']}개)")
    if features.get("url_length", 0) > 100:
        detected.append(f"비정상적으로 긴 URL ({features['url_length']}자)")
    if features.get("hyphens", 0) >= 4:
        detected.append(f"하이픈 과다 ({features['hyphens']}개)")
    if features.get("digits", 0) > 15:
        detected.append(f"숫자 과다 ({features['digits']}개)")
    if features.get("percent", 0) > 5:
        detected.append(f"URL 인코딩 과다 ({features['percent']}개)")
    if features.get("se_domain", 0) > 4.5:
        detected.append(f"도메인 엔트로피 높음 ({features['se_domain']})")
    
    # top_features가 pkl에 없을 경우를 대비한 안전 장치
    return detected if detected else (top_features[:5] if top_features else ["정상적인 특징 패턴"])

# [수정] Pickle 로드 전 네임스페이스 주입
# pickle이 __main__에서 함수를 찾으려 할 때 이 파일에 정의된 함수를 보여주도록 연결
__main__.extract_features_from_url = extract_features_from_url
__main__.get_detected_features = get_detected_features

# 모델 로드
# [수정] 상대 경로("./") 대신 파일 위치 기준 절대 경로 사용 : 서버 안정성 확보
_MODEL_PATH = Path(__file__).parent.parent / "models" / "url_model.pkl"
with open(_MODEL_PATH, "rb") as f:
    _loaded = pickle.load(f)
# _loaded = pickle.load(open(_MODEL_PATH, "rb"))

# 변수명 그대로 사용(의존성 방지)
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
    """
    설계된 분석 프로세스를 서버 환경에 맞춰 실행합니다.
    """
    # 1. 피처 추출
    feat = _extract_features(url)
 
    # 2. 모델 입력 형식으로 변환
    X_in = pd.DataFrame([feat])[_numeric_features + _categorical_features]
 
    # 3. 추론
    pred  = int(_pipeline.predict(X_in)[0])
    proba = float(_pipeline.predict_proba(X_in)[0][1])
 
    # 4. 탐지 피처 서술
    detected = _get_detected_features(feat, [])
 
    # 결과 반환 : JSON 규격을 통일하여 에이전트 호환성 확보
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