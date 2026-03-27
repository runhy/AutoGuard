import re
import html
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import hstack, csr_matrix

# 모델 로드
_MODEL_PATH = Path(__file__).parent.parent / "models" / "spam_V2.pkl"
data = joblib.load(_MODEL_PATH)

model   = data["model"]
tfidf   = data["tfidf"]
scaler  = data["scaler"]
columns = data["v3_cols"]


#  메일 본문 정제
def clean_text(raw: str) -> str:

    #원본 메일 본문 → 순수 텍스트 반환

    #처리 순서
    # 1. HTML 엔티티 디코딩  (&amp; → &)
    # 2. <style> / <script> 블록 제거
    # 3. HTML 태그 전체 제거
    # 4. 메일 헤더 패턴 제거  (From:, To:, Subject: 등)
    # 5. Base64 잔여물 제거   (40자 이상 연속 영숫자)
    # 6. 연속 공백·탭 → 단일 공백 / 3줄 이상 줄바꿈 → 2줄
    # 7. 앞뒤 공백 제거

    if not raw:
        return ""

    text = html.unescape(raw)
    text = re.sub(r'<(style|script)[^>]*>.*?</\1>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(
        r'^(From|To|Cc|Bcc|Subject|Date|Message-ID|Content-Type'
        r'|Content-Transfer-Encoding|Reply-To|Received|X-[\w-]+)\s*:.*$',
        '', text, flags=re.MULTILINE | re.IGNORECASE
    )
    text = re.sub(r'[A-Za-z0-9+/=]{40,}', '', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# 수치 피처 추출
def _char_entropy(s):
    if not s:
        return 0.0
    freq = pd.Series(list(s)).value_counts(normalize=True)
    return float(-(freq * np.log2(freq + 1e-10)).sum())

_FEATURE_MAP = {
    'text_len':       lambda _t, _w, n: n,
    'word_count':     lambda _t, w, _n: len(w),
    'sentence_count': lambda t, _w, _n: len(re.findall(r'[.!?\n]', t)) + 1,
    'exclaim_count':  lambda t, _w, _n: t.count('!'),
    'digit_ratio':    lambda t, _w, n: len(re.findall(r'[0-9]', t))   / (n + 1),
    'special_ratio2': lambda t, _w, n: len(re.findall(r'[^\w\s]', t)) / (n + 1),
    'ko_ratio':       lambda t, _w, n: len(re.findall(r'[가-힣]', t)) / (n + 1),
    'avg_word_len':   lambda _t, w, n: n / (len(w) + 1),
    'space_ratio':    lambda t, _w, n: t.count(' ')                    / (n + 1),
    'star_count':     lambda t, _w, _n: t.count('*'),
    'url_count':      lambda t, _w, _n: len(re.findall(r'https?://|www\.|\.com|\.kr/', t)),
    'entropy':        lambda t, _w, _n: _char_entropy(t),
}

def extract_features(text: str) -> dict:
    t = text.strip()
    w = t.split()
    n = len(t)
    return {col: _FEATURE_MAP[col](t, w, n) for col in columns}


# TF-IDF + 수치 피처 결합 후 추론
def predict_mail(clean_txt: str) -> tuple:
    feature   = extract_features(clean_txt)
    X_tfidf   = tfidf.transform([clean_txt])
    X_num     = scaler.transform([[feature[c] for c in columns]])
    X_input   = hstack([X_tfidf, csr_matrix(X_num)])
    y_prob    = model.predict_proba(X_input)[:, 1]
    y_pred    = (y_prob > 0.5).astype(int)
    return y_pred[0], float(y_prob[0]), feature



# 메인 분석 함수
# 병합하실때
# from mail import analyze_mail -> 이거하신뒤
# analyze_mail 이 함수 사용하시면 됩니다
# result = analyze_mail("메일 본문", attachments=["파일명.exe"])
_FILE_EXTS = re.compile(
    r'\.(exe|dll|bat|cmd|ps1|vbs|js|jar|scr|pif|com|msi|reg'
    r'|doc|docx|xls|xlsx|ppt|pptx|pdf|hwp|zip|rar|7z|iso'
    r'|txt|png|jpg|jpeg|gif|svg|mp4|mp3|csv)$',
    re.IGNORECASE,
)

def analyze_mail(raw_text: str, attachments: list = None) -> dict:
    if attachments is None:
        attachments = []

    # 1. raw HTML에서 href URL / 첨부파일 선추출 (clean 작업전에)
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', raw_text, flags=re.IGNORECASE)
    href_urls   = [h for h in hrefs if re.match(r'https?://', h, re.IGNORECASE)]
    href_files  = [h.split('/')[-1] for h in hrefs if _FILE_EXTS.search(h)]

    # 2. 본문 정제
    clean_txt = clean_text(raw_text)

    # 3. 추론
    pred, prob, feature = predict_mail(clean_txt)

    # 4. URL / 첨부파일 탐지 (텍스트 + href 병합, 중복 제거)
    text_urls  = re.findall(r'https?://[^\s]+|www\.[^\s]+', clean_txt)
    found_urls = list(dict.fromkeys(href_urls + text_urls))   # 순서 유지 중복 제거
    all_files  = list(dict.fromkeys(href_files + attachments))
    has_url    = len(found_urls) > 0
    has_file   = len(all_files) > 0

    # 5. URL 또는 첨부파일 있으면 단독 판정 보류
    is_malicious = None if (has_url or has_file) else int(pred)

    # 6. 탐지 피처 서술
    detected = []
    if has_url:
        detected.append(f"URL {len(found_urls)}개 포함 - URL_Analyzer 검증 필요")
    if has_file:
        detected.append(f"첨부파일 {len(all_files)}개 포함"
                        f"({', '.join(all_files)}) - File_Analyzer 검증 필요")
    if feature['star_count'] >= 2:
        detected.append(f"이름 마스킹 패턴 {feature['star_count']}개 감지")           # 홍*동, 김** 패턴
    if feature['ko_ratio'] < 0.3:
        detected.append(f"한글 비율 낮음 ({feature['ko_ratio']*100:.1f}%)")           # 영문/특수문자 혼재
    if feature['special_ratio2'] > 0.15:
        detected.append(f"특수문자 비율 높음 ({feature['special_ratio2']*100:.1f}%)") # URL·마스킹 등
    if feature['space_ratio'] < 0.08:
        detected.append(f"공백 비율 낮음 ({feature['space_ratio']*100:.1f}%)")        # 연속 문자열 패턴
    if feature['entropy'] > 4.8:
        detected.append(f"문자 복잡도 높음 (entropy={feature['entropy']:.2f})")       # 고엔트로피 = 다양한 문자 혼재
    if not detected:
        detected.append("특이 패턴 없음 - 정상 텍스트로 판단" if prob < 0.5
                        else "텍스트 패턴 기반 악성 의심")

    result_json = {
        # 분석 모듈 식별자
        "module":              "Email_Analyzer",

        # 악성 여부 판정
        # 0: 정상 / 1: 악성 / None: URL 또는 첨부파일 존재로 단독 판정 보류 
        # -> URL,mal모델을 거친후최종 에이전트가 종합 판단
        "is_malicious":        is_malicious,

        # ML 모델의 악성 확률값
        "confidence_score":    round(prob, 4),

        # URL 포함 여부 → True이면 URL_Analyzer 추가 검증 필요
        "requires_url_check":  has_url,

        # 첨부파일 포함 여부 → True이면 File_Analyzer 추가 검증 필요
        "requires_file_check": has_file,

        # 탐지된 URL 목록 (href + 본문 텍스트 URL 병합, 중복 제거)
        # ex) ["https://phishing.com/login", "http://malware.kr/down"]
        # ex) []  ← URL 없음
        "urls":                found_urls,

        # 첨부파일명 목록 (href 파일링크 + analyze_mail() 인자로 받은 파일명 병합)
        # ex) ["invoice.exe", "견적서.docx"]
        # ex) []  ← 첨부파일 없음
        "attachments":         all_files,

        # 탐지된 특이 패턴 설명 목록
        # ex) ["URL 2개 포함 - URL_Analyzer 검증 필요",
        #      "첨부파일 1개 포함(invoice.exe) - File_Analyzer 검증 필요",
        #      "이름 마스킹 패턴 3개 감지",
        #      "특수문자 비율 높음 (18.3%)"]
        # ex) ["특이 패턴 없음 - 정상 텍스트로 판단"]  ← 정상 메일
        # ex) ["텍스트 패턴 기반 악성 의심"]           ← URL/파일 없이 텍스트만으로 악성
        "detected_features":   detected,
    }
    return result_json
