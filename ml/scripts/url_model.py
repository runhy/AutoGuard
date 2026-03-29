# app/ml/scripts/url_model.py
import re
import math
import pickle
import pandas as pd
from urllib.parse import urlparse, parse_qs
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score


# 경로 설정
DATA_PATH  = "urls_1.csv"
MODEL_PATH = "url_model.pkl"


# 데이터 로드
df = pd.read_csv(DATA_PATH)

# url, label 제외한 전체 특징 사용 (나중에 상관계수 보고 제외할 것)
NUMERIC_FEATURES = [col for col in df.columns if col not in ["url", "label", "tld"]]
CATEGORICAL_FEATURES = ["tld"]

X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"전체: {len(df):,}개")
print(f"학습: {len(X_train):,}개  /  테스트: {len(X_test):,}개")
print(f"악성 비율: {y.mean()*100:.1f}%")
print(f"수치형 특징: {len(NUMERIC_FEATURES)}개 / 범주형 특징: {len(CATEGORICAL_FEATURES)}개")


# 전처리 + 학습

# 수치형: 결측치만 중앙값으로 채우기
numeric_transformer = SimpleImputer(strategy="median")

# 범주형(tld): 결측치 채우기 → OneHotEncoding
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
    ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

# 수치형/범주형 각각 전처리 후 합치기
preprocessor = ColumnTransformer([
    ("num", numeric_transformer,     NUMERIC_FEATURES),
    ("cat", categorical_transformer, CATEGORICAL_FEATURES),
])

# 전처리 + 모델을 하나로 묶기
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced",
    )),
])


# 모델 학습
pipeline.fit(X_train, y_train)
print("학습 완료")


# 모델 평가
y_pred  = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred, target_names=["정상(0)", "악성(1)"]))
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")

# 상위 중요 특징 출력
# clf = pipeline.named_steps["classifier"]
# ohe_names = (
#     pipeline.named_steps["preprocessor"]
#     .named_transformers_["cat"]
#     .named_steps["onehot"]
#     .get_feature_names_out(CATEGORICAL_FEATURES)
#     .tolist()
# )
# all_feature_names = NUMERIC_FEATURES + ohe_names
# importance_pairs = sorted(
#     zip(all_feature_names, clf.feature_importances_),
#     key=lambda x: x[1], reverse=True
# )
# top_features = [name for name, _ in importance_pairs[:15]]

# print("\n상위 10개 중요 특징:")
# for rank, (name, score) in enumerate(importance_pairs[:10], 1):
#     bar = "█" * int(score * 400)
#     print(f"  {rank:2}. {name:<35} {score:.4f} {bar}")


# URL 문자열 1개 → 특징 추출 함수 (FastAPI 재사용용)

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

    def _entropy(s):
        if not s:
            return 0.0
        freq = [s.count(c) / len(s) for c in set(s)]
        return round(-sum(p * math.log2(p) for p in freq), 2)

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

# 의심 항목 반환
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

    return detected if detected else top_features[:5]


# pkl 저장

save_obj = {
    "pipeline":              pipeline,
    "numeric_features":      NUMERIC_FEATURES,
    "categorical_features":  CATEGORICAL_FEATURES,
  #  "top_features":          top_features,
    "extract_features":      extract_features_from_url,
    "get_detected_features": get_detected_features,
}

with open(MODEL_PATH, "wb") as f:
    pickle.dump(save_obj, f)

print(f"\npkl 저장 완료: {MODEL_PATH}")


# # pkl 동작 확인

# print("==테스트==")

# loaded = pickle.load(open(MODEL_PATH, "rb"))

# test_cases = [
#     ("https://www.google.com/search?q=hello",           "정상 예상"),
#     ("http://eu.battle.net.blizzardenteaitaccout.com/", "악성 예상"),
#     ("http://paypal-login-verify.secure-account.tk/",   "악성 예상"),
#     ("https://www.github.com/openai/whisper",           "정상 예상"),
#     ("http://192.168.1.1/admin/login.php",              "악성 예상"),
# ]

# for url, expected in test_cases:
#     feat     = loaded["extract_features"](url)
#     X_in     = pd.DataFrame([feat])[
#                    loaded["numeric_features"] + loaded["categorical_features"]
#                ]
#     pred     = loaded["pipeline"].predict(X_in)[0]
#     proba    = loaded["pipeline"].predict_proba(X_in)[0][1]
#     detected = loaded["get_detected_features"](feat, loaded["top_features"])

#     mark = "O" if (
#         (expected == "악성 예상" and pred == 1) or
#         (expected == "정상 예상" and pred == 0)
#     ) else "X"

#     print(f"\n[{mark}] {expected}")
#     print(f"  URL      : {url[:60]}")
#     print(f"  결과     : is_malicious={int(pred)}, confidence={round(float(proba), 4)}")
#     print(f"  detected : {detected}")