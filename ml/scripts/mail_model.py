import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from scipy.sparse import hstack, csr_matrix
import xgboost as xgb

OUT_DIR   = Path('c:/python_prj/spam/eda_output')
MODEL_DIR = Path('c:/python_prj/spam/model')
MODEL_DIR.mkdir(exist_ok=True)


# 1. 데이터 로드
print("=" * 60)
print("1. 데이터 로드")
print("=" * 60)

# 한국어 데이터
df_ko = pd.read_csv(OUT_DIR / 'ko_merged.csv', encoding='utf-8-sig')
print(f"한국어: {len(df_ko)}행  |  정상(0): {(df_ko['label']==0).sum()}  악성(1): {(df_ko['label']==1).sum()}")

# 영문 데이터 (spam.csv)
df_en = pd.read_csv('c:/python_prj/spam/spam.csv', encoding='latin-1')[['v1', 'v2']]
df_en.columns = ['label_str', 'content']
df_en['label'] = (df_en['label_str'] == 'spam').astype(int)
df_en = df_en[['content', 'label']].dropna(subset=['content'])
print(f"영문  : {len(df_en)}행  |  정상(0): {(df_en['label']==0).sum()}  악성(1): {(df_en['label']==1).sum()}")

# 병합
df = pd.concat([df_ko[['content', 'label']], df_en], ignore_index=True)
print(f"통합  : {len(df)}행  |  정상(0): {(df['label']==0).sum()}  악성(1): {(df['label']==1).sum()}")


# 2. 피처 엔지니어링
print("\n" + "=" * 60)
print("2. 피처 엔지니어링")
print("=" * 60)

def extract_features(texts: pd.Series) -> pd.DataFrame:
    t = texts.fillna('')

    def char_entropy(s):
        if not s:
            return 0.0
        freq = pd.Series(list(s)).value_counts(normalize=True)
        return -(freq * np.log2(freq + 1e-10)).sum()

    feat = pd.DataFrame()
    feat['text_len']       = t.str.len()
    feat['word_count']     = t.str.split().str.len()
    feat['sentence_count'] = t.str.count(r'[.!?\n]') + 1
    feat['exclaim_count']  = t.str.count(r'!')
    feat['digit_ratio']    = t.str.count(r'[0-9]')   / (feat['text_len'] + 1)
    feat['special_ratio2'] = t.str.count(r'[^\w\s]') / (feat['text_len'] + 1)
    feat['ko_ratio']       = t.str.count(r'[가-힣]') / (feat['text_len'] + 1)
    feat['avg_word_len']   = feat['text_len'] / (feat['word_count'] + 1)
    feat['space_ratio']    = t.str.count(r' ')        / (feat['text_len'] + 1)
    feat['star_count']     = t.str.count(r'\*')
    feat['url_count']      = t.str.count(r'https?://|www\.|\.com|\.kr/')
    feat['entropy']        = t.apply(char_entropy)
    return feat

V3_COLS = [
    'text_len', 'word_count', 'sentence_count', 'exclaim_count',
    'digit_ratio', 'special_ratio2',
    'ko_ratio', 'avg_word_len', 'space_ratio',
    'star_count', 'url_count', 'entropy',
]

feat_df = extract_features(df['content'])
print(f"수치 피처 {len(V3_COLS)}개 추출 완료")


# 3. TF-IDF + Scaler fit (전체 데이터 기준)
print("\n" + "=" * 60)
print("3. TF-IDF + Scaler fit")
print("=" * 60)

X_text = df['content'].fillna('')
y      = df['label'].values

tfidf = TfidfVectorizer(
    analyzer='char_wb',
    ngram_range=(2, 4),
    max_features=20000,
    min_df=3,
    sublinear_tf=True
)
X_tfidf = tfidf.fit_transform(X_text)
print(f"char TF-IDF (2~4gram): {X_tfidf.shape[1]}차원")

scaler = StandardScaler()
X_num  = scaler.fit_transform(feat_df[V3_COLS].fillna(0).values)
X_all  = hstack([X_tfidf, csr_matrix(X_num)])
print(f"최종 피처 차원: {X_all.shape[1]}")

# 4. Train/Test Split
X_tr, X_te, y_tr, y_te = train_test_split(
    X_all, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTrain: {X_tr.shape[0]}행  Test: {X_te.shape[0]}행")

scale_pos = (y == 0).sum() / (y == 1).sum()

# 5. 학습 (고정 하이퍼파라미터)
print("\n" + "=" * 60)
print("5. XGBoost 학습")
print("=" * 60)

best_params = {
    'n_estimators':     303,
    'max_depth':        5,
    'learning_rate':    0.14171239821124512,
    'subsample':        0.9274946708125203,
    'colsample_bytree': 0.6679205603162712,
    'min_child_weight': 2,
    'gamma':            0.1671660096632387,
    'reg_alpha':        1.43939974511495e-08,
    'reg_lambda':       5.301682684433521e-07,
    'scale_pos_weight': scale_pos,
    'eval_metric':      'logloss',
    'random_state':     42,
    'n_jobs':           -1,
    'tree_method':      'hist',
}

xgb_model = xgb.XGBClassifier(**best_params)
xgb_model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

y_pred = xgb_model.predict(X_te)
y_prob = xgb_model.predict_proba(X_te)[:, 1]
auc    = roc_auc_score(y_te, y_prob)
rpt    = classification_report(y_te, y_pred, output_dict=True)

print(f"AUC    : {auc:.4f}")
print(f"정확도 : {rpt['accuracy']:.4f}")
print()
print("%-10s %8s %8s %8s" % ("", "Precision", "Recall", "F1"))
print("-" * 38)
print("%-10s %8.4f %8.4f %8.4f" % ("정상(0)", rpt['0']['precision'], rpt['0']['recall'], rpt['0']['f1-score']))
print("%-10s %8.4f %8.4f %8.4f" % ("악성(1)", rpt['1']['precision'], rpt['1']['recall'], rpt['1']['f1-score']))
print("-" * 38)
print("%-10s %8.4f %8.4f %8.4f" % ("macro avg", rpt['macro avg']['precision'], rpt['macro avg']['recall'], rpt['macro avg']['f1-score']))

# 6. 모델 번들 저장 (.pkl)
print("\n" + "=" * 60)
print("6. 모델 저장")
print("=" * 60)

model_bundle = {
    'tfidf':    tfidf,
    'scaler':   scaler,
    'model':    xgb_model,
    'v3_cols':  V3_COLS,
    'auc':      auc,
    'accuracy': rpt['accuracy'],
}

save_path = MODEL_DIR / 'spam_V3.pkl'
joblib.dump(model_bundle, save_path)
print(f"저장 완료: {save_path}")
print(f"파일 크기: {save_path.stat().st_size / 1024 / 1024:.1f} MB")
print("\n[완료] 한국어 + 영문 통합 모델 저장 완료")
