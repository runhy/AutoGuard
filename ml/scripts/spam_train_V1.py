import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

OUT_DIR   = Path('출력물을 저장할 폴더 경로')
MODEL_DIR = Path('모델이 위치한폴더 경로')
MODEL_DIR.mkdir(exist_ok=True)

N_TRIALS = 50
CV_FOLDS = 3


# 1. 데이터 로드
print("=" * 60)
print("1. 데이터 로드")
print("=" * 60)

df = pd.read_csv(OUT_DIR / 'ko_merged.csv', encoding='utf-8-sig')
print(f"총 {len(df)}행  |  정상(0): {(df['label']==0).sum()}  악성(1): {(df['label']==1).sum()}")

# 2. 피처 엔지니어링 (수치 12개)
print("\n" + "=" * 60)
print("2. 피처 엔지니어링 (수치 피처 12개)")
print("=" * 60)

def extract_features(texts: pd.Series) -> pd.DataFrame:
    t = texts.fillna('')

    def char_entropy(s):
        if not s:
            return 0.0
        freq = pd.Series(list(s)).value_counts(normalize=True)
        return -(freq * np.log2(freq + 1e-10)).sum()

    feat = pd.DataFrame()
    # 1. 텍스트 전체 길이
    feat['text_len']       = t.str.len()
    # 2. 단어 수: 공백 기준 단어 개수 산출
    feat['word_count']     = t.str.split().str.len()
    #3. 문장 수: 줄바꿈이나 마침표 기반의 문장 단위 분리
    feat['sentence_count'] = t.str.count(r'[.!?\n]') + 1
    #4. 느낌표 개수
    feat['exclaim_count']  = t.str.count(r'!')
    #5. 숫자 비율: 전화 유도나 인증번호 사칭 메시지 타겟
    feat['digit_ratio']    = t.str.count(r'[0-9]')   / (feat['text_len'] + 1)
    # 6. 특수문자 비율
    feat['special_ratio2'] = t.str.count(r'[^\w\s]') / (feat['text_len'] + 1)
    # 7. 한글 비율
    feat['ko_ratio']       = t.str.count(r'[가-힣]') / (feat['text_len'] + 1)
    # 8. 평균 단어 길이: URL 포함 시 급격히 상승
    feat['avg_word_len']   = feat['text_len'] / (feat['word_count'] + 1)
    # 9. 공백 비율: 띄어쓰기 없이 단어를 나열하는 우회 시도 탐지용
    feat['space_ratio']    = t.str.count(r' ')        / (feat['text_len'] + 1)
    # 10. (*) 수: 가짜 개인정보 마스킹 패턴
    feat['star_count']     = t.str.count(r'\*')
    # 11. URL 형식 탐지 지표: http, www, .com 등
    feat['url_count']      = t.str.count(r'https?://|www\.|\.com|\.kr/')
    # 12. 텍스트 엔트로피
    feat['entropy']        = t.apply(char_entropy)
    return feat

V3_COLS = [
    'text_len', 'word_count', 'sentence_count', 'exclaim_count',
    'digit_ratio', 'special_ratio2',
    'ko_ratio', 'avg_word_len', 'space_ratio',
    'star_count', 'url_count', 'entropy',
]

feat_df = extract_features(df['content'])
print(f"수치 피처 {len(V3_COLS)}개 추출 완료: {V3_COLS}")


# 3. Scaler fit
print("\n" + "=" * 60)
print("3. Scaler fit")
print("=" * 60)

y      = df['label'].values
scaler = StandardScaler()
X_all  = scaler.fit_transform(feat_df[V3_COLS].fillna(0).values)
print(f"입력 shape: {X_all.shape}")

# 4. Train/Test Split
X_tr, X_te, y_tr, y_te = train_test_split(
    X_all, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {X_tr.shape[0]}행  Test: {X_te.shape[0]}행")

scale_pos = (y == 0).sum() / (y == 1).sum()

# 5. Optuna 하이퍼파라미터 튜닝
print("\n" + "=" * 60)
print(f"4. Optuna 하이퍼파라미터 튜닝 ({N_TRIALS} trials, {CV_FOLDS}-Fold CV)")
print("=" * 60)

def objective(trial):
    params = {
        'n_estimators':     trial.suggest_int('n_estimators', 100, 500),
        'max_depth':        trial.suggest_int('max_depth', 3, 8),
        'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma':            trial.suggest_float('gamma', 0.0, 1.0),
        'reg_alpha':        trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda':       trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        'scale_pos_weight': scale_pos,
        'eval_metric':      'logloss',
        'random_state':     42,
        'n_jobs':           -1,
        'tree_method':      'hist',
    }
    model = xgb.XGBClassifier(**params)
    cv    = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    score = cross_val_score(model, X_tr, y_tr, cv=cv,
                            scoring='roc_auc', n_jobs=-1)
    return score.mean()

study = optuna.create_study(direction='maximize',
                             sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

best_params = study.best_params.copy()
best_params.update({
    'scale_pos_weight': scale_pos,
    'eval_metric':      'logloss',
    'random_state':     42,
    'n_jobs':           -1,
    'tree_method':      'hist',
})

print(f"\n최적 AUC (CV): {study.best_value:.4f}")
print("최적 파라미터:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")


# 6. 최적 파라미터로 최종 학습
print("\n" + "=" * 60)
print("5. 최적 파라미터로 최종 학습")
print("=" * 60)

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

# 7. 모델 번들 저장 (.pkl)
print("\n" + "=" * 60)
print("6. 모델 저장")
print("=" * 60)

model_bundle = {
    'scaler':      scaler,
    'model':       xgb_model,
    'v3_cols':     V3_COLS,
    'best_params': study.best_params,
    'auc':         auc,
    'accuracy':    rpt['accuracy'],
}

save_path = MODEL_DIR / 'email_spam_model_num.pkl'
joblib.dump(model_bundle, save_path)
print(f"저장 완료: {save_path}")
print(f"파일 크기: {save_path.stat().st_size / 1024 / 1024:.2f} MB")
print("\n[완료] 수치 피처 전용 모델 저장 완료")
