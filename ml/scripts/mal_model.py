import pandas as pd
import numpy as np
import pefile
import math
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.calibration import CalibratedClassifierCV

THRESHOLD = 0.5  # 악성 판정 기준 — 보안 도구 특성상 0.4~0.5 권장 (높일수록 악성 누락 위험)

# 파일 불러오기 (데이터셋)
def read_file():
    fileName = "./drive/MyDrive/PE_Dataset_Labeled.csv"
    df = pd.read_csv(fileName)
    return df

# hex 값을 숫자로 변환
def convert_hex(x):
    try:
        # PE 데이터는 hex 값을 주로 사용하기 때문에 변환이 필요함
        if isinstance(x, str) and x.startswith("0x"):
            return int(x, 16)
        return float(x)
    except:
        return np.nan


def data_Pro_Processing(df):

    # 인덱스열 제거
    df = df.drop(df.columns[0], axis=1)
    # 라벨을 숫자로 변환 | 정상 -> 0, 악성 -> 1
    y = df['Label'].map({
        'Benign': 0,
        'Malicious': 1
    })
    # 불필요한 값은 제거 -> 악성 탐지에 꼭 필요하지 않은 값들
    drop_cols = [
        'File_Name',
        'e_cblp','e_cp','e_cparhdr','e_crlc','e_cs',
        'e_lfarlc','e_magic',
        'e_maxalloc','e_minalloc','e_oemid','e_oeminfo',
        'e_ovno','e_res','e_res2','e_sp','e_ss', 'Label'
    ]
    df = df.drop(columns=drop_cols)

    df.replace(["NULL","null","NaN","nan","None",""], np.nan, inplace=True)

    # 필드들의 hex 값을 숫자로 변환
    for col in df.columns:
        df[col] = df[col].apply(convert_hex)

    # 베이스 데이터의 결측 여부 피처칸 생성 -> 베이스 데이터가 결측인 경우
    if 'Base_of_Data' in df.columns:
        df['Base_of_Data_missing'] = df['Base_of_Data'].isnull().astype(int)

    # 필요한 파생 피처 생성 (fillna 전에 계산 — 0으로 채워지기 전에 해야 개수 오계산 방지)
    # 섹션 개수 -> 많으면 패킹/변조 가능
    df['num_sections'] = df['Sections'].apply(
        lambda x: len(str(x).split(',')) if pd.notna(x) and str(x) not in ['0', 'nan', ''] else 0
    )
    # DDL 개수 -> 외부 API 다수 사용
    df['num_dlls'] = df['DLLs'].apply(
        lambda x: len(str(x).split(',')) if pd.notna(x) and str(x) not in ['0', 'nan', ''] else 0
    )
    # 함수 개수 -> 코드 복잡도
    df['num_functions'] = df['Functions'].apply(
        lambda x: len(str(x).split(',')) if pd.notna(x) and str(x) not in ['0', 'nan', ''] else 0
    )

    # 원본 피처 제거 (원본 문자열은 모델에서 사용할 수 없기 때문에 갯수로 구분하여 대체)
    df = df.drop(columns=['Sections','DLLs','Functions'], errors='ignore')

    # 결측치 0
    df = df.fillna(0)

    # 상위 0.1%의 이상치 제거 (upper는 예측 시에도 동일하게 적용하기 위해 반환)
    upper = df.quantile(0.999)
    df = df.clip(lower=0, upper=upper, axis=1)

    # 로그로 변환하여 큰 값 편향 감소
    for col in ['Size_of_Image','Size_of_Code']:
        if col in df.columns:
            df[col] = np.log1p(df[col])

    # Entropy -> 난독화나 패킹 여부를 위한 피처
    # 현재 데이터셋에 있는 파일의 데이터들은 실제 파일이 아니기 때문에 Entropy를 당장 추출할 순 없음
    # 그래서 0으로 채움
    #df["Entropy"] = 0

    columns = df.columns
    return df, y, columns, upper  # upper 반환 — 예측 시 동일한 클리핑 적용에 필요

def calculate_ece(y_true, y_prob, n_bins=15):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1

    ece = 0.0

    for i in range(n_bins):
        mask = bin_ids == i
        if np.sum(mask) > 0:
            acc = np.mean(y_true[mask])
            conf = np.mean(y_prob[mask])
            ece += np.abs(acc - conf) * np.sum(mask) / len(y_true)

    return ece

df = read_file()
X, y, columns, upper = data_Pro_Processing(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 랜덤포레스트 모델 사용
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)
#model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
# 모델 학습
model.fit(X_train, y_train)

y_prob = model.predict_proba(X_test)[:, 1] # 확률
y_pred = (y_prob > THRESHOLD).astype(int)   # 악성 여부

# 모델 평가
print(f"[threshold = {THRESHOLD}]")
print("Accuracy:",  accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:",    recall_score(y_test, y_pred))
print("F1:",        f1_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

ece = calculate_ece(y_test.values, y_prob)

print(f"ECE (15 bins): {ece:.4f}")

# 객체 생성
with open("malware_model.pkl","wb") as f:
    pickle.dump({
        "model":     model,
        "columns":   columns,
        "upper":     upper,      # 클리핑 기준값 저장 — 예측 시 동일한 전처리 적용에 필요
        "threshold": THRESHOLD   # threshold 저장 — 예측 모듈과 기준 통일
    }, f)
