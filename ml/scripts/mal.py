# ml/scripts/mal.py
# PE 파일 헤더를 분석하여 악성코드 여부를 판단
# [수정 사항] pickle 로드 시 발생하는 AttributeError 방지 및 경로 유연성 확보
# [수정 사항] 테스트 환경 오류 발생 KeyError 발생

import pandas as pd
import numpy as np
import pefile
import math
import pickle
import sys
import __main__ # [추가] Pickle이 로드 시 함수 설계도를 찾을 수 있게 Main 영역 접근
from pathlib import Path # [추가] OS에 상관없이 파일 경로를 안전하게 계산


# 엔트로피 계산
def get_entropy(file_path):
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        if not data:
            return 0
        entropy = 0
        for x in range(256):
            p = data.count(x) / len(data)
            if p > 0:
                entropy -= p * math.log2(p)
        return round(entropy, 4)
    except:
        return 0

# [추가] Pickle은 로드 시 함수가 정의된 위치를 찾습니다. 
# 현재 실행 주체인 __main__에 설계도를 미리 주입하여 AttributeError를 방지합니다.
__main__.get_entropy = get_entropy

# 피처 추출
def extract_file_feature(fileName, columns):
    try:
        pe = pefile.PE(fileName)

        oh  = pe.OPTIONAL_HEADER
        dos = pe.DOS_HEADER

        # 머신 러닝에 사용되는 피처들 목록
        feature_map = {
            "Address_of_Entry_Point":     lambda: oh.AddressOfEntryPoint,
            "Base_of_Code":               lambda: oh.BaseOfCode,
            "Base_of_Data":               lambda: getattr(oh, 'BaseOfData', 0),
            "Checksum":                   lambda: oh.CheckSum,
            "DLL_Characteristics":        lambda: oh.DllCharacteristics,
            "File_Alignment":             lambda: oh.FileAlignment,
            "Image_Base":                 lambda: oh.ImageBase,
            "Loader_Flags":               lambda: oh.LoaderFlags,
            "Magic":                      lambda: oh.Magic,
            "Major_Image_Version":        lambda: oh.MajorImageVersion,
            "Major_Linker_Version":       lambda: oh.MajorLinkerVersion,
            "Major_OS_Version":           lambda: oh.MajorOperatingSystemVersion,
            "Major_Subsystem_Version":    lambda: oh.MajorSubsystemVersion,
            "Minor_Image_Version":        lambda: oh.MinorImageVersion,
            "Minor_Linker_Version":       lambda: oh.MinorLinkerVersion,
            "Minor_OS_Version":           lambda: oh.MinorOperatingSystemVersion,
            "Minor_Subsystem_Version":    lambda: oh.MinorSubsystemVersion,
            "Number_of_Rva_and_Sizes":    lambda: oh.NumberOfRvaAndSizes,
            "Section_Alignment":          lambda: oh.SectionAlignment,
            "Size_of_Code":               lambda: oh.SizeOfCode,
            "Size_of_Headers":            lambda: oh.SizeOfHeaders,
            "Size_of_Heap_Commit":        lambda: oh.SizeOfHeapCommit,
            "Size_of_Heap_Reserve":       lambda: oh.SizeOfHeapReserve,
            "Size_of_Image":              lambda: oh.SizeOfImage,
            "Size_of_Initialized_Data":   lambda: oh.SizeOfInitializedData,
            "Size_of_Stack_Commit":       lambda: oh.SizeOfStackCommit,
            "Size_of_Stack_Reserve":      lambda: oh.SizeOfStackReserve,
            "Size_of_Uninitialized_Data": lambda: oh.SizeOfUninitializedData,
            "Subsystem":                  lambda: oh.Subsystem,
            "Win32_Version_Value":        lambda: oh.Win32VersionValue,
            # 도스 Header
            "e_csum":                     lambda: dos.e_csum,
            "e_ip":                       lambda: dos.e_ip,
            "e_lfanew":                   lambda: dos.e_lfanew,
            "num_sections":  lambda: len(pe.sections),
            "num_dlls":      lambda: len(pe.DIRECTORY_ENTRY_IMPORT) if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT') else 0,
            "num_functions": lambda: sum(len(e.imports) for e in pe.DIRECTORY_ENTRY_IMPORT) if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT') else 0,
            # [추가] 보고서에 명시된 결측치 피처
            "Base_of_Data_missing":       lambda: 1 if getattr(oh, 'BaseOfData', None) is None else 0
        }

        file_feature = {}
        for col in columns:
            try:
                if col in feature_map:
                    file_feature[col] = feature_map[col]()
                else:
                    file_feature[col] = 0
            except Exception as e:
                file_feature[col] = 0
            # print(f"[warn] {col}: {e}")  # 디버깅 시 주석 해제
        pe.close()
        return file_feature
    except: return {col: 0 for col in columns}


# [추가] 모델 파일이 사용하는 함수 설계도를 전역으로 노출
__main__.extract_file_feature = extract_file_feature


# [경로 수정] 상대 경로("./")는 서버 실행 위치에 따라 깨질 수 있음
# -> 현재 파일 위치를 기준으로 절대 경로를 계산하여 어디서든 모델을 읽을 수 있게 조정
_MODEL_PATH = Path(__file__).parent.parent / "models" / "malware_model.pkl"
# [수정] 절대 경로 설정
with open(_MODEL_PATH, "rb") as f:
    data = pickle.load(f)

model     = data["model"]
columns   = data["columns"]
# [수정] KeyError 방지 : 'upper'가 없을 시 아주 큰 값으로 대체, data.get 사용
upper     = data.get("upper", 999999999)      # 클리핑 기준값 — 학습 때와 동일한 전처리 적용에 필요
THRESHOLD = data.get("threshold", 0.5)  # 학습 때와 동일한 threshold 사용

# 악성 여부 판단
def predict_file(feature, model, upper, columns, threshold):
    feature = pd.DataFrame([feature])
    # 데이터 전처리
    for col in ['Size_of_Image', 'Size_of_Code']:
        if col in feature.columns:
            feature[col] = np.log1p(feature[col])

    # 학습 데이터와 컬럼 순서 맞추기
    feature = feature[columns]

    # 학습 때와 동일한 클리핑(이상치 제한) 적용
    feature = feature.clip(lower=0, upper=upper, axis=1)

    y_prob = model.predict_proba(feature)[:, 1]
    y_pred = (y_prob > threshold).astype(int)

    return y_pred[0], y_prob[0]

# import sys
# sys.path.append("/home/user/malware_module")
# from mal import analyze_file -> 이거 하셔가지고
# 이 함수 사용하면 됩니당 -> analyze_file("저장된 파일경로")

# [통합 인터페이스] 서버 + 에이전트가 사용하는 최종 분석 함수
def analyze_file(path):
    try:
        file_feature = extract_file_feature(path, columns)
        file_feature["Entropy"] = get_entropy(path)
        pred, prob = predict_file(file_feature, model, upper, columns, THRESHOLD)
        # [결과 규격화] 모든 모듈의 리턴 형식을 통일하여 에이전트가 읽기 편하게 함
        result_json = {
            "module": "File_Analyzer",
            "is_malicious": int(pred),
            "confidence_score": float(round(prob,4)),
            "detected_features": [
                f"Entropy={round(file_feature['Entropy'],4)}", # 파일 난수성 -> 높을수록 악성일 확률 높음(패커나 암호화된 파일)
                f"Sections={file_feature.get('num_sections',0)}", # 섹션 개수 -> 패킹이나 변조 여부 추정(섹션 수가 비정상정으로 많을 시 패킹/숨김/변조 가능)
                f"DLLs={file_feature.get('num_dlls',0)}", # 외부 API 호출 -> 많으면 악성 가능성이 높지만 확정은 아님
                f"Size_of_Image={file_feature.get('Size_of_Image',0)}", # PE 크기
                f"Subsystem={file_feature.get('Subsystem',0)}" # 실행환경 정보
            ]
        }
        return result_json
    except Exception as e:
        return {"module": "File_Analyzer", "error": f"분석 중 오류: {str(e)}"}
#analyze_file("/content/drive/MyDrive/sample_mal/Win32.Wannacry.exe")