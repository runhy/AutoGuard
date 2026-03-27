import pandas as pd
import numpy as np
import pefile
import math

with open("./malware_model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
scaler = data["scaler"]
columns = data["columns"]

def get_entropy(file_path):
    with open(file_path,"rb") as f:
        data = f.read()
    if len(data) == 0:
        return 0
    entropy = 0
    for x in range(256):
        p = data.count(x)/len(data)
        if p > 0:
            entropy -= p*math.log2(p)
    return entropy

def extract_file_feature(fileName, columns):
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
    }

    file_feature = {}
    for col in columns:
        try:
            if col in feature_map:
                file_feature[col] = feature_map[col]()
            else:
                file_feature[col] = 0
        except:
            file_feature[col] = 0

    pe.close()

    file_feature["Entropy"] = get_entropy(fileName)

    return file_feature

def predict_file(feature, model, scaler, columns):

    feature = pd.DataFrame([feature])
    for col in ['Size_of_Image', 'Size_of_Code']:
        if col in feature.columns:
            feature[col] = np.log1p(feature[col])

    feature = feature[columns]

    feature_scaled = scaler.transform(feature)

    y_prob = model.predict_proba(feature_scaled)[:, 1]

    threshold = 0.5
    y_pred = (y_prob > threshold).astype(int)

    return y_pred[0], y_prob[0]

# import sys
# sys.path.append("/home/user/malware_module")
# from mal import analyze_file -> 이거 하셔가지고
# 이 함수 사용하면 됩니당 -> analyze_file("저장된 파일경로")
def analyze_file(path):
    file_feature = extract_file_feature(path, columns)
    pred, prob = predict_file(file_feature, model, scaler, columns)
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

#analyze_file("/content/drive/MyDrive/sample_malware3.exe")
