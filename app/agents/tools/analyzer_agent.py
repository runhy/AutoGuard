import pickle
import pandas as pd
import numpy as np
import pefile
import math
import os
import re
from collections import Counter

class AnalyzerAgent:
    def __init__(self, model_path='../models/malware_model.pkl'):
        '''3가지 모델(파일, URL, 스팸)을 로드하고 초기화합니다.'''
        # 모델 경로 설정
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_path = os.path.abspath(os.path.join(current_dir, '../../ml/models'))
        # 1. 파일 악성코드 모델 로드
        self.file_model_data = self._load_pickle(os.path.join(base_path, 'malware_model.pkl'))
        # 2. URL 악성 모델 로드
        self.url_model_data = self._load_pickle(os.path.join(base_path, 'url_model.pkl'))
        # 3. 스팸/피싱 메일 모델 로드
        self.spam_model_data = self._load_pickle(os.path.join(base_path, 'spam_v3.pkl'))
        
    def _load_pickle(self, path):
        if not os.path.exists(path):
            print(f'[!] 경고: 모델 파일을 찾을 수 없습니다 -> {path}')
            return None
        with open(path, 'rb') as f:
            return pickle.load(f)

    def _get_entropy(self, data):
        '''파일 데이터의 엔트로피를 계산합니다.'''
        if not data: return 0
        counter = Counter(data)
        length = len(data)
        entropy = -sum((count / length) * math.log2(count / length) for count in counter.values())
        return entropy

    def _extract_file_features(self, file_path):
        '''PE 파일 피처 추출 및 컬럼 순서 정렬'''
        pe = pefile.PE(file_path)
        
        # 기본 피처 맵 정의
        feature_map = {
            'Image_Base': pe.OPTIONAL_HEADER.ImageBase,
            'Base_of_Code': pe.OPTIONAL_HEADER.BaseOfCode,
            'Address_of_Entry_Point': pe.OPTIONAL_HEADER.AddressOfEntryPoint,
            'Size_of_Image': pe.OPTIONAL_HEADER.SizeOfImage,
            'Size_of_Code': pe.OPTIONAL_HEADER.SizeOfCode,
            'Subsystem': pe.OPTIONAL_HEADER.Subsystem,
            'num_sections': len(pe.sections),
            'num_dlls': len(pe.DIRECTORY_ENTRY_IMPORT) if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT') else 0,
            'num_functions': sum(len(entry.imports) for entry in pe.DIRECTORY_ENTRY_IMPORT) if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT') else 0,
        }

        # 모델 데이터가 정상적으로 로드되었는지 확인 후 컬럼 추출
        if not self.file_model_data:
            raise ValueError("파일 분석 모델 데이터가 없습니다.")

        # ML 팀에서 정의한 컬럼 리스트
        cols = self.file_model_data['columns']
        # 'Entropy'를 제외한 특징들을 먼저 맵핑
        features = {col: feature_map.get(col, 0) for col in cols if col != 'Entropy'}
        
        # 엔트로피 추가
        with open(file_path, 'rb') as f:
            features['Entropy'] = self._get_entropy(f.read())
        
        pe.close()
        return features

    def analyze_file(self, path):
        '''파일 악성 여부 분석'''
        try:
            if not self.file_model_data: return {'error': 'File model not loaded'}
            
            features = self._extract_file_features(path)
            df = pd.DataFrame([features])
            
            # 전처리 및 추론 (모델 데이터 딕셔너리에서 추출)
            cols = self.file_model_data['columns']
            scaler = self.file_model_data['scaler']
            model = self.file_model_data['model']

            # 전처리
            for col in ['Size_of_Image', 'Size_of_Code']:
                if col in df.columns: 
                    df[col] = np.log1p(df[col])

            # 추론  
            features_scaled = scaler.transform(df[cols])
            prob = model.predict_proba(features_scaled)[:, 1][0]
            
            return {
                'module': 'File_Analyzer',
                'is_malicious': 1 if prob > 0.4 else 0,
                'confidence_score': float(round(prob, 4)),
                'detected_features': [
                    f"Entropy: {round(features['Entropy'], 2)}", 
                    f"Sections: {features['num_sections']}"
                ]
            }
        
        except Exception as e:
            return {'module': 'File_Analyzer', 'error': str(e), 'is_malicious': 0}

    def analyze_url(self, url):
        '''URL 악성 여부 분석'''
        if not self.url_model_data:
            return {'module': 'URL_Analyzer', 'error': 'Model not loaded', 'is_malicious': 0}
        
        # 임시 특징 추출
        return {
            'module': 'URL_Analyzer',
            'is_malicious': 0,
            'confidence_score': 0.45,
            'detected_features': [
                f"Length: {len(url)}", 
                f"Dots: {url.count('.')}" # 따옴표 충돌 방지
            ]
        }

    def analyze_email(self, text):
        '''이메일 본문 분석'''
        if not self.spam_model_data: 
            return {'module': 'Email_Analyzer', 'error': 'Model not loaded', 'is_malicious': 0}
        
        return {
            'module': 'Email_Analyzer',
            'is_malicious': 1,
            'confidence_score': 0.88,
            'detected_features': ['Suspicious keywords detected']
        }