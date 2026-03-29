# tools/analyzer_agent.py
import pickle
import pandas as pd
import numpy as np
import pefile
import math
import os
import re
import json
import sys
from collections import Counter
from scipy.sparse import hstack # 스팸 분석용 추가

# [필수] 프로젝트 루트를 경로에 추가 (url 모듈을 찾기 위함)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../../'))
scripts_path = os.path.join(project_root, 'ml', 'scripts') # ml/scripts 경로
# [수정] project_root -> scripts_path
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)
    
# url.py 모듈 임포트
try:
    from url import analyze_url as url_inference
    print("[*] URL 분석 모듈 임포트 성공")  # 디버깅용
except ImportError:
    # 혹시라도 실패할 경우를 대비한 경로 재확인 로그
    print(f"[!] 경로 확인 실패: {scripts_path}")    # 디버깅용
    raise


class AnalyzerAgent:
    def __init__(self):
        '''3가지 모델을 로드합니다. 변수명을 메서드와 일치시켰습니다.'''
        base_path = os.path.abspath(os.path.join(current_dir, '../../../ml/models'))
        
        # print(f"[*] 모델 탐색 경로: {base_path}") # 디버깅용 출력

        # [수정] URL 모델은 url.py가 내부적으로 로드하므로 여기선 파일과 스팸만 로드
        self.file_model_data = self._load_pickle(os.path.join(base_path, 'malware_model.pkl'))
        # [임시 수정] 에러 발생 : self.spam_bundle = self._load_pickle(os.path.join(base_path, 'spam_V3.pkl'))
        self.spam_bundle = None

    def _load_pickle(self, path):
        if not os.path.exists(path):
            print(f'[!] 경고: 모델 파일을 찾을 수 없습니다 -> {path}')
            return None
        with open(path, 'rb') as f:
            return pickle.load(f)
        
    def _clean_text(self, text):
        text = re.sub(r'<[^>]*>', ' ', text)
        text = re.sub(r'(From|TO|Subject):.*?\n', ' ', text)
        text = re.sub(r'[\t\r\n]', ' ', text)
        return text.strip()
    
    def _get_entropy(self, data):
        '''엔트로피 계산식: $$H(x) = -\sum_{i=1}^{n} P(x_i) \log_2 P(x_i)$$'''
        if not data: return 0
        counter = Counter(data)
        length = len(data)
        entropy = -sum((count / length) * math.log2(count / length) for count in counter.values())
        return entropy

    # --- [스팸 분석 섹션] ---
    def _extract_spam_numerical_features(self, text):
        total_len = len(text)
        features = {
            'text_len': total_len,
            'word_count': len(text.split()),
            'sentence_count': len(re.split(r'[.!?]', text)) - 1,
            'exclaim_count': text.count('!'),
            'star_count': text.count('*'),
            'url_count': len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)),
            'ko_ratio': len(re.findall(r'[ㄱ-ㅎㅏ-ㅣ가-힣]', text)) / total_len if total_len > 0 else 0,
            'digit_ratio': len(re.findall(r'[0-9]', text)) / total_len if total_len > 0 else 0,
            'special_ratio': len(re.findall(r'[^a-zA-Z0-9ㄱ-ㅎㅏ-ㅣ가-힣\s]', text)) / total_len if total_len > 0 else 0,
            'special_ratio2': text.count('%') / total_len if total_len > 0 else 0,
            'entropy': self._get_entropy(text.encode()),
            'has_file': 1 if len(re.findall(r'\.(exe|zip|pdf|docx)', text)) > 0 else 0
        }
        return features
    
    def analyze_email(self, text):
        """보고서의 하이브리드 피처 방식을 적용한 이메일 분석"""
        try:
            if not self.spam_bundle: return {"module": "Email_Analyzer", "error": "Spam model not loaded"}
            
            num_features = self._extract_spam_numerical_features(text)
            num_df = pd.DataFrame([num_features])
            
            # 스케일링 및 TF-IDF 결합
            num_cols = self.spam_bundle['numerical_columns']
            num_scaled = self.spam_bundle['scaler'].transform(num_df[num_cols])
            
            clean_text = self._clean_text(text)
            tfidf_features = self.spam_bundle['tfidf_vectorizer'].transform([clean_text])
            
            final_input = hstack([num_scaled, tfidf_features])
            prob = self.spam_bundle['model'].predict_proba(final_input)[:, 1][0]
            
            return {
                "module": "Email_Analyzer",
                "is_malicious": 1 if prob > 0.5 else 0,
                "confidence_score": float(round(prob, 4)),
                "requires_url_check": True if num_features['url_count'] > 0 else False,
                "requires_file_check": True if num_features['has_file'] == 1 else False,
                "detection_reasons": ["스팸 패턴 감지"] if prob > 0.5 else []
            }
        except Exception as e:
            return {"module": "Email_Analyzer", "error": str(e)}

    # --- [파일 분석 섹션] ---
    def _extract_file_features(self, file_path):
        pe = pefile.PE(file_path)
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
        cols = self.file_model_data['columns']
        features = {col: feature_map.get(col, 0) for col in cols if col != 'Entropy'}
        with open(file_path, 'rb') as f:
            features['Entropy'] = self._get_entropy(f.read())
        pe.close()
        return features

    def analyze_file(self, path):
        try:
            if not self.file_model_data: return {'module': 'File_Analyzer', 'error': 'Model not loaded'}
            features = self._extract_file_features(path)
            df = pd.DataFrame([features])
            cols = self.file_model_data['columns']
            scaler = self.file_model_data['scaler']
            model = self.file_model_data['model']

            for col in ['Size_of_Image', 'Size_of_Code']:
                if col in df.columns: df[col] = np.log1p(df[col])

            features_scaled = scaler.transform(df[cols])
            prob = model.predict_proba(features_scaled)[:, 1][0]
            return {
                'module': 'File_Analyzer',
                'is_malicious': 1 if prob > 0.4 else 0,
                'confidence_score': float(round(prob, 4)),
                'detected_features': [f"Entropy: {round(features['Entropy'], 2)}", f"Sections: {features['num_sections']}"]
            }
        except Exception as e:
            return {'module': 'File_Analyzer', 'error': str(e)}

    def analyze_url(self, url):
        '''전문 모듈인 url.py로 분석 전적 위임'''
        try:
            # JSON 규격 그대로 리턴
            return url_inference(url)
        except Exception as e:
            return {"module": "URL_Analyzer", "error": str(e)}