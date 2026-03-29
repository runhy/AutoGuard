"""
AutoGuard AI - 보안 분석 시스템 (최종본)
URL / Email / File 입력창을 하나로 통합 → 입력값 자동 감지
사이드바를 마지막에 렌더링하여 실시간 동기화 적용
"""

import streamlit as st
import streamlit.components.v1 as components
import time
import os
import io
import re
import json
import datetime
from collections import defaultdict

import PyPDF2
import numpy as np

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Table, TableStyle, HRFlowable
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False

try:
    from pdf2image import convert_from_bytes
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


VS_JSON_PATH = "vector_store_cache.json"
HISTORY_PATH = "analysis_history.json"


def register_korean_font() -> str:
    base_dir     = os.path.dirname(os.path.abspath(__file__))
    project_font = os.path.join(base_dir, "fonts", "NanumGothic.ttf")

    font_candidates = [
        project_font,
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/Library/Fonts/AppleGothic.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        os.path.join("C:\\", "Windows", "Fonts", "malgun.ttf"),
        os.path.join("C:\\", "Windows", "Fonts", "gulim.ttc"),
    ]

    for path in font_candidates:
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont("KoreanFont", path))
                return "KoreanFont"
            except Exception:
                continue

    try:
        pdfmetrics.registerFont(UnicodeCIDFont("HYSMyeongJo-Medium"))
        return "HYSMyeongJo-Medium"
    except Exception:
        pass

    return "Helvetica"


KOREAN_FONT = register_korean_font()


class VectorStore:
    def __init__(self):
        self.chunks: list       = []
        self.vectors            = None
        self.vocab: dict        = {}
        self.idf                = None
        self.is_built: bool     = False
        self.source_name: str   = ""
        self.use_semantic: bool = False
        self._semantic_model    = None

    @staticmethod
    def _tokenize(text: str) -> list:
        tokens = re.findall(r"[가-힣]{2,}|[a-zA-Z]{3,}", text.lower())
        stopwords = {
            "the","and","for","that","this","with","from","are","was","has",
            "have","not","but","its","you","they","이","그","저","것","수",
            "있","없","등","및","또","하는","하여","하고","때문","경우",
            "대한","위한","통해","기반","관련","사용","이후","이전"
        }
        return [t for t in tokens if t not in stopwords]

    @staticmethod
    def _compute_tf(tokens: list) -> dict:
        tf = defaultdict(float)
        if not tokens:
            return dict(tf)
        for t in tokens:
            tf[t] += 1.0
        total = len(tokens)
        return {k: v / total for k, v in tf.items()}

    def _build_tfidf(self, chunks: list) -> None:
        N         = len(chunks)
        tokenized = [self._tokenize(c) for c in chunks]

        all_words = set()
        for tokens in tokenized:
            all_words.update(tokens)
        self.vocab = {w: i for i, w in enumerate(sorted(all_words))}
        V          = len(self.vocab)
        if V == 0:
            return

        df = np.zeros(V, dtype=np.float32)
        for tokens in tokenized:
            for w in set(tokens):
                if w in self.vocab:
                    df[self.vocab[w]] += 1.0
        self.idf = np.log((N + 1) / (df + 1)) + 1.0

        matrix = np.zeros((N, V), dtype=np.float32)
        for idx, tokens in enumerate(tokenized):
            tf = self._compute_tf(tokens)
            for w, tf_val in tf.items():
                if w in self.vocab:
                    j             = self.vocab[w]
                    matrix[idx,j] = tf_val * self.idf[j]

        norms           = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms==0] = 1.0
        self.vectors    = matrix / norms
        self.is_built   = True

    def _build_semantic(self, chunks: list) -> None:
        model                = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        self._semantic_model = model
        embeddings           = model.encode(chunks, show_progress_bar=False)
        norms                = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms==0]      = 1.0
        self.vectors         = embeddings / norms
        self.is_built        = True
        self.use_semantic    = True

    def build(self, chunks: list, source_name: str = "") -> None:
        self.chunks      = chunks
        self.source_name = source_name
        if SEMANTIC_AVAILABLE:
            self._build_semantic(chunks)
        else:
            self._build_tfidf(chunks)

    def _query_to_vector_tfidf(self, query: str) -> np.ndarray:
        tokens = self._tokenize(query)
        V      = len(self.vocab)
        vec    = np.zeros(V, dtype=np.float32)
        tf     = self._compute_tf(tokens)
        for w, tf_val in tf.items():
            if w in self.vocab:
                j      = self.vocab[w]
                vec[j] = tf_val * self.idf[j]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def search(self, query: str, top_k: int = 3) -> list:
        if not self.is_built or self.vectors is None:
            return []

        if self.use_semantic and self._semantic_model:
            q_vec = self._semantic_model.encode([query])
            norm  = np.linalg.norm(q_vec)
            if norm > 0:
                q_vec /= norm
            scores = cosine_similarity(q_vec, self.vectors)[0]
        else:
            q_vec  = self._query_to_vector_tfidf(query)
            scores = self.vectors @ q_vec

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for rank, idx in enumerate(top_indices, 1):
            score = float(scores[idx])
            if score > 0.0:
                results.append({
                    "chunk": self.chunks[idx],
                    "score": round(score * 100, 1),
                    "rank":  rank,
                })
        return results


def save_vector_store_json(vs: VectorStore) -> None:
    try:
        data = {
            "chunks":       vs.chunks,
            "source_name":  vs.source_name,
            "vocab":        vs.vocab,
            "idf":          vs.idf.tolist()     if vs.idf     is not None else None,
            "vectors":      vs.vectors.tolist() if vs.vectors is not None else None,
            "use_semantic": vs.use_semantic,
        }
        with open(VS_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception as e:
        st.warning(f"벡터 DB 저장 실패: {e}")


def load_vector_store_json() -> "VectorStore | None":
    if not os.path.exists(VS_JSON_PATH):
        return None
    try:
        with open(VS_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        vs              = VectorStore()
        vs.chunks       = data["chunks"]
        vs.source_name  = data.get("source_name", "")
        vs.vocab        = data.get("vocab", {})
        vs.use_semantic = data.get("use_semantic", False)
        vs.idf          = np.array(data["idf"],     dtype=np.float32) if data.get("idf")     else None
        vs.vectors      = np.array(data["vectors"], dtype=np.float32) if data.get("vectors") else None
        vs.is_built     = vs.vectors is not None and len(vs.chunks) > 0

        if vs.use_semantic:
            if SEMANTIC_AVAILABLE:
                vs._semantic_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            else:
                vs.use_semantic = False

        return vs if vs.is_built else None
    except Exception:
        return None


def load_history() -> list:
    if not os.path.exists(HISTORY_PATH):
        return []
    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_history(entry: dict) -> None:
    history = load_history()
    history.insert(0, entry)
    history = history[:30]
    try:
        with open(HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False)
    except Exception:
        pass


def delete_history_item(index: int) -> None:
    history = load_history()
    if 0 <= index < len(history):
        history.pop(index)
    try:
        with open(HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False)
    except Exception:
        pass
    st.session_state.history = history


def load_pdf_with_ocr(file) -> tuple[str, bool]:
    raw_bytes = file.read()
    file.seek(0)

    reader = PyPDF2.PdfReader(io.BytesIO(raw_bytes))
    text   = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"

    if len(text.strip()) > 100:
        return text, False

    if OCR_AVAILABLE:
        try:
            images   = convert_from_bytes(raw_bytes)
            ocr_text = ""
            for img in images:
                ocr_text += pytesseract.image_to_string(img, lang="kor+eng") + "\n"
            if len(ocr_text.strip()) > 50:
                return ocr_text, True
        except Exception:
            pass

    return text, False


def split_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list:
    sentences = re.split(r"(?<=[.!?。\n])\s+", text.strip())
    chunks    = []
    current   = ""
    for sent in sentences:
        if len(current) + len(sent) <= chunk_size:
            current += " " + sent
        else:
            if current.strip():
                chunks.append(current.strip())
            overlap_text = current[-overlap:] if len(current) > overlap else current
            current      = overlap_text + " " + sent
    if current.strip():
        chunks.append(current.strip())
    return [c for c in chunks if len(c) > 20]


def build_vector_store(file, source_name: str = "") -> tuple["VectorStore", bool]:
    text, used_ocr = load_pdf_with_ocr(file)
    chunks         = split_text(text)
    vs             = VectorStore()
    vs.build(chunks, source_name=source_name)
    return vs, used_ocr


def agent_classify(text: str, uploaded_file=None) -> tuple[str, str, bool]:
    text_stripped = text.strip()
    archive_exts  = {".zip", ".rar", ".7z", ".tar", ".gz", ".bz2", ".xz", ".cab", ".iso"}

    email_body_kw = ["수신", "발신", "제목", "받는사람", "보낸사람", "from:", "to:", "subject:"]
    is_email_body = any(kw in text_stripped.lower() for kw in email_body_kw)
    is_email_addr = bool(re.search(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}', text_stripped))
    is_email      = is_email_body or is_email_addr
    has_url       = bool(re.search(r'https?://', text_stripped, re.IGNORECASE))
    has_file      = uploaded_file is not None

    if is_email and has_url and has_file:
        return "Email+URL+File", "📧🔗📂 [1차 에이전트] 이메일 본문 + URL + 첨부파일 복합 감지", False

    if is_email and has_file:
        return "Email+File", "📧📂 [1차 에이전트] 이메일 본문 + 첨부파일 복합 감지", False

    if is_email and has_url:
        return "Email+URL", "📧🔗 [1차 에이전트] 이메일 본문 + URL 복합 감지", False

    if has_file:
        ext = os.path.splitext(uploaded_file.name)[1].lower()
        if ext in archive_exts:
            return "File", f"🗜️ [1차 에이전트] 압축 파일 감지: `{uploaded_file.name}` ({ext})", False
        return "File", f"📂 [1차 에이전트] 파일 업로드 감지: `{uploaded_file.name}`", False

    if not text_stripped:
        return "Unknown", "", True

    if re.match(r'^https?://', text_stripped, re.IGNORECASE):
        return "URL", "🔗 [1차 에이전트] http(s):// URL 형식 감지", False
    if re.match(r'^www\.', text_stripped, re.IGNORECASE):
        return "URL", "🔗 [1차 에이전트] www. 도메인 형식 감지", False
    if re.match(r'^[a-zA-Z0-9\-]+\.[a-zA-Z]{2,}(/.*)?$', text_stripped):
        return "URL", "🔗 [1차 에이전트] 도메인 패턴 감지", False

    if is_email_addr or is_email_body:
        return "Email", "📧 [1차 에이전트] 이메일 형식 감지", False

    return "Unknown", "⚠️ [1차 에이전트] 분류 불확실 → 2차 API로 배분합니다.", True


def agent_dispatch_to_api(text: str) -> tuple[str, str]:
    text_lower = text.lower()

    file_keywords = ["악성코드", "malware", "바이러스", "virus", "exe", "dll", "랜섬웨어", "ransomware", "trojan", "worm"]
    if any(kw in text_lower for kw in file_keywords):
        return "File", "📂 [2차 API 배분] 악성파일 관련 키워드 감지 → File_Analyzer 배분"

    email_keywords = ["스팸", "spam", "피싱메일", "phishing", "첨부파일", "attachment", "이메일", "email", "mail"]
    if any(kw in text_lower for kw in email_keywords):
        return "Email", "📧 [2차 API 배분] 이메일/스팸 관련 키워드 감지 → Email_Analyzer 배분"

    url_keywords = ["링크", "link", "사이트", "site", "도메인", "domain", "접속", "redirect", "클릭"]
    if any(kw in text_lower for kw in url_keywords):
        return "URL", "🔗 [2차 API 배분] URL/링크 관련 키워드 감지 → URL_Analyzer 배분"

    return "URL", "🔗 [2차 API 배분] 감지 불가 → URL_Analyzer fallback 처리"


def build_rag_query(user_input: str, analysis_type: str, result: dict) -> str:
    base         = user_input.strip()
    features     = result.get("ml_result", {}).get("detected_features", [])
    feature_text = " ".join(features)
    type_keywords = {
        "URL":   "악성 URL 피싱 도메인 리다이렉트",
        "Email": "스팸 이메일 피싱 사회공학",
        "File":  "악성 파일 악성코드 실행파일 바이러스",
    }
    type_text = type_keywords.get(analysis_type, "보안 위협 악성")
    if base:
        return f"{base} {feature_text} {type_text}".strip()
    return f"{feature_text} {type_text}".strip()


def sanitize_for_pdf(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return text.replace("\n", " ").replace("\r", " ").strip()


def generate_pdf_report(user_input, analysis_type, result, rag_results) -> bytes:
    buffer = io.BytesIO()
    doc    = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=20*mm, leftMargin=20*mm,
        topMargin=20*mm,   bottomMargin=20*mm,
    )

    def S(name, **kw):
        base = dict(fontName=KOREAN_FONT, leading=16)
        base.update(kw)
        return ParagraphStyle(name, **base)

    T    = S("T",    fontSize=18, textColor=colors.HexColor("#1a1a2e"), spaceAfter=6, leading=26)
    H    = S("H",    fontSize=13, textColor=colors.HexColor("#16213e"), spaceBefore=12, spaceAfter=4)
    B    = S("B",    fontSize=10, textColor=colors.HexColor("#333333"), spaceAfter=4)
    Sm   = S("Sm",   fontSize=9,  textColor=colors.HexColor("#555555"))
    Lb   = S("Lb",   fontSize=9,  textColor=colors.HexColor("#1565c0"), spaceAfter=2)
    Sc   = S("Sc",   fontSize=8,  textColor=colors.HexColor("#888888"), spaceAfter=6)
    Warn = S("Warn", fontSize=9,  textColor=colors.HexColor("#b71c1c"), spaceAfter=4)

    story     = []
    risk      = float(result["ml_result"]["confidence_score"])
    risk_color = (
        colors.HexColor("#d32f2f") if risk > 70
        else colors.HexColor("#f57c00") if risk > 30
        else colors.HexColor("#388e3c")
    )
    risk_label = "위험" if risk > 70 else "주의" if risk > 30 else "안전"

    story.append(Paragraph("AutoGuard AI 보안 분석 리포트", T))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1a1a2e")))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "⚠️ 본 리포트는 데모용 Mock 데이터 기반입니다. 실제 ML/API 연동 전까지 참고용으로만 활용하세요.", Warn))
    story.append(Spacer(1, 6))

    story.append(Paragraph("분석 대상", H))
    story.append(Paragraph(f"분석 유형: {sanitize_for_pdf(analysis_type)}", B))
    story.append(Paragraph(f"입력값: {sanitize_for_pdf(user_input) or '(없음)'}", B))
    story.append(Spacer(1, 4))

    story.append(Paragraph("위험도 평가", H))
    features = result["ml_result"].get("detected_features", [])
    td = [
        ["항목", "결과"],
        ["위험도 점수",            f"{int(risk)}%"],
        ["악성 여부",              "악성" if result["ml_result"]["is_malicious"] else "정상"],
        ["탐지 비율 (VirusTotal)", result["vt_result"]["detection_ratio"]],
        ["위험 등급",              risk_label],
    ]
    if features:
        td.append(["탐지된 특징", " / ".join(features)])

    tbl = Table(td, colWidths=[60*mm, 100*mm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0),  colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR",     (0,0),(-1,0),  colors.white),
        ("FONTNAME",      (0,0),(-1,-1), KOREAN_FONT),
        ("FONTSIZE",      (0,0),(-1,-1), 10),
        ("ALIGN",         (0,0),(-1,-1), "LEFT"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.HexColor("#f5f5f5"), colors.white]),
        ("GRID",          (0,0),(-1,-1), 0.5, colors.HexColor("#cccccc")),
        ("TOPPADDING",    (0,0),(-1,-1), 6),
        ("BOTTOMPADDING", (0,0),(-1,-1), 6),
        ("LEFTPADDING",   (0,0),(-1,-1), 8),
        ("TEXTCOLOR",     (1,1),(1,1),   risk_color),
        ("TEXTCOLOR",     (1,4),(1,4),   risk_color),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 8))

    story.append(Paragraph("AI 분석 요약", H))
    story.append(Paragraph(sanitize_for_pdf(result["ai_agent_report"]["summary"]), B))
    story.append(Spacer(1, 6))

    story.append(Paragraph("권장 대응 방법", H))
    for i, step in enumerate(result["ai_agent_report"]["steps_to_take"], 1):
        story.append(Paragraph(f"{i}. {sanitize_for_pdf(step)}", B))
    story.append(Spacer(1, 8))

    story.append(Paragraph("보안 가이드 관련 내용 (벡터 DB 검색)", H))
    if rag_results:
        for r in rag_results:
            story.append(Paragraph(f"[검색 결과 {r['rank']}]", Lb))
            story.append(Paragraph(f"유사도 점수: {r['score']}%", Sc))
            story.append(Paragraph(sanitize_for_pdf(r["chunk"][:500]), Sm))
            story.append(Spacer(1, 6))
    else:
        story.append(Paragraph("업로드된 PDF에서 관련 내용을 찾을 수 없습니다.", B))

    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#aaaaaa")))
    story.append(Paragraph("⚠️ AutoGuard AI 데모 버전 자동 생성 리포트 (Mock 데이터 기반)", Warn))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()


def mock_fastapi_request(_user_input: str = "", analysis_type: str = "URL") -> dict:
    time.sleep(0.5)

    mock_profiles = {
        "URL": {
            "ml_result": {
                "module":            "URL_Analyzer",
                "is_malicious":      True,
                "confidence_score":  92.5,
                "detected_features": ["URL 길이 과다", "의심 도메인", "리다이렉트 다수"],
            },
            "vt_result": {"detection_ratio": "15/90", "malicious_count": 15, "harmless_count": 75},
            "ai_agent_report": {
                "summary": (
                    "이 URL은 피싱(Phishing) 특성을 보이며, "
                    "사용자 정보를 탈취하려는 악성 시도로 판단됩니다."
                ),
                "steps_to_take": [
                    "의심스러운 링크를 클릭하지 마세요.",
                    "즉시 비밀번호를 변경하세요.",
                    "보안팀에 즉시 보고하세요.",
                    "해당 URL을 차단 목록에 추가하세요.",
                ],
            },
        },
        "Email": {
            "ml_result": {
                "module":            "Email_Analyzer",
                "is_malicious":      True,
                "confidence_score":  78.3,
                "detected_features": ["발신자 도메인 위조", "긴급 유도 표현", "악성 첨부파일 의심"],
            },
            "vt_result": {"detection_ratio": "8/90", "malicious_count": 8, "harmless_count": 82},
            "ai_agent_report": {
                "summary": (
                    "이 이메일은 스피어 피싱 공격으로 판단됩니다. "
                    "발신자 정보가 위조되었으며, 첨부파일에 악성코드가 포함되어 있을 가능성이 높습니다."
                ),
                "steps_to_take": [
                    "첨부파일을 절대 열지 마세요.",
                    "이메일 내 링크를 클릭하지 마세요.",
                    "발신자에게 별도 채널(전화 등)로 진위를 확인하세요.",
                    "보안팀에 해당 이메일을 신고하세요.",
                ],
            },
        },
        "File": {
            "ml_result": {
                "module":            "File_Analyzer",
                "is_malicious":      True,
                "confidence_score":  85.7,
                "detected_features": ["실행 가능 코드 포함", "난독화 탐지", "의심 API 호출", "자동 실행 등록 시도"],
            },
            "vt_result": {"detection_ratio": "22/90", "malicious_count": 22, "harmless_count": 68},
            "ai_agent_report": {
                "summary": (
                    "이 파일은 악성코드(Malware) 특성을 보이며, "
                    "난독화된 실행 코드와 시스템 레지스트리 조작 시도가 탐지되었습니다."
                ),
                "steps_to_take": [
                    "해당 파일을 절대 실행하지 마세요.",
                    "파일을 격리(Quarantine) 처리하세요.",
                    "전체 시스템 바이러스 검사를 수행하세요.",
                    "보안팀에 파일 샘플을 전달하세요.",
                ],
            },
        },
    }

    if analysis_type == "Email+File":
        profile = dict(mock_profiles["Email"])
        profile["ml_result"] = dict(profile["ml_result"])
        profile["ml_result"]["detected_features"] = [
            "발신자 도메인 위조", "긴급 유도 표현",
            "첨부파일 악성코드 포함", "난독화 실행코드 탐지",
        ]
        profile["ml_result"]["confidence_score"] = 88.1
        profile["ai_agent_report"] = {
            "summary": (
                "이메일 본문과 첨부파일 모두에서 악성 특성이 탐지되었습니다. "
                "스피어 피싱 공격으로 판단되며, 첨부파일에 악성코드가 포함되어 있습니다."
            ),
            "steps_to_take": [
                "첨부파일을 절대 열지 마세요.",
                "이메일을 즉시 격리하세요.",
                "발신자 진위를 별도 채널로 확인하세요.",
                "보안팀에 이메일과 파일 샘플을 전달하세요.",
                "전체 시스템 바이러스 검사를 수행하세요.",
            ],
        }
        return {"status": "success", **profile}

    if analysis_type == "Email+URL":
        profile = dict(mock_profiles["Email"])
        profile["ml_result"] = dict(profile["ml_result"])
        profile["ml_result"]["detected_features"] = [
            "발신자 도메인 위조", "악성 URL 포함", "리다이렉트 감지", "긴급 유도 표현",
        ]
        profile["ml_result"]["confidence_score"] = 84.6
        profile["ai_agent_report"] = {
            "summary": (
                "이메일 본문 내 악성 URL이 포함된 피싱 이메일로 판단됩니다. "
                "링크 클릭 시 사용자 정보가 탈취될 위험이 높습니다."
            ),
            "steps_to_take": [
                "이메일 내 링크를 절대 클릭하지 마세요.",
                "해당 URL을 차단 목록에 추가하세요.",
                "발신자 진위를 별도 채널로 확인하세요.",
                "보안팀에 즉시 신고하세요.",
            ],
        }
        return {"status": "success", **profile}

    if analysis_type == "Email+URL+File":
        profile = dict(mock_profiles["Email"])
        profile["ml_result"] = dict(profile["ml_result"])
        profile["ml_result"]["detected_features"] = [
            "발신자 도메인 위조", "악성 URL 포함", "첨부파일 악성코드", "난독화 탐지", "자동 실행 시도",
        ]
        profile["ml_result"]["confidence_score"] = 96.2
        profile["ai_agent_report"] = {
            "summary": (
                "이메일 본문, 악성 URL, 첨부파일 3가지 모두에서 악성 특성이 탐지된 "
                "고위험 복합 공격입니다. 즉각적인 격리와 대응이 필요합니다."
            ),
            "steps_to_take": [
                "이메일을 즉시 격리하세요.",
                "첨부파일을 절대 열지 마세요.",
                "이메일 내 링크를 클릭하지 마세요.",
                "보안팀에 즉시 신고 및 전체 시스템 검사를 수행하세요.",
                "해당 발신 도메인을 차단 목록에 추가하세요.",
            ],
        }
        return {"status": "success", **profile}

    profile = mock_profiles.get(analysis_type, mock_profiles["URL"])
    return {"status": "success", **profile}


# ── 페이지 설정 ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="AutoGuard AI", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
    .main-header { font-size:2rem; font-weight:800; color:#1a1a2e; margin-bottom:0.2rem; }
    .sub-header  { color:#555; font-size:0.95rem; margin-bottom:1rem; }
    .risk-high   { color:#d32f2f; font-weight:bold; font-size:1.3rem; }
    .risk-medium { color:#f57c00; font-weight:bold; font-size:1.3rem; }
    .risk-safe   { color:#388e3c; font-weight:bold; font-size:1.3rem; }
    .rag-box {
        background:#f0f4ff; border-left:4px solid #3f51b5;
        padding:10px 14px; border-radius:4px;
        font-size:0.88rem; margin-bottom:8px;
    }
    .score-badge {
        display:inline-block; background:#e8eaf6; color:#3f51b5;
        font-size:0.78rem; font-weight:600; padding:2px 8px;
        border-radius:10px; margin-bottom:6px;
    }
    .vs-status {
        background:#e8f5e9; border:1px solid #a5d6a7;
        padding:8px 12px; border-radius:6px;
        font-size:0.85rem; color:#2e7d32; margin-bottom:8px;
    }
    .mock-warning {
        background:#fff3e0; border-left:4px solid #f57c00;
        padding:8px 14px; border-radius:4px;
        font-size:0.85rem; color:#e65100; margin-bottom:12px;
    }
    .detect-badge {
        background:#e3f2fd; border-left:4px solid #1976d2;
        padding:6px 12px; border-radius:4px;
        font-size:0.85rem; color:#1565c0; margin-top:6px;
    }
    .input-error-msg {
        color: #d32f2f;
        font-size: 0.85rem;
        margin-top: -8px;
        margin-bottom: 4px;
    }
    /* 사이드바 이력 카드 텍스트 설정 */
    .history-content {
        font-size: 0.85rem;
        color: #333;
        line-height: 1.5;
        word-break: break-all;
        margin-top: -8px !important;
    }
    .history-content p {
        margin: 0 !important;
        padding: 0 !important;
    }
    /* 카드 전체 크기와 여백 설정 */
    section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.history-content) {
        background: #fdfdfd !important;
        border: 1px solid #e0e0e0 !important;
        border-radius: 8px !important;
        margin-bottom: 12px !important;
        gap: 0 !important;
        align-items: center !important;
        padding: 14px 16px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03) !important;
    }
    /* 위험도 색상별 테두리 포인트 */
    section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.history-high) {
        border-left: 5px solid #d32f2f !important;
    }
    section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.history-medium) {
        border-left: 5px solid #f57c00 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.history-safe) {
        border-left: 5px solid #388e3c !important;
    }
    /* X 삭제 버튼 스타일 */
    section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.history-content) button {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: #999 !important;
        font-size: 1.1rem !important;
        width: 28px !important;
        height: 28px !important;
        min-height: unset !important;
        line-height: 1 !important;
        border-radius: 6px !important;
        cursor: pointer !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.history-content) button:hover {
        color: #d32f2f !important;
        background: rgba(211, 47, 47, 0.1) !important;
        width: 28px !important;
        height: 28px !important;
        border-radius: 6px !important;
        padding: 0 !important;
    }
    /* Streamlit 파일 업로더 문구 숨기기 */
    [data-testid="stFileUploader"] small { display: none !important; }
    [data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] small { display: none !important; }
    .stFileUploader small { display: none !important; }
    .uploadedFile small { display: none !important; }
    /* 텍스트 입력창 기본 스타일 */
    div[data-testid="stTextArea"] textarea {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-left: 5px solid #3b82f6 !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.15), 0 2px 4px -1px rgba(0,0,0,0.1) !important;
    }
    /* 파일 업로드 영역 */
    div[data-testid="stFileUploader"] > section {
        background-color: #f1f5f9 !important;
        border: 2px dashed #64748b !important;
        border-radius: 8px !important;
        padding: 1.5rem !important;
    }
    /* 분석 시작 버튼 */
    div.stButton > button[kind="primary"] {
        background-color: #1D4ED8 !important;
        border-color: #1D4ED8 !important;
        color: #ffffff !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #1e40af !important;
        border-color: #1e40af !important;
    }
</style>
""", unsafe_allow_html=True)


# ── session_state 초기화 ──────────────────────────────────────────────────────
if "vector_store" not in st.session_state:
    restored = load_vector_store_json()
    st.session_state.vector_store   = restored
    st.session_state.vs_pdf_name    = restored.source_name if restored else None
    st.session_state.vs_chunk_count = len(restored.chunks) if restored else 0

if "history" not in st.session_state:
    st.session_state.history = load_history()

if "input_error" not in st.session_state:
    st.session_state.input_error = False

if "analyze_target" not in st.session_state:
    st.session_state.analyze_target = ""


# ── 헤더 ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🛡️ AutoGuard AI 보안 분석 시스템</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">URL · 이메일 · 파일을 AI로 즉시 분석하고 PDF 리포트를 생성합니다.</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="mock-warning">'
    '⚠️ <b>데모 모드</b> : 현재 분석 결과는 Mock(더미) 데이터입니다. '
    '실제 ML 모델 및 VirusTotal API 연동 후 정확한 분석이 가능합니다.'
    '</div>',
    unsafe_allow_html=True,
)

col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown("#### 🔎 분석 대상 입력")
    st.caption("URL · 이메일 · 파일 중 하나 또는 복합 입력이 가능합니다. 유형은 자동으로 감지됩니다.")

    # on_change 콜백: 글자 입력 즉시 오류 상태 해제 + rerun으로 CSS도 즉시 제거
    def clear_input_error():
        if st.session_state.get("main_input", "").strip():
            st.session_state.input_error = False

    # 스타일 렌더링 전에 먼저 오류 상태를 정리해야 바로 파란 테두리로 복원됨
    current_input_text = st.session_state.get("main_input", "")
    current_uploaded_file = st.session_state.get("target_file_uploader")
    if st.session_state.input_error and (current_input_text.strip() or current_uploaded_file is not None):
        st.session_state.input_error = False

    error_css = ""
    if st.session_state.input_error:
        error_css = """
        div[data-testid='stTextArea'] textarea:placeholder-shown {
            border: 2px solid #d32f2f !important;
            border-left: 5px solid #d32f2f !important;
            box-shadow: 0 0 0 3px rgba(211,47,47,0.15) !important;
        }
        div[data-testid='stAppViewContainer']:has(
            div[data-testid='stTextArea'] textarea:not(:placeholder-shown)
        ) .input-error-msg {
            display: none !important;
        }
        """

    st.markdown(
        f"""
        <style>
        div[data-testid="stTextArea"] textarea {{
            border: 1px solid #cbd5e1 !important;
            border-left: 5px solid #3b82f6 !important;
            box-shadow: none !important;
        }}
        {error_css}
        </style>
        """,
        unsafe_allow_html=True,
    )

    user_input = st.text_area(
        "텍스트 입력",
        placeholder=(
            "예시:\n"
            "• URL       → https://suspicious-site.com\n"
            "• 이메일    → attacker@phishing.com\n"
            "• 이메일 본문 → 수신: 홍길동 / 발신: admin@fake.com / 제목: 긴급 공지\n"
            "• 복합 입력 → 본문 내 URL·첨부파일 정보를 함께 입력하세요"
        ),
        height=140,
        label_visibility="collapsed",
        key="main_input",
        on_change=clear_input_error,
    )

    # 오류 상태일 때만 경고 문구 표시
    if st.session_state.input_error:
        st.markdown(
            """
            <style>
            .input-error-msg { display: block !important; }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="input-error-msg">⚠️ URL, 이메일, 또는 파일을 입력해 주세요.</p>',
            unsafe_allow_html=True,
        )

    st.markdown(
        "**파일 업로드** "
        "<span style='color:#888;font-size:0.82rem'>"
        "(단독 악성파일 분석 또는 이메일 첨부파일로 함께 분석 가능 · 최대 1,000 KB)</span>",
        unsafe_allow_html=True,
    )

    uploaded_target_file = st.file_uploader(
        "분석 대상 파일",
        type=[
            "exe", "dll", "bat", "sh", "py", "js",
            "pdf", "doc", "docx", "hwp",
            "zip", "rar", "7z", "tar", "gz", "bz2", "xz",
            "cab", "iso",
        ],
        key="target_file_uploader",
        label_visibility="collapsed",
    )

    if uploaded_target_file:
        file_kb = uploaded_target_file.size / 1024
        if file_kb > 1000:
            st.error(f"❌ 파일 크기 초과: `{file_kb:.1f} KB` (최대 1,000 KB)")
        else:
            if user_input.strip():
                st.success(
                    f"📎 복합 입력 감지: 텍스트 + 파일 `{uploaded_target_file.name}` ({file_kb:.1f} KB)\n\n"
                    f"→ 이메일 본문 + 첨부파일로 함께 분석합니다."
                )
            else:
                st.info(
                    f"📂 파일명: `{uploaded_target_file.name}`  |  크기: `{file_kb:.1f} KB`"
                )

    detected_type, detect_reason, needs_api = agent_classify(user_input, uploaded_target_file)
    if detect_reason:
        badge_color = {
            "URL":     "#e3f2fd",
            "Email":   "#f3e5f5",
            "File":    "#e8f5e9",
            "Unknown": "#fff8e1",
        }.get(detected_type, "#fff8e1")
        st.markdown(
            f'<div class="detect-badge" style="background:{badge_color};">'
            f'감지 결과: <b>{detected_type}</b> &nbsp;|&nbsp; {detect_reason}'
            f'{"&nbsp; → 2차 API 배분 예정" if needs_api else ""}'
            f'</div>',
            unsafe_allow_html=True,
        )

with col_right:
    uploaded_pdf = st.file_uploader(
        "📄 보안 가이드 PDF 업로드",
        type="pdf",
        key="guide_pdf_uploader",
        help="KISA 가이드, 침해사고 매뉴얼 등을 업로드하면 벡터 DB에 자동 인덱싱됩니다.",
    )

    if uploaded_pdf is not None:
        if uploaded_pdf.name != st.session_state.vs_pdf_name:
            with st.spinner("🔄 벡터 DB 구축 중..."):
                vs, used_ocr = build_vector_store(uploaded_pdf, source_name=uploaded_pdf.name)
                if vs.is_built:
                    st.session_state.vector_store   = vs
                    st.session_state.vs_pdf_name    = uploaded_pdf.name
                    st.session_state.vs_chunk_count = len(vs.chunks)
                    save_vector_store_json(vs)
                    ocr_msg = " (OCR 사용)" if used_ocr else ""
                    st.success(
                        f"✅ 벡터 DB 완성{ocr_msg}!\n\n"
                        f"파일: {uploaded_pdf.name}\n"
                        f"청크 {len(vs.chunks)}개 · "
                        f"{'Semantic' if vs.use_semantic else 'TF-IDF'} 검색 활성"
                    )
                else:
                    st.error("PDF 텍스트 추출 실패. OCR도 사용 불가 상태입니다.")
        else:
            st.markdown(
                f'<div class="vs-status">'
                f'🗄️ <b>벡터 DB 활성</b><br>'
                f'파일: {st.session_state.vs_pdf_name}<br>'
                f'인덱싱 청크: {st.session_state.vs_chunk_count}개'
                f'</div>',
                unsafe_allow_html=True,
            )
    elif st.session_state.vector_store is not None:
        st.markdown(
            f'<div class="vs-status">'
            f'🗄️ <b>벡터 DB 복원됨 (캐시)</b><br>'
            f'파일: {st.session_state.vs_pdf_name}<br>'
            f'인덱싱 청크: {st.session_state.vs_chunk_count}개'
            f'</div>',
            unsafe_allow_html=True,
        )

st.divider()
def start_analysis_callback():
    # 현재 입력된 텍스트를 백업 변수로 이동
    st.session_state.analyze_target = st.session_state.get("main_input", "")
    # 텍스트 입력창 비우기
    st.session_state.main_input = ""

# ── 분석 시작 버튼 ────────────────────────────────────────────────────────────
analyze_clicked = st.button("🔍 분석 시작", type="primary", use_container_width=True, on_click=start_analysis_callback)

# Enter: 분석 시작, Shift+Enter: 줄바꿈 (전역 캡처, 버튼 렌더 이후 바인딩)
components.html(
    """
    <script>
    (function () {
      const doc = window.parent.document;

      function getFocusedTextarea() {
        const el = doc.activeElement;
        if (!el || el.tagName !== "TEXTAREA") return null;
        const wrap = el.closest('div[data-testid="stTextArea"]');
        return wrap ? el : null;
      }

      function getAnalyzeButton() {
        const candidates = Array.from(doc.querySelectorAll("button"));
        return candidates.find((btn) => {
          const t = (btn.textContent || "").replace(/\\s+/g, " ").trim();
          return t.includes("분석 시작");
        }) || null;
      }

      function triggerAnalyzeClick() {
        const btn = getAnalyzeButton();
        if (!btn || btn.disabled) return false;
        btn.dispatchEvent(new MouseEvent("mousedown", { bubbles: true }));
        btn.dispatchEvent(new MouseEvent("mouseup", { bubbles: true }));
        btn.click();
        return true;
      }

      function onEnterSubmit(e) {
        if (e.isComposing) return;
        if (e.key !== "Enter") return;
        if (e.shiftKey) return; // Shift+Enter는 줄바꿈 허용
        const ta = getFocusedTextarea();
        if (!ta) return;

        e.preventDefault();
        e.stopPropagation();
        e.stopImmediatePropagation();

        // 핵심: 현재 textarea 값을 먼저 Streamlit 상태로 반영
        // (포커스 아웃 시 반영되는 환경을 고려해 blur까지 수행)
        const currentValue = ta.value || "";
        const nativeSetter = Object.getOwnPropertyDescriptor(
          window.HTMLTextAreaElement.prototype,
          "value"
        ).set;
        nativeSetter.call(ta, currentValue);
        ta.dispatchEvent(new Event("input", { bubbles: true }));
        ta.dispatchEvent(new Event("change", { bubbles: true }));
        ta.blur();

        // 값 반영 후 분석 버튼 클릭(재시도)
        setTimeout(() => {
          if (triggerAnalyzeClick()) return;
          setTimeout(triggerAnalyzeClick, 80);
          setTimeout(triggerAnalyzeClick, 160);
        }, 30);
      }

      // 이전 핸들러 제거 후 재바인딩
      if (window.__autoguardEnterHandler) {
        doc.removeEventListener("keydown", window.__autoguardEnterHandler, true);
        window.parent.removeEventListener("keydown", window.__autoguardEnterHandler, true);
      }
      window.__autoguardEnterHandler = onEnterSubmit;

      doc.addEventListener("keydown", onEnterSubmit, true);
      window.parent.addEventListener("keydown", onEnterSubmit, true);
    })();
    </script>
    """,
    height=0,
)

if analyze_clicked:
    
    current_input = st.session_state.get("analyze_target", "").strip()
    user_input = current_input

    if uploaded_target_file is None and not current_input:
        st.session_state.input_error = True
        st.rerun()

    st.session_state.input_error = False

    if uploaded_target_file is not None:
        if uploaded_target_file.size > 1000 * 1024:
            st.error(
                f"❌ 파일 크기 초과: `{uploaded_target_file.size / 1024:.1f} KB` "
                f"(최대 1,000 KB)\n\n업로드 가능한 파일 크기는 **1,000 KB 이하**입니다."
            )
            st.stop()

    analysis_type, detect_reason, needs_api = agent_classify(user_input, uploaded_target_file)

    progress_bar = st.progress(0)
    status_text  = st.empty()

    status_text.markdown("🤖 **[1차 에이전트] 입력값 분류 중...**")
    progress_bar.progress(15)
    time.sleep(0.4)

    if needs_api:
        status_text.markdown("🔀 **[2차 API 배분] 전문 분석 API로 배분 중...**")
        progress_bar.progress(30)
        time.sleep(0.4)
        analysis_type, detect_reason = agent_dispatch_to_api(user_input)
        st.info(f"🔀 2차 API 배분 완료: **{analysis_type}** | {detect_reason}")
    else:
        st.success(f"✅ 1차 에이전트 분류 완료: **{analysis_type}** | {detect_reason}")

    if uploaded_target_file is not None:
        user_input = uploaded_target_file.name

    status_text.markdown(f"📡 **{analysis_type} 전문 분석 중...**")
    progress_bar.progress(45)
    result = mock_fastapi_request(user_input, analysis_type)

    status_text.markdown("⚙️ **ML 모델 결과 처리 중...**")
    progress_bar.progress(65)
    time.sleep(0.3)

    status_text.markdown("🔎 **벡터 DB 보안 가이드 검색 중...**")
    progress_bar.progress(78)

    rag_results = []
    rag_query   = ""
    if st.session_state.vector_store is not None and st.session_state.vector_store.is_built:
        rag_query   = build_rag_query(user_input, analysis_type, result)
        rag_results = st.session_state.vector_store.search(rag_query, top_k=3)

    status_text.markdown("📄 **PDF 리포트 생성 중...**")
    progress_bar.progress(88)
    pdf_bytes = generate_pdf_report(user_input, analysis_type, result, rag_results)

    progress_bar.progress(100)
    status_text.markdown("✅ **분석 완료!**")
    time.sleep(0.4)
    progress_bar.empty()
    status_text.empty()

    risk_val = float(result["ml_result"]["confidence_score"])
    save_history({
        "input": user_input,
        "type":  analysis_type,
        "risk":  int(risk_val),
        "time":  datetime.datetime.now().strftime("%m/%d %H:%M"),
    })
    st.session_state.history = load_history()

    stage_label = "2차 API 배분" if needs_api else "1차 에이전트"
    st.info(f"🤖 [{stage_label}] 분석 유형: **{analysis_type}** | {detect_reason}")

    vt = result["vt_result"]
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("위험도 점수", f"{int(risk_val)}%")
        if risk_val > 70:
            st.markdown('<span class="risk-high">🚨 위험</span>', unsafe_allow_html=True)
        elif risk_val > 30:
            st.markdown('<span class="risk-medium">⚠️ 주의</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="risk-safe">✅ 안전</span>', unsafe_allow_html=True)
    with c2:
        st.metric("악성 여부", "악성 탐지" if result["ml_result"]["is_malicious"] else "정상")
    with c3:
        st.metric("VirusTotal 탐지", vt["detection_ratio"])

    features = result["ml_result"].get("detected_features", [])
    if features:
        st.markdown("**탐지된 특징:** " + "  ".join(f"`{f}`" for f in features))

    st.subheader("📌 AI 분석 요약")
    st.info(result["ai_agent_report"]["summary"])

    st.subheader("📌 권장 대응 방법")
    for step in result["ai_agent_report"]["steps_to_take"]:
        st.write(f"• {step}")

    if rag_results:
        st.subheader("📚 보안 가이드 검색 결과 (벡터 DB)")
        st.caption(f"🔍 검색 쿼리: `{rag_query}`")
        for r in rag_results:
            st.markdown(
                f'<div class="rag-box">'
                f'<b>[결과 {r["rank"]}]</b> '
                f'<span class="score-badge">유사도 {r["score"]}%</span><br>'
                f'{r["chunk"][:350].replace(chr(10), " ")}'
                f'</div>',
                unsafe_allow_html=True,
            )
    elif st.session_state.vector_store is None:
        st.info("💡 보안 가이드 PDF를 업로드하면 벡터 DB 검색 결과가 리포트에 포함됩니다.")

    st.success("✅ 분석 완료! 아래에서 리포트를 다운로드하세요.")
    st.download_button(
        label="📥 PDF 리포트 다운로드",
        data=pdf_bytes,
        file_name="autoguard_security_report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )


# ── 사이드바 ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📋 분석 이력")
    st.caption(f"총 {len(st.session_state.history)}건")

    if st.button("🗑️ 이력 초기화", use_container_width=True):
        st.session_state.history = []
        if os.path.exists(HISTORY_PATH):
            os.remove(HISTORY_PATH)
        st.rerun()

    if not st.session_state.history:
        st.info("아직 분석 이력이 없습니다.")
    else:
        for idx, item in enumerate(st.session_state.history):
            risk_val = item.get("risk", 0)
            css_cls  = "history-high" if risk_val > 70 else "history-medium" if risk_val > 30 else "history-safe"
            icon     = "🚨" if risk_val > 70 else "⚠️" if risk_val > 30 else "✅"
            label    = item.get("input", "(없음)")[:28]
            atype    = item.get("type", "")
            ts       = item.get("time", "")
            cols = st.columns([7, 1])
            with cols[0]:
                st.markdown(
                    f'<div class="history-content {css_cls}">'
                    f'{icon} <b>[{atype}]</b><br>'
                    f'{label}<br>'
                    f'<span style="color:#888;font-size:0.78rem">위험도 {risk_val}% · {ts}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with cols[1]:
                if st.button("✕", key=f"del_history_{idx}", help="이 항목 삭제"):
                    delete_history_item(idx)
                    st.rerun()

    st.divider()
    st.markdown("### ⚙️ 시스템 정보")

    def get_badge(is_active):
        if is_active:
            bg_color   = "#dcfce7"
            text_color = "#166534"
            border     = "1px solid #bbf7d0"
            icon       = "✅ 활성"
        else:
            bg_color   = "#fee2e2"
            text_color = "#b91c1c"
            border     = "1px solid #fecaca"
            icon       = "❌ 비활성"
        return (
            f'<span style="background:{bg_color}; color:{text_color}; border:{border}; '
            f'padding:4px 12px; border-radius:14px; font-size:0.75rem; font-weight:800; '
            f'margin-left:8px; box-shadow:0 1px 2px rgba(0,0,0,0.1);">{icon}</span>'
        )

    search_engine = "🧠 Semantic" if SEMANTIC_AVAILABLE else "📊 TF-IDF"
    ocr_badge     = get_badge(OCR_AVAILABLE)
    vdb_badge     = get_badge(st.session_state.vector_store is not None)

    st.markdown(f"**검색 엔진** : {search_engine}")
    st.markdown(f"**OCR 지원** {ocr_badge}", unsafe_allow_html=True)
    st.markdown(f"**벡터 DB** {vdb_badge}", unsafe_allow_html=True)