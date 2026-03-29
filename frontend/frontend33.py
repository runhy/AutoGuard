import streamlit as st
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


VS_JSON_PATH  = "vector_store_cache.json"  
HISTORY_PATH  = "analysis_history.json"    


# 1. 한글 폰트
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


# 2. 벡터 DB (TF-IDF 기본 / Sentence-Transformers 고도화)
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

        all_words  = set()
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


# 3. 벡터 DB JSON 영속성 (Pickle 보안 이슈 방지)
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



# 4. 분석 이력 저장/불러오기
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
    history = history[:30]   # 최대 30건 유지
    try:
        with open(HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False)
    except Exception:
        pass


# 5. PDF 텍스트 추출 + OCR 폴백
def load_pdf_with_ocr(file) -> tuple[str, bool]:
    """
    1차: PyPDF2로 텍스트 추출
    2차: 텍스트 부족 시 OCR(pdf2image + pytesseract) 시도
    """
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



# 6. RAG 검색 쿼리 보강
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


# 7. PDF 리포트 생성
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

    story    = []
    risk     = float(result["ml_result"]["confidence_score"])
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
        ["위험도 점수",           f"{int(risk)}%"],
        ["악성 여부",             "악성" if result["ml_result"]["is_malicious"] else "정상"],
        ["탐지 비율 (VirusTotal)", result["vt_result"]["detection_ratio"]],
        ["위험 등급",             risk_label],
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



# 8. Mock API
def mock_fastapi_request(_user_input: str = "") -> dict:
    time.sleep(0.5)
    return {
        "status": "success",
        "ml_result": {
            "module":            "URL_Analyzer",
            "is_malicious":      True,
            "confidence_score":  92.5,
            "detected_features": ["URL 길이 과다", "의심 도메인", "리다이렉트 다수"],
        },
        "vt_result": {
            "detection_ratio": "15/90",
            "malicious_count": 15,
            "harmless_count":  75,
        },
        "ai_agent_report": {
            "summary": (
                "이 입력값은 피싱(Phishing) 특성을 보이며, "
                "사용자 정보를 탈취하려는 악성 시도로 판단됩니다. "
                "즉각적인 대응이 필요합니다."
            ),
            "steps_to_take": [
                "의심스러운 링크를 클릭하지 마세요.",
                "즉시 비밀번호를 변경하세요.",
                "보안팀에 즉시 보고하세요.",
                "해당 URL을 차단 목록에 추가하세요.",
            ],
        },
    }



# 9. Streamlit UI
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
    .history-item   { background:#f9f9f9; border:1px solid #e0e0e0; padding:8px 10px;
                      border-radius:6px; font-size:0.82rem; margin-bottom:6px; color:#333; }
    .history-high   { border-left:4px solid #d32f2f; }
    .history-medium { border-left:4px solid #f57c00; }
    .history-safe   { border-left:4px solid #388e3c; }
</style>
""", unsafe_allow_html=True)


if "vector_store" not in st.session_state:
    restored = load_vector_store_json()
    st.session_state.vector_store   = restored
    st.session_state.vs_pdf_name    = restored.source_name if restored else None
    st.session_state.vs_chunk_count = len(restored.chunks) if restored else 0

if "history" not in st.session_state:
    st.session_state.history = load_history()


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
        for item in st.session_state.history:
            risk_val = item.get("risk", 0)
            css_cls  = "history-high" if risk_val > 70 else "history-medium" if risk_val > 30 else "history-safe"
            icon     = "🚨" if risk_val > 70 else "⚠️" if risk_val > 30 else "✅"
            label    = item.get("input", "(없음)")[:28]
            atype    = item.get("type", "")
            ts       = item.get("time", "")
            st.markdown(
                f'<div class="history-item {css_cls}">'
                f'{icon} <b>[{atype}]</b> {label}<br>'
                f'<span style="color:#888;font-size:0.78rem">위험도 {risk_val}% · {ts}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.divider()
    st.markdown("### ⚙️ 시스템 정보")
    st.markdown(f"- 검색 엔진: {'🧠 Semantic' if SEMANTIC_AVAILABLE else '📊 TF-IDF'}")
    st.markdown(f"- OCR 지원: {'✅ 활성' if OCR_AVAILABLE else '❌ 비활성'}")
    st.markdown(f"- 벡터 DB: {'✅ 활성' if st.session_state.vector_store else '❌ 미로드'}")


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
    analysis_type = st.radio("분석 유형", ["URL", "Email", "File"], horizontal=True)

    uploaded_target_file = None
    if analysis_type == "File":

        st.markdown("**분석할 파일을 업로드하세요**")
        uploaded_target_file = st.file_uploader(
            "분석 대상 파일 (exe, dll, zip, pdf 등)",
            type=["exe","dll","zip","pdf","doc","docx","hwp","js","py","sh","bat"],
            key="target_file_uploader",
        )
        if uploaded_target_file:
            st.info(
                f"📂 파일명: `{uploaded_target_file.name}`  |  "
                f"크기: `{uploaded_target_file.size / 1024:.1f} KB`"
            )
        user_input = uploaded_target_file.name if uploaded_target_file else ""
    else:
        user_input = st.text_area(
            "분석할 내용을 입력하세요",
            placeholder="예: https://suspicious-site.com" if analysis_type == "URL"
                        else "예: 수신된 이메일 본문 또는 발신자 주소",
            height=120,
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



if st.button("🔍 분석 시작", type="primary", use_container_width=True):

    if analysis_type == "File" and uploaded_target_file is None:
        st.warning("분석할 파일을 업로드해 주세요.")
        st.stop()
    elif analysis_type != "File" and not user_input.strip():
        st.warning("분석할 내용을 입력해 주세요.")
        st.stop()

    progress_bar = st.progress(0)
    status_text  = st.empty()

    status_text.markdown("🔍 **요청 수신 및 초기화 중...**")
    progress_bar.progress(10)
    time.sleep(0.3)

    status_text.markdown("📡 **AI 에이전트에 분석 요청 중...**")
    progress_bar.progress(30)
    result = mock_fastapi_request(user_input)

    status_text.markdown("⚙️ **ML 모델 결과 처리 중...**")
    progress_bar.progress(55)
    time.sleep(0.3)

    status_text.markdown("🔎 **벡터 DB 보안 가이드 검색 중...**")
    progress_bar.progress(70)


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
        "input": user_input or (uploaded_target_file.name if uploaded_target_file else ""),
        "type":  analysis_type,
        "risk":  int(risk_val),
        "time":  datetime.datetime.now().strftime("%m/%d %H:%M"),
    })
    st.session_state.history = load_history()


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