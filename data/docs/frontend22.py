"""
AutoGuard AI - 보안 분석 시스템
벡터 DB 기반 RAG (TF-IDF 임베딩 + 코사인 유사도) 완전 구현
"""

import streamlit as st
import time
import os
import io
import re
import pickle
from collections import defaultdict

import PyPDF2
import numpy as np

# PDF 생성: reportlab (한글 완벽 지원)
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


# ============================================================
# 1. 한글 폰트 등록
# ============================================================
def register_korean_font() -> str:
    font_candidates = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/Library/Fonts/AppleGothic.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/gulim.ttc",
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


# ============================================================
# 2. 벡터 DB 클래스 (TF-IDF 임베딩 + 코사인 유사도)
# ============================================================
class VectorStore:
    """
    TF-IDF 기반 인메모리 벡터 DB.
    sentence-transformers/faiss 없이 numpy만으로 구현.
    - PDF 업로드 시 build() 1회 실행 → 벡터 행렬 생성
    - 검색 시 쿼리를 동일 공간에 벡터화 → 코사인 유사도 계산
    """

    def __init__(self):
        self.chunks: list = []
        self.vectors = None        # np.ndarray (N x V)
        self.vocab: dict = {}      # 단어 -> 인덱스
        self.idf = None            # np.ndarray (V,)
        self.is_built = False

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

    def build(self, chunks: list) -> None:
        self.chunks = chunks
        N = len(chunks)
        tokenized = [self._tokenize(c) for c in chunks]

        all_words = set()
        for tokens in tokenized:
            all_words.update(tokens)
        self.vocab = {w: i for i, w in enumerate(sorted(all_words))}
        V = len(self.vocab)

        if V == 0:
            self.is_built = False
            return

        # IDF 계산
        df = np.zeros(V, dtype=np.float32)
        for tokens in tokenized:
            for w in set(tokens):
                if w in self.vocab:
                    df[self.vocab[w]] += 1.0
        self.idf = np.log((N + 1) / (df + 1)) + 1.0

        # TF-IDF 행렬
        matrix = np.zeros((N, V), dtype=np.float32)
        for idx, tokens in enumerate(tokenized):
            tf = self._compute_tf(tokens)
            for w, tf_val in tf.items():
                if w in self.vocab:
                    j = self.vocab[w]
                    matrix[idx, j] = tf_val * self.idf[j]

        # L2 정규화
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.vectors = matrix / norms
        self.is_built = True

    def _query_to_vector(self, query: str) -> np.ndarray:
        tokens = self._tokenize(query)
        V = len(self.vocab)
        vec = np.zeros(V, dtype=np.float32)
        tf = self._compute_tf(tokens)
        for w, tf_val in tf.items():
            if w in self.vocab:
                j = self.vocab[w]
                vec[j] = tf_val * self.idf[j]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def search(self, query: str, top_k: int = 3) -> list:
        if not self.is_built or self.vectors is None:
            return []
        q_vec = self._query_to_vector(query)
        scores = self.vectors @ q_vec
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for rank, idx in enumerate(top_indices, 1):
            score = float(scores[idx])
            if score > 0.0:
                results.append({
                    "chunk": self.chunks[idx],
                    "score": round(score * 100, 1),
                    "rank": rank,
                })
        return results


# ============================================================
# 3. PDF 텍스트 추출 + 청크 분할
# ============================================================
def load_pdf(file) -> str:
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text


def split_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list:
    sentences = re.split(r"(?<=[.!?。\n])\s+", text.strip())
    chunks = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) <= chunk_size:
            current += " " + sent
        else:
            if current.strip():
                chunks.append(current.strip())
            overlap_text = current[-overlap:] if len(current) > overlap else current
            current = overlap_text + " " + sent
    if current.strip():
        chunks.append(current.strip())
    return [c for c in chunks if len(c) > 20]


def build_vector_store(file) -> VectorStore:
    pdf_text = load_pdf(file)
    chunks = split_text(pdf_text)
    vs = VectorStore()
    vs.build(chunks)
    return vs


# ============================================================
# 4. PDF 리포트 생성
# ============================================================
def sanitize_for_pdf(text: str) -> str:
    """reportlab Paragraph에 넣기 전 특수문자 이스케이프 처리"""
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    return text.strip()


def generate_pdf_report(user_input: str, analysis_type: str, result: dict, rag_results: list) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=20*mm, leftMargin=20*mm,
        topMargin=20*mm, bottomMargin=20*mm,
    )

    def S(name, **kw):
        base = dict(fontName=KOREAN_FONT, leading=16)
        base.update(kw)
        return ParagraphStyle(name, **base)

    T  = S("T",  fontSize=18, textColor=colors.HexColor("#1a1a2e"), spaceAfter=6, leading=26)
    H  = S("H",  fontSize=13, textColor=colors.HexColor("#16213e"), spaceBefore=12, spaceAfter=4)
    B  = S("B",  fontSize=10, textColor=colors.HexColor("#333333"), spaceAfter=4)
    Sm = S("Sm", fontSize=9,  textColor=colors.HexColor("#555555"))
    Lb = S("Lb", fontSize=9,  textColor=colors.HexColor("#1565c0"), spaceAfter=2)
    Sc = S("Sc", fontSize=8,  textColor=colors.HexColor("#888888"), spaceAfter=6)

    story = []
    risk = float(result["ml_result"]["confidence_score"])
    risk_color = (
        colors.HexColor("#d32f2f") if risk > 70
        else colors.HexColor("#f57c00") if risk > 30
        else colors.HexColor("#388e3c")
    )
    risk_label = "위험" if risk > 70 else "주의" if risk > 30 else "안전"

    story.append(Paragraph("AutoGuard AI 보안 분석 리포트", T))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1a1a2e")))
    story.append(Spacer(1, 6))

    story.append(Paragraph("분석 대상", H))
    story.append(Paragraph(f"분석 유형: {sanitize_for_pdf(analysis_type)}", B))
    story.append(Paragraph(f"입력값: {sanitize_for_pdf(user_input) or '(없음)'}", B))
    story.append(Spacer(1, 4))

    story.append(Paragraph("위험도 평가", H))
    features = result["ml_result"].get("detected_features", [])
    td = [
        ["항목", "결과"],
        ["위험도 점수", f"{int(risk)}%"],
        ["악성 여부", "악성" if result["ml_result"]["is_malicious"] else "정상"],
        ["탐지 비율 (VirusTotal)", result["vt_result"]["detection_ratio"]],
        ["위험 등급", risk_label],
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
    story.append(Paragraph("본 리포트는 AutoGuard AI에 의해 자동 생성되었습니다.", Sm))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()


# ============================================================
# 5. Mock API
# ============================================================
def mock_fastapi_request(_user_input: str = "") -> dict:
    time.sleep(1)
    return {
        "status": "success",
        "ml_result": {
            "module": "URL_Analyzer",
            "is_malicious": True,
            "confidence_score": 92.5,
            "detected_features": ["URL 길이 과다", "의심 도메인", "리다이렉트 다수"],
        },
        "vt_result": {
            "detection_ratio": "15/90",
            "malicious_count": 15,
            "harmless_count": 75,
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


# ============================================================
# 6. Streamlit UI
# ============================================================
st.set_page_config(page_title="AutoGuard AI", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
    .main-header { font-size:2rem; font-weight:800; color:#1a1a2e; margin-bottom:0.2rem; }
    .sub-header  { color:#555; font-size:0.95rem; margin-bottom:1.5rem; }
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
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🛡️ AutoGuard AI 보안 분석 시스템</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">URL · 이메일 · 파일을 AI로 즉시 분석하고 PDF 리포트를 생성합니다.</div>', unsafe_allow_html=True)

# 세션 초기화
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "vs_pdf_name" not in st.session_state:
    st.session_state.vs_pdf_name = None
if "vs_chunk_count" not in st.session_state:
    st.session_state.vs_chunk_count = 0

# 입력 영역
col_left, col_right = st.columns([2, 1])

with col_left:
    analysis_type = st.radio("분석 유형", ["URL", "Email", "File"], horizontal=True)
    user_input = st.text_area(
        "분석할 내용을 입력하세요",
        placeholder="예: https://suspicious-site.com",
        height=120,
    )

with col_right:
    uploaded_pdf = st.file_uploader(
        "📄 보안 가이드 PDF 업로드",
        type="pdf",
        help="KISA 가이드, 침해사고 매뉴얼 등을 업로드하면 벡터 DB에 자동 인덱싱됩니다.",
    )

    # PDF 업로드 시 벡터 DB 자동 구축
    if uploaded_pdf is not None:
        if uploaded_pdf.name != st.session_state.vs_pdf_name:
            with st.spinner("🔄 벡터 DB 구축 중... (PDF 인덱싱)"):
                vs = build_vector_store(uploaded_pdf)
                if vs.is_built:
                    st.session_state.vector_store = vs
                    st.session_state.vs_pdf_name = uploaded_pdf.name
                    st.session_state.vs_chunk_count = len(vs.chunks)
                    st.success(
                        f"✅ 벡터 DB 완성!\n\n"
                        f"파일: {uploaded_pdf.name}\n"
                        f"청크 {len(vs.chunks)}개 · 어휘 {len(vs.vocab)}개 인덱싱 완료"
                    )
                else:
                    st.error("PDF 텍스트 추출 실패 (스캔 이미지 PDF는 OCR 필요)")
        else:
            st.markdown(
                f'<div class="vs-status">'
                f'🗄️ <b>벡터 DB 활성</b><br>'
                f'파일: {st.session_state.vs_pdf_name}<br>'
                f'인덱싱 청크: {st.session_state.vs_chunk_count}개'
                f'</div>',
                unsafe_allow_html=True,
            )

st.divider()

# 분석 버튼
if st.button("🔍 분석 시작", type="primary", use_container_width=True):

    if not user_input.strip() and not uploaded_pdf:
        st.warning("분석할 내용을 입력하거나 PDF를 업로드해 주세요.")
        st.stop()

    with st.spinner("AI가 분석 중입니다..."):
        result = mock_fastapi_request(user_input)

    risk = float(result["ml_result"]["confidence_score"])
    vt   = result["vt_result"]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("위험도 점수", f"{int(risk)}%")
        if risk > 70:
            st.markdown('<span class="risk-high">🚨 위험</span>', unsafe_allow_html=True)
        elif risk > 30:
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

    # 벡터 DB RAG 검색
    rag_results = []

    if st.session_state.vector_store is not None and st.session_state.vector_store.is_built:
        with st.spinner("🔎 벡터 DB에서 관련 보안 가이드 검색 중..."):
            rag_results = st.session_state.vector_store.search(user_input, top_k=3)

        st.subheader("📚 보안 가이드 검색 결과 (벡터 DB)")
        if rag_results:
            for r in rag_results:
                st.markdown(
                    f'<div class="rag-box">'
                    f'<b>[결과 {r["rank"]}]</b> '
                    f'<span class="score-badge">유사도 {r["score"]}%</span><br>'
                    f'{r["chunk"][:350].replace(chr(10), " ")}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.warning("관련 보안 가이드 내용을 찾지 못했습니다.")
    elif uploaded_pdf is None:
        st.info("💡 보안 가이드 PDF를 업로드하면 벡터 DB 검색 결과가 리포트에 포함됩니다.")

    # PDF 리포트 생성
    with st.spinner("PDF 리포트 생성 중..."):
        pdf_bytes = generate_pdf_report(user_input, analysis_type, result, rag_results)

    st.success("✅ 분석 완료! 아래에서 리포트를 다운로드하세요.")
    st.download_button(
        label="📥 PDF 리포트 다운로드",
        data=pdf_bytes,
        file_name="autoguard_security_report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )