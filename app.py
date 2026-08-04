"""Streamlit frontend for the Labor Law RAG chatbot."""
from __future__ import annotations

import time
from typing import Callable

import streamlit as st

st.set_page_config(
    page_title="Luật Lao Động AI | Genius RAG",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink:#152238; --muted:#667085; --line:#e4e9f1; --blue:#2563eb; --soft:#eff6ff; --gold:#f59e0b; }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; color:var(--ink); }
    .stApp { background:#f7f9fc; } .block-container { max-width:1380px; padding-top:2rem; }
    h1,h2,h3 { font-family:'Space Grotesk',sans-serif; letter-spacing:-.03em; }
    .hero { background:linear-gradient(120deg,#10213e,#1d4d8e 64%,#2563eb); border-radius:22px; padding:2rem 2.3rem; color:white; margin-bottom:1.3rem; box-shadow:0 14px 32px #1b3f7a25; }
    .hero h1 { margin:.35rem 0; font-size:2.2rem; } .hero p { margin:0; color:#dceaff; }
    .pill { display:inline-block; font-size:.75rem; border:1px solid #ffffff3b; border-radius:999px; padding:.28rem .7rem; background:#ffffff18; }
    .metric, .source-card { background:white; border:1px solid var(--line); border-radius:15px; padding:1rem 1.1rem; }
    .metric small, .muted { color:var(--muted); } .metric b { display:block; font-size:1.35rem; margin-top:.15rem; }
    .answer { background:#fff; border:1px solid var(--line); border-left:4px solid var(--blue); border-radius:14px; padding:1.1rem 1.25rem; line-height:1.7; }
    .source-card { margin:.55rem 0; } .tag { color:#1d4ed8; background:var(--soft); padding:.2rem .48rem; border-radius:6px; font-size:.73rem; }
    .legal-note { color:#7c5200; background:#fff9e8; border:1px solid #fde8b0; border-radius:11px; padding:.7rem .85rem; font-size:.87rem; }
    .stButton > button { border-radius:10px; font-weight:600; } .stTabs [data-baseweb="tab"] { font-weight:600; }
    </style>
    """,
    unsafe_allow_html=True,
)

SUGGESTIONS = [
    "Thời gian thử việc tối đa cho vị trí lập trình viên là bao lâu và lương thử việc tối thiểu bằng bao nhiêu % lương chính thức?",
    "Công ty sa thải tôi qua tin nhắn Zalo mà không báo trước 30 ngày thì có đúng luật không?",
    "Người lao động được làm thêm tối đa bao nhiêu giờ?",
    "Tôi được nghỉ phép năm bao nhiêu ngày?",
]


def source_title(source: dict, index: int) -> str:
    meta = source.get("metadata") or {}
    return str(meta.get("title") or meta.get("source") or meta.get("source_file") or f"Nguồn {index}")


def source_kind(source: dict) -> str:
    meta = source.get("metadata") or {}
    return str(source.get("source") or meta.get("type") or meta.get("corpus_type") or "retrieval")


def render_sources(sources: list[dict]) -> None:
    if not sources:
        st.caption("Không có evidence trả về từ retrieval.")
        return
    for index, source in enumerate(sources, start=1):
        meta = source.get("metadata") or {}
        score = float(source.get("score", 0.0))
        st.markdown(
            f"<div class='source-card'><b>{index:02d} · {source_title(source, index)}</b> "
            f"<span class='tag'>{source_kind(source)}</span><br>"
            f"<span class='muted'>relevance: <b>{score:.4f}</b>"
            f" · chunk {meta.get('chunk_index', '—')}</span></div>",
            unsafe_allow_html=True,
        )
        with st.expander(f"Xem nội dung nguồn {index}"):
            st.write(source.get("content", ""))


def run_comparison(query: str, top_k: int) -> dict[str, tuple[list[dict], float, str | None]]:
    from src.task5_semantic_search import semantic_search
    from src.task6_lexical_search import lexical_search
    from src.task8_pageindex_vectorless import pageindex_search
    from src.task9_retrieval_pipeline import retrieve

    methods: list[tuple[str, Callable[[str, int], list[dict]]]] = [
        ("Semantic", semantic_search),
        ("BM25", lexical_search),
        ("Hybrid RRF", retrieve),
        ("PageIndex", pageindex_search),
    ]
    output = {}
    for label, search in methods:
        started = time.perf_counter()
        try:
            output[label] = (search(query, top_k=top_k), time.perf_counter() - started, None)
        except Exception as exc:
            output[label] = ([], time.perf_counter() - started, str(exc))
    return output


if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

with st.sidebar:
    st.markdown("## ⚖️ Luật Lao Động AI")
    st.caption("Trợ lý RAG cho người trẻ tại Việt Nam")
    st.divider()
    st.markdown("**Câu hỏi gợi ý**")
    for index, suggestion in enumerate(SUGGESTIONS):
        if st.button(suggestion, key=f"suggestion_{index}", use_container_width=True):
            st.session_state.pending_query = suggestion
    st.divider()
    st.markdown("**Thiết lập retrieval**")
    top_k = st.slider("Số nguồn đưa vào context", min_value=3, max_value=8, value=5)
    show_debug = st.toggle("Hiển thị thông tin kỹ thuật", value=False)
    st.divider()
    st.caption("Semantic + BM25 → RRF → PageIndex fallback → LLM có citation")

st.markdown(
    """<div class='hero'><span class='pill'>GROUNDED LEGAL RAG · VIETNAM</span>
    <h1>Hiểu luật lao động, bảo vệ quyền lợi của bạn.</h1>
    <p>Tra cứu quy định về thử việc, hợp đồng, làm thêm, nghỉ phép và chấm dứt việc làm với nguồn trích dẫn minh bạch.</p></div>""",
    unsafe_allow_html=True,
)

chat_tab, compare_tab = st.tabs(["💬 Hỏi trợ lý", "⚖️ So sánh search"])

with chat_tab:
    metrics = st.columns(3)
    with metrics[0]:
        st.markdown("<div class='metric'><small>Chiến lược</small><b>Hybrid RRF</b></div>", unsafe_allow_html=True)
    with metrics[1]:
        st.markdown("<div class='metric'><small>Fallback</small><b>PageIndex</b></div>", unsafe_allow_html=True)
    with metrics[2]:
        st.markdown("<div class='metric'><small>Grounding</small><b>Citation required</b></div>", unsafe_allow_html=True)
    st.write("")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander(f"Nguồn tham khảo ({len(message['sources'])})"):
                    render_sources(message["sources"])

    user_input = st.chat_input("Hỏi về thử việc, OT, nghỉ phép, hợp đồng hoặc sa thải...")
    query = user_input or st.session_state.pending_query
    if query:
        st.session_state.pending_query = None
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        with st.chat_message("assistant"):
            with st.spinner("Đang truy xuất điều luật và tổng hợp câu trả lời có trích dẫn..."):
                try:
                    from src.task10_generation import generate_with_citation
                    result = generate_with_citation(query, top_k=top_k)
                    answer = result["answer"]
                    sources = result.get("sources", [])
                    retrieval_source = result.get("retrieval_source", "none")
                except Exception as exc:
                    answer = (
                        "⚠️ **Chưa thể tạo câu trả lời.** "
                        f"Pipeline trả về lỗi: `{exc}`\n\n"
                        "Kiểm tra lại môi trường, vector index và API key trong `.env`."
                    )
                    sources, retrieval_source = [], "error"
            st.markdown(answer)
            if sources:
                st.markdown(f"<div class='legal-note'>Nguồn retrieval: <b>{retrieval_source}</b>. Hãy đọc nội dung trích dẫn và tham khảo chuyên gia pháp lý khi cần.</div>", unsafe_allow_html=True)
                with st.expander(f"Nguồn tham khảo ({len(sources)})"):
                    render_sources(sources)
            if show_debug:
                st.caption(f"retrieval_source={retrieval_source}; top_k={top_k}")
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})

with compare_tab:
    st.markdown("### So sánh phương pháp tìm kiếm")
    st.caption("Chạy cùng một câu hỏi qua Semantic, BM25, Hybrid RRF và PageIndex để minh hoạ lợi ích của hybrid retrieval.")
    compare_query = st.text_area("Câu hỏi để so sánh", value=SUGGESTIONS[0], height=82)
    if st.button("Chạy so sánh retrieval", use_container_width=False):
        if not compare_query.strip():
            st.warning("Hãy nhập câu hỏi trước khi chạy so sánh.")
        else:
            with st.spinner("Đang chạy bốn retriever trên cùng một câu hỏi..."):
                comparison = run_comparison(compare_query.strip(), min(top_k, 5))
            columns = st.columns(4)
            for column, label in zip(columns, comparison):
                rows, elapsed, error = comparison[label]
                with column:
                    st.markdown(f"#### {label}")
                    st.metric("Latency", f"{elapsed:.2f}s")
                    if error:
                        st.error(error)
                    elif not rows:
                        st.caption("Không tìm thấy kết quả.")
                    else:
                        for rank, row in enumerate(rows[:3], start=1):
                            st.markdown(
                                f"<div class='source-card'><b>#{rank} {source_title(row, rank)}</b><br>"
                                f"<span class='muted'>{row.get('content', '')[:150]}…</span><br>"
                                f"<span class='tag'>{float(row.get('score', 0)):.4f}</span></div>",
                                unsafe_allow_html=True,
                            )
            st.info("RRF score chỉ dùng để xếp hạng sau fusion; không phải xác suất đúng. Pipeline dùng điểm cosine gốc để quyết định PageIndex fallback.")
    else:
        st.info("Bấm **Chạy so sánh retrieval** để lấy kết quả thật từ index hiện tại.")
