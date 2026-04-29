"""Local RAG — Streamlit frontend.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import yaml

sys.path.insert(0, str(Path(__file__).parent))

# ---------------------------------------------------------------------------
# Page config (must be the very first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Local RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Light theme CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #f8f9fb;
        color: #1a1a2e;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e4e7ed;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #1a1a2e; }
    [data-testid="block-container"] { padding-top: 1.5rem; }

    /* Chat bubbles */
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        margin-bottom: 8px;
        padding: 4px 8px;
    }

    /* Source cards */
    .source-card {
        background: #ffffff;
        border: 1px solid #e4e7ed;
        border-left: 4px solid #6366f1;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 0.85rem;
    }
    .source-card .doc-name {
        font-weight: 600;
        color: #4f46e5;
        margin-bottom: 4px;
    }
    .source-card .snippet {
        color: #4b5563;
        font-style: italic;
        border-left: 2px solid #d1d5db;
        padding-left: 8px;
        margin-top: 6px;
        font-size: 0.82rem;
    }
    .source-card .score-badge {
        display: inline-block;
        background: #eef2ff;
        color: #4f46e5;
        border-radius: 20px;
        padding: 1px 8px;
        font-size: 0.75rem;
        float: right;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        border: 1px solid #6366f1;
        color: #4f46e5;
        background: #ffffff;
        font-weight: 500;
    }
    .stButton > button:hover {
        background: #eef2ff;
        border-color: #4f46e5;
    }

    /* Status pills */
    .status-pill { display: inline-block; border-radius: 20px; padding: 2px 10px;
                   font-size: 0.78rem; font-weight: 600; }
    .status-ok   { background:#dcfce7; color:#166534; }
    .status-warn { background:#fef9c3; color:#854d0e; }
    .status-err  { background:#fee2e2; color:#991b1b; }

    hr { border-color: #e4e7ed; }
    .stTextInput > div > div > input { border-radius: 8px; border: 1px solid #d1d5db; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
CONFIG_PATH = "config/config.yaml"


@st.cache_data
def get_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@st.cache_resource(show_spinner="Loading RAG pipeline…")
def get_pipeline():
    from src.rag.factory import build_pipeline
    return build_pipeline(CONFIG_PATH)


def get_chunker():
    from src.ingestion.parent_chunker import ParentChildChunker
    c = get_config()["chunking"]
    return ParentChildChunker(
        child_chunk_size=c["child_chunk_size"],
        child_chunk_overlap=c["child_chunk_overlap"],
        parent_chunk_size=c["parent_chunk_size"],
        parent_chunk_overlap=c["parent_chunk_overlap"],
        min_chunk_size=c["min_chunk_size"],
        strategy=c["strategy"],
        semantic_threshold=c["semantic_threshold"],
    )


def get_converter():
    from src.ingestion.pdf_converter import PDFConverter
    cfg = get_config()
    return PDFConverter(
        raw_dir=cfg["ingestion"]["raw_dir"],
        processed_dir=cfg["ingestion"]["processed_dir"],
    )


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []   # [{role, content, sources}]

# ---------------------------------------------------------------------------
# Helper: render source reference cards
# ---------------------------------------------------------------------------
def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"📎 {len(sources)} source section(s) retrieved", expanded=False):
        for hit in sources:
            doc_name = hit.get("source", "unknown")
            score = hit.get("score", 0)
            raw_snippet = hit.get("snippet", "")
            snippet = raw_snippet[:320] + ("…" if len(raw_snippet) > 320 else "")
            st.markdown(
                f"""<div class="source-card">
                    <span class="score-badge">{score:.0%} match</span>
                    <div class="doc-name">📄 {doc_name}</div>
                    <div class="snippet">{snippet}</div>
                </div>""",
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📚 Local RAG")
    st.markdown("---")

    cfg = get_config()
    llm_model = cfg["ollama"]["llm_model"]
    embed_model = cfg["ollama"]["embed_model"]

    # ── Ollama status ──────────────────────────────────────────────────────
    st.markdown("### System status")
    try:
        import ollama as _ollama
        _client = _ollama.Client(host=cfg["ollama"]["base_url"])
        available = [m["name"].split(":")[0] for m in _client.list().get("models", [])]
        llm_ok = llm_model.split(":")[0] in available
        emb_ok = embed_model.split(":")[0] in available

        col1, col2 = st.columns(2)
        with col1:
            cls = "status-ok" if llm_ok else "status-warn"
            st.markdown(f'<span class="status-pill {cls}">LLM {"✓" if llm_ok else "✗"}</span>', unsafe_allow_html=True)
            st.caption(llm_model)
        with col2:
            cls = "status-ok" if emb_ok else "status-warn"
            st.markdown(f'<span class="status-pill {cls}">Embed {"✓" if emb_ok else "✗"}</span>', unsafe_allow_html=True)
            st.caption(embed_model)

        if not llm_ok:
            st.warning(f"Run: `ollama pull {llm_model}`")
        if not emb_ok:
            st.warning(f"Run: `ollama pull {embed_model}`")
    except Exception as e:
        st.markdown('<span class="status-pill status-err">Ollama offline</span>', unsafe_allow_html=True)
        st.caption(str(e))

    st.markdown("---")

    # ── Upload ─────────────────────────────────────────────────────────────
    st.markdown("### Upload documents")
    uploaded = st.file_uploader(
        "Drop PDF or Markdown files",
        type=["pdf", "md", "markdown", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded:
        if st.button("⬆️  Ingest selected files", use_container_width=True):
            pipeline = get_pipeline()
            chunker = get_chunker()
            converter = get_converter()
            raw_dir = Path(cfg["ingestion"]["raw_dir"])
            raw_dir.mkdir(parents=True, exist_ok=True)

            progress = st.progress(0, text="Starting…")
            results: list[tuple] = []

            for i, uf in enumerate(uploaded):
                progress.progress(i / len(uploaded), text=f"Processing {uf.name}…")
                dest = raw_dir / uf.name
                dest.write_bytes(uf.getvalue())

                try:
                    from scripts.ingest import ingest_file
                    n = ingest_file(dest, chunker, pipeline.vector_store, converter)
                    results.append((uf.name, n, None))
                except Exception as exc:
                    results.append((uf.name, 0, str(exc)))

            progress.progress(1.0, text="Done!")
            for name, n, err in results:
                if err:
                    st.error(f"❌ {name}: {err}")
                else:
                    st.success(f"✅ {name} — {n} chunks ingested")
            st.rerun()

    st.markdown("---")

    # ── Knowledge base list ────────────────────────────────────────────────
    st.markdown("### Knowledge base")
    try:
        pipeline = get_pipeline()
        sources = pipeline.vector_store.list_sources()
        n_children = pipeline.vector_store.count_children()
        n_parents = pipeline.vector_store.count_parents()

        st.caption(f"{len(sources)} documents · {n_parents} sections · {n_children} chunks")

        for src in sources:
            col_a, col_b = st.columns([5, 1])
            with col_a:
                st.markdown(f"📄 `{src}`")
            with col_b:
                if st.button("✕", key=f"del_{src}", help=f"Remove {src}"):
                    pipeline.vector_store.delete_source(src)
                    st.success(f"Removed {src}")
                    st.rerun()

        if not sources:
            st.info("No documents yet — upload files above.")
    except Exception as exc:
        st.error(f"DB error: {exc}")

    st.markdown("---")

    # ── Query settings ─────────────────────────────────────────────────────
    with st.expander("⚙️ Query settings"):
        top_k = st.slider("Sections to retrieve", 1, 10, cfg["vector_store"]["top_k"])
        mmr_lambda = st.slider(
            "MMR: 0 = diverse, 1 = relevant",
            0.0, 1.0, float(cfg["vector_store"]["mmr_lambda"]), step=0.05,
        )
        use_mmr = st.checkbox("Use MMR retrieval", value=True)
        do_stream = st.checkbox("Stream response", value=True)

    if st.button("🗑️  Clear conversation", use_container_width=True):
        st.session_state.messages = []
        try:
            get_pipeline().memory._short_term.clear()
        except Exception:
            pass
        st.rerun()


# ---------------------------------------------------------------------------
# Main — chat interface
# ---------------------------------------------------------------------------
st.markdown("## 💬 Chat with your documents")
st.markdown("---")

# Replay history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents…"):
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            pipeline = get_pipeline()
            pipeline.top_k = top_k
            pipeline.mmr_lambda = mmr_lambda
            pipeline.use_mmr = use_mmr

            if do_stream:
                placeholder = st.empty()
                full_response = ""
                for token in pipeline.query(prompt, stream=True):
                    full_response += token
                    placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)
            else:
                with st.spinner("Thinking…"):
                    full_response = pipeline.query(prompt, stream=False)
                st.markdown(full_response)

            sources = getattr(pipeline, "last_sources", [])
            render_sources(sources)

            st.session_state.messages.append(
                {"role": "assistant", "content": full_response, "sources": sources}
            )

        except Exception as exc:
            err_msg = f"Error: {exc}"
            st.error(err_msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": err_msg, "sources": []}
            )
