"""
Streamlit chat UI for agentforge.
Run:  streamlit run app.py
"""
import logging
import tempfile
import os

import streamlit as st

from agentforge.main import run_agent
from agentforge.rag.document_store import ingest_file, load_corpus
from agentforge.logger import compute_cost_summary, compute_trace_cost, generate_trace_id

logger = logging.getLogger(__name__)

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(page_title="AgentForge Chat", page_icon="🤖", layout="centered")

# ── Session state defaults ───────────────────────────────────────────
if "user_id" not in st.session_state:
    st.session_state.user_id = "demo_user"
if "history" not in st.session_state:
    st.session_state.history = []          # [{role, content}, ...]
if "messages" not in st.session_state:
    st.session_state.messages = []         # [{role, content, trace_id?}, ...]

# ── Sidebar — settings, ingestion, corpus ─────────────────────────────
with st.sidebar:
    st.header("Settings")
    st.session_state.user_id = st.text_input("User ID", value=st.session_state.user_id)

    st.divider()

    # ── File ingestion ───────────────────────────────────────────────
    st.header("Ingest Document")
    uploaded_file = st.file_uploader(
        "Upload a .txt or .md file",
        type=["txt", "md"],
        help="The file will be chunked, embedded, and added to the corpus for RAG.",
    )
    doc_id_input = st.text_input("Document ID (optional)", placeholder="auto-derived from filename")

    if st.button("Ingest", disabled=uploaded_file is None):
        with st.spinner("Chunking & embedding..."):
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            try:
                custom_id = doc_id_input.strip() if doc_id_input.strip() else None
                n = ingest_file(tmp_path, doc_id=custom_id)
                st.success(f"Ingested **{uploaded_file.name}**: {n} chunk(s) added.")
                logger.info("Ingested file=%s chunks=%d doc_id=%s", uploaded_file.name, n, custom_id)
            finally:
                os.unlink(tmp_path)

    st.divider()

    # ── Corpus info ──────────────────────────────────────────────────
    corpus = load_corpus()
    st.metric("Corpus chunks", len(corpus))
    if corpus:
        sources = sorted({c.get("source", "?") for c in corpus})
        st.caption("Sources: " + ", ".join(sources))

# ── Main chat area ───────────────────────────────────────────────────
st.title("🤖 AgentForge Chat")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("trace_id"):
            trace_cost = compute_trace_cost(msg["trace_id"])
            if trace_cost["calls"] > 0:
                st.caption(
                    f"💰 {trace_cost['calls']} LLM call(s), "
                    f"${trace_cost['cost_usd']:.4f}, "
                    f"{trace_cost['prompt_tokens'] + trace_cost['completion_tokens']:,} tokens "
                    f"({', '.join(trace_cost['operations'])})"
                )

# ── User input ───────────────────────────────────────────────────────
if user_input := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    tid = generate_trace_id()

    with st.chat_message("assistant"):
        try:
            result = run_agent(
                user_id=st.session_state.user_id,
                session_id="streamlit",
                user_input=user_input,
                history=st.session_state.history,
                stream=True,
                trace_id=tid,
            )

            if isinstance(result, str):
                st.markdown(result)
            else:
                result = st.write_stream(result)
        except Exception as e:
            logger.error("run_agent failed: %s", e, exc_info=True)
            result = f"Something went wrong: {e}"
            st.markdown(result)

    st.session_state.history.append({"role": "user", "content": user_input})
    st.session_state.history.append({"role": "assistant", "content": result})
    st.session_state.messages.append({"role": "assistant", "content": result, "trace_id": tid})
    st.rerun()

# ── Sidebar — cost tracker (renders AFTER chat processing) ───────────
with st.sidebar:
    st.divider()
    st.header("Cost Tracker")
    cost_data = compute_cost_summary()
    total = cost_data.get("total", {})
    col1, col2 = st.columns(2)
    col1.metric("Total cost", f"${total.get('cost_usd', 0):.4f}")
    col2.metric("LLM calls", total.get("calls", 0))

    if total.get("prompt_tokens", 0) > 0:
        st.caption(
            f"Prompt: {total['prompt_tokens']:,} tokens | "
            f"Completion: {total['completion_tokens']:,} tokens"
        )

    by_op = cost_data.get("by_operation", {})
    if by_op:
        with st.expander("Cost by operation"):
            for op, stats in by_op.items():
                st.text(
                    f"{op}: {stats['calls']} calls, "
                    f"${stats['cost_usd']:.6f}, "
                    f"{stats['prompt_tokens'] + stats['completion_tokens']:,} tokens"
                )

    st.divider()
    if st.button("Clear chat"):
        st.session_state.history = []
        st.session_state.messages = []
        st.rerun()
