"""
Streamlit chat UI for agentforge.
Run:  streamlit run app.py
"""
import json
import logging
import tempfile
import os

import streamlit as st

from agentforge.approval import ApprovalRequest, ApprovalRequired
from agentforge.main import resume_agent, run_agent
from agentforge.rag.document_store import ingest_file, load_corpus
from agentforge.logger import compute_cost_summary, compute_trace_cost, generate_trace_id
from agentforge.tools import prime_tool_catalog

logger = logging.getLogger(__name__)


# ── Human-in-the-loop approval (Step 17f): interrupt → checkpoint → resume ──
# Streamlit cannot pause mid-run to ask a question — the script runs
# top-to-bottom and ends; a button click re-runs it from line 1. So the
# approval handler can't block like the CLI's y/n prompt. Instead:
#   1. INTERRUPT — the handler always raises ApprovalRequired. As it unwinds,
#      the pipeline loop attaches a *continuation* (the pending call + the
#      frozen loop state) to the exception.
#   2. CHECKPOINT — the except-block below stashes that exception in session
#      state; the next run renders the Allow/Deny card.
#   3. RESUME — the click calls main.resume_agent(interrupt, decision), which
#      re-enters the loop mid-flight and settles the EXACT stored call with
#      the human's decision. The LLM is never asked to regenerate the call —
#      regeneration is non-deterministic and was the flaw in the replay
#      design this replaced (the model's argument values drifted between
#      replays, so a recorded "yes" stopped matching and the user was asked
#      twice for the same thing). Deny resumes too: the model receives the
#      "user declined" observation and adapts.
# This is hand-rolled LangGraph interrupt() + checkpointer; the production
# version arrives with the Step 21 multi-agent track.
def _streamlit_approval_handler(request: ApprovalRequest) -> bool:
    raise ApprovalRequired(request)

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(page_title="AgentForge Chat", page_icon="🤖", layout="centered")

# ── Discover MCP tools once per session (Step 17c.1) ─────────────────
# Streamlit reruns this script top-to-bottom on every interaction, so guard the
# (subprocess-spawning) discovery behind a session flag — prime only on first load.
if "_mcp_primed" not in st.session_state:
    prime_tool_catalog()
    st.session_state._mcp_primed = True

# ── Session state defaults ───────────────────────────────────────────
if "user_id" not in st.session_state:
    st.session_state.user_id = "demo_user"
if "history" not in st.session_state:
    st.session_state.history = []          # [{role, content}, ...]
if "messages" not in st.session_state:
    st.session_state.messages = []         # [{role, content, trace_id?}, ...]
# Step 17f interrupt/resume state: the turn in flight (survives the rerun an
# interrupt causes), the caught interrupt (carries the pending call AND the
# frozen loop state), and the human's not-yet-consumed decision.
if "pending_input" not in st.session_state:
    st.session_state.pending_input = None    # user input whose turn is in flight
if "pending_tid" not in st.session_state:
    st.session_state.pending_tid = None      # trace id kept stable across the turn
if "pending_approval" not in st.session_state:
    st.session_state.pending_approval = None  # the ApprovalRequired exception
if "resume_decision" not in st.session_state:
    st.session_state.resume_decision = None   # True/False once a button is clicked

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

# ── Approval card (Step 17f) ─────────────────────────────────────────
# Rendered when a turn was interrupted by ApprovalRequired and the human has
# not decided yet. The arguments are shown in full — an informed decision needs
# the exact URL/args the tool would receive. (Shown on screen only; the audit
# log records arg names, never values.)
if st.session_state.pending_approval is not None and st.session_state.resume_decision is None:
    req = st.session_state.pending_approval.request
    with st.chat_message("assistant"):
        st.warning("The agent wants to call a tool that requires your approval.")
        st.code(
            json.dumps(
                {"tool": req.tool, "server": req.server, "arguments": req.arguments},
                indent=2, ensure_ascii=False,
            ),
            language="json",
        )
        allow_col, deny_col = st.columns(2)
        # Either click records the decision and reruns — the processing block
        # below then RESUMES the interrupted turn, settling this exact stored
        # call with the decision. Deny also resumes: the model receives a
        # readable "user declined" observation and can adapt (same semantics as
        # the CLI's 'n'), instead of the turn silently dying.
        if allow_col.button("✅ Allow", use_container_width=True):
            st.session_state.resume_decision = True
            st.rerun()
        if deny_col.button("🚫 Deny", use_container_width=True):
            st.session_state.resume_decision = False
            st.rerun()

# ── User input ───────────────────────────────────────────────────────
# Disabled while a decision is pending — one turn in flight at a time.
if user_input := st.chat_input(
    "Ask me anything...", disabled=st.session_state.pending_approval is not None
):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.pending_input = user_input
    st.session_state.pending_tid = generate_trace_id()
    # Fall through — the processing block below runs in this same script run.

# ── Turn processing ──────────────────────────────────────────────────
# Two ways in: a fresh input (run_agent from the top), or an Allow/Deny click
# (resume_agent re-enters the interrupted loop mid-flight — no re-classify, no
# re-run of earlier steps, the stored call is settled with the decision).
ready_to_resume = (st.session_state.pending_approval is not None
                   and st.session_state.resume_decision is not None)
if st.session_state.pending_input is not None and (
        st.session_state.pending_approval is None or ready_to_resume):
    turn_input = st.session_state.pending_input
    tid = st.session_state.pending_tid
    interrupted = False

    with st.chat_message("assistant"):
        try:
            if ready_to_resume:
                interrupt = st.session_state.pending_approval
                decision = st.session_state.resume_decision
                st.session_state.pending_approval = None
                st.session_state.resume_decision = None
                result = resume_agent(interrupt, decision,
                                      approval_handler=_streamlit_approval_handler)
                st.markdown(result)
            else:
                result = run_agent(
                    user_id=st.session_state.user_id,
                    session_id="streamlit",
                    user_input=turn_input,
                    history=st.session_state.history,
                    stream=True,
                    trace_id=tid,
                    approval_handler=_streamlit_approval_handler,
                )
                if isinstance(result, str):
                    st.markdown(result)
                else:
                    result = st.write_stream(result)
        except ApprovalRequired as exc:
            # INTERRUPT: stash the exception (request + continuation); the next
            # run renders the card. A resumed turn can interrupt again on a NEW
            # gated call — that lands here too, with its own continuation.
            logger.info("Turn interrupted for approval: tool=%s server=%s",
                        exc.request.tool, exc.request.server)
            interrupted = True
            st.session_state.pending_approval = exc
            st.session_state.resume_decision = None
        except Exception as e:
            logger.error("run_agent failed: %s", e, exc_info=True)
            result = f"Something went wrong: {e}"
            st.markdown(result)

    if interrupted:
        st.rerun()

    # Turn complete — commit to history and clear the in-flight state.
    st.session_state.history.append({"role": "user", "content": turn_input})
    st.session_state.history.append({"role": "assistant", "content": result})
    st.session_state.messages.append({"role": "assistant", "content": result, "trace_id": tid})
    st.session_state.pending_input = None
    st.session_state.pending_tid = None
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
