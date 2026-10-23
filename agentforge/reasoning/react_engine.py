# agent/reasoning/react_engine.py

import json
import logging
from dataclasses import dataclass
from typing import Any, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command
from openai import OpenAI

from agentforge.approval import decision_label, run_interruptible
from agentforge.config import (
    OPENAI_MODEL,
    OPENAI_BASE_URL,
    REACT_OBS_COMPRESS_THRESHOLD,
    REACT_TRACE_ARG_MAX_CHARS,
)
from agentforge.prompts import (
    build_prompt,
    render_tool_catalog,
    OBSERVATION_COMPRESSION_PROMPT,
    OUTPUT_SCHEMA,
    REACT_TOOL_EFFICIENCY,
    REACT_TOOL_ERROR_RECOVERY,
    SPOTLIGHT_INSTRUCTIONS,
    earlier_conversation_message,
)
from agentforge.hitl_graph import (
    CHECKPOINTER,
    GraphApprovalHandler,
    PendingApproval,
    apending_approval,
    new_thread_id,
    pending_approval,
    run_graph,
)
from agentforge.mcp_client import mcp_gateway
from agentforge.memory.semantic import get_relevant_memories, store_memory
from agentforge.output_guardrail import scan_structured
from agentforge.safety import wrap_untrusted
from agentforge.logger import log_event, log_token_usage
from agentforge.reply import reply_text

_client = None  # created on first API call, not at import time

def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(base_url=OPENAI_BASE_URL) if OPENAI_BASE_URL else OpenAI()
    return _client

logger = logging.getLogger(__name__)


def react_loop(user_id: str, user_input: str, max_steps: int = 5,
               approval_handler=None, trace_id: str = None,
               model: str = None, history: list = None) -> str:
    """Core ReAct reasoning loop (think → act → observe → repeat).

    Public entry point. Existing positional args are unchanged, so callers
    (main.py) are unaffected; internally it runs an async loop because tools are
    reached over MCP. One MCP gateway is opened for the whole turn (see
    ``_react_loop_async``). ``approval_handler`` is the front-end's
    human-in-the-loop callback (Step 17f), passed through to the gateway.
    ``trace_id`` is the turn's trace ID from run_agent (issue #7): without it,
    every log record this module emits — token usage, react_* events, the
    gateway's security audit entries — is orphaned from its turn, so per-trace
    cost aggregation undercounts REACT turns.
    ``model`` is the routed model for this turn (Step 28). None → OPENAI_MODEL,
    so every pre-routing caller is unaffected. REACT is the "hard" intent, so
    run_agent routes it to the frontier tier; the step-level reasoning calls use
    it. The observation-compressor is a routine sub-task and stays on the small
    model deliberately (see _maybe_compress_observation).
    ``history`` is the turn's (trimmed) chat history, given to the model as one
    wrapped data block so a follow-up like "and the news there?" can be
    resolved (issue #59). None → no block, as before.
    """
    return run_interruptible(_react_loop_async(user_id, user_input, max_steps,
                                               approval_handler=approval_handler,
                                               trace_id=trace_id, model=model,
                                               history=history))


def resume_react_loop(thread_id: str, decision,
                      approval_handler=None) -> str:
    """Resume a ReAct turn that was interrupted for human approval (Step 17f).

    ``thread_id`` is the paused graph's handle (``ApprovalRequired.thread_id``,
    Step 21.1). ``decision`` is the human's Deny (False) / Allow once (True) / APPROVE_TURN
    (allow this tool for the rest of the turn — issue #6) for the stored
    pending call. Resume — not replay — so the decision settles the EXACT call
    the human looked at: the graph re-runs only the ``act`` node, which reads
    the call from checkpointed state; the LLM is never asked to regenerate it.
    A Deny resumes too: the model receives the "user declined" observation and
    adapts.
    """
    return run_interruptible(
        _react_resume_async(thread_id, decision, approval_handler))


def pending_react_approval(thread_id: str) -> PendingApproval:
    """The paused ReAct turn ``thread_id``; ValueError if no longer pending."""
    return pending_approval(_graph(), thread_id, "react")


# ------------------------- the ReAct graph (Step 21.1) -------------------------
#
# The loop is a two-node LangGraph graph:
#
#     START -> think --(tool)--> act -> think ...
#                    --(reply)-> END
#
# The node boundary is the point. LangGraph checkpoints state after every node
# and, on resume, RE-RUNS the interrupted node from its first line. So:
#   - ``think`` holds the non-deterministic LLM call and never pauses. Its
#     output (the tool call) is checkpointed before ``act`` starts.
#   - ``act`` can pause (the approval gate inside gw.call). Everything before
#     the pause reads checkpointed state, so the re-run reaches the SAME
#     approval request — the human's decision binds to the stored call.
# Compression stays inside ``act``: it runs after the pause, so it runs once,
# and it never raises (it falls back to the raw observation).

_MAX_STEPS_REPLY = "Agent stopped: too many reasoning steps."


class _ReactState(TypedDict):
    """Checkpointed turn state. Plain data only — the checkpointer serializes
    it. Live objects (the gateway) travel in ``_ReactContext`` instead."""
    user_id: str
    messages: list
    step: int                 # think steps taken so far
    max_steps: int
    model: Optional[str]      # routed model (Step 28) — survives a resume
    trace_id: Optional[str]   # issue #7 — the resumed half logs on the same trace
    tool_pending: bool        # think proposed a tool call for act to run
    tool_name: Optional[str]
    tool_input: Any
    raw: Optional[str]        # the model's JSON for the proposing step
    granted: list             # APPROVE_TURN grants as [server, tool] pairs (issue #6)
    reply: Optional[str]      # set once -> the graph ends


@dataclass
class _ReactContext:
    """Per-invocation, never-checkpointed dependencies for the nodes."""
    gw: Any


_GRAPH = None

# Public names for callers that run this graph nested in their own graph (the
# supervisor's research worker, Step 21.2).
ReactContext = _ReactContext


def react_graph():
    """The compiled ReAct graph, for running it nested inside another graph."""
    return _graph()


def _graph():
    """The compiled ReAct graph (built once, on first use)."""
    global _GRAPH
    if _GRAPH is None:
        builder = StateGraph(_ReactState, context_schema=_ReactContext)
        builder.add_node("think", _think_node)
        builder.add_node("act", _act_node)
        builder.add_edge(START, "think")
        builder.add_conditional_edges("think", _route_after_think,
                                      ["act", "think", END])
        builder.add_edge("act", "think")
        _GRAPH = builder.compile(checkpointer=CHECKPOINTER)
    return _GRAPH


def _recursion_limit(max_steps: int) -> int:
    # Each step is up to two graph steps (think + act), plus the final think.
    # LangGraph's default recursion limit (25) would cut off max_steps > 12.
    return 2 * max_steps + 5


async def _react_loop_async(user_id: str, user_input: str, max_steps: int = 5,
                            approval_handler=None, trace_id: str = None,
                            model: str = None, history: list = None) -> str:
    """Async ReAct loop with tools served over MCP (Step 17c.2).

    Tools are discovered at runtime and rendered into the prompt
    (render_tool_catalog) — the agent has zero hardcoded tools. One MCP
    gateway is opened for the whole turn, around the graph, so every step
    reuses the same warm server sessions.

    Each step uses response_format={"type": "json_object"} (constrained decoding)
    so the API guarantees valid JSON on every step. The json.JSONDecodeError
    guard is kept as a last resort, and shape validation (action.get("type")
    etc.) stays too: valid JSON does not mean correct keys or values.
    """
    # No front-end handler -> pass None, so the gateway denies by default and
    # audits it as "no approval handler configured".
    handler = (GraphApprovalHandler(approval_handler)
               if approval_handler is not None else None)
    async with mcp_gateway(trace_id, approval_handler=handler) as gw:
        initial = build_react_input(user_id, user_input, gw, max_steps=max_steps,
                                    trace_id=trace_id, model=model, history=history)
        return await _run(initial, new_thread_id("react"), max_steps, gw, trace_id)


def build_react_input(user_id: str, user_input: str, gw, max_steps: int = 5,
                      trace_id: str = None, model: str = None,
                      history: list = None) -> dict:
    """The initial graph state for one ReAct turn, built inside an open gateway.

    Shared by ``react_loop`` and the supervisor's research worker (Step 21.2),
    which runs this graph nested inside its own (``react_graph()`` with
    ``ReactContext``). The prompt is built once, with the live tool catalog,
    so the model can only reference tools that are actually served this turn.
    The supervisor's workers pass no ``history``: their brief is self-contained.
    """
    memory_chunks = get_relevant_memories(user_id, user_input)
    earlier = earlier_conversation_message(history)
    log_event("react_start", {"user_input": user_input, "max_steps": max_steps,
                              "history_messages": len(history or [])},
              trace_id=trace_id)
    messages = build_prompt(user_input, memory_chunks)
    if earlier:
        messages.insert(len(messages) - 1, earlier)  # just before the user's message
    messages.append({"role": "system", "content": render_tool_catalog(gw.catalog)})
    messages.append({"role": "system", "content": REACT_TOOL_EFFICIENCY})
    messages.append({"role": "system", "content": REACT_TOOL_ERROR_RECOVERY})
    messages.append({"role": "system", "content": OUTPUT_SCHEMA})
    log_event("react_tools_discovered", {"tools": [t["name"] for t in gw.catalog]},
              trace_id=trace_id)
    return {
        "user_id": user_id, "messages": messages, "step": 0,
        "max_steps": max_steps, "model": model, "trace_id": trace_id,
        "tool_pending": False, "tool_name": None, "tool_input": None,
        "raw": None, "granted": [], "reply": None,
    }


def tools_called(messages: list) -> list:
    """Names of the tools a finished ReAct turn ran, in order, read from its
    observation messages (``Observation from tool '<name>':``)."""
    prefix = "Observation from tool '"
    names = []
    for m in messages:
        content = str(m.get("content", ""))
        if m.get("role") == "user" and content.startswith(prefix):
            names.append(content[len(prefix):].split("'", 1)[0])
    return names


async def _react_resume_async(thread_id: str, decision, fallback=None) -> str:
    """Resume a paused ReAct graph with the human's decision (Step 17f / 21.1).

    Opens a fresh gateway (the original's server subprocesses died when the
    interrupt unwound the turn) and resumes the checkpointed thread with
    ``Command(resume=decision)``. LangGraph re-runs ``act``; the gateway's
    approval gate gets the decision back from ``interrupt()``. Later gated
    calls go to ``fallback`` (the front-end's normal handler) and can pause
    again — each gets its own card.
    """
    pending = await apending_approval(_graph(), thread_id, "react")
    trace_id = pending.trace_id
    log_event("react_resume", {
        "thread_id": thread_id,
        "step": pending.values.get("step"),
        "tool": pending.request.tool,
        "decision": decision_label(decision),
    }, trace_id=trace_id)

    handler = GraphApprovalHandler(fallback, resuming=True)
    async with mcp_gateway(trace_id, approval_handler=handler) as gw:
        return await _run(Command(resume=decision), thread_id,
                          pending.values["max_steps"], gw, trace_id)


async def _run(graph_input, thread_id: str, max_steps: int, gw,
               trace_id: Optional[str]) -> str:
    return await run_graph(
        _graph(), graph_input, thread_id=thread_id,
        recursion_limit=_recursion_limit(max_steps),
        context=_ReactContext(gw=gw), pipeline="react",
        trace_id=trace_id, output_key="reply", progress_key="step")


def _user_question(messages: list) -> str:
    """The original user question in a ReAct message list.

    Observations are also appended as ``user`` messages (issue #1), so the
    question is the first user message that is NOT a tool observation. Works
    on fresh and resumed (checkpointed) message lists alike — no extra field
    needed in the graph state.
    """
    for m in messages:
        if m.get("role") == "user" and not str(m.get("content", "")).startswith(
                "Observation from tool"):
            return m["content"]
    return ""


def _maybe_compress_observation(observation: str, messages: list, gw,
                                trace_id: str = None) -> str:
    """Query-focused compression of an oversized observation (issue #8).

    Every observation appended to ``messages`` is re-sent on EVERY later step,
    so a raw 3-8k-char fetch chunk inflates the cost of each remaining step and
    buries the relevant facts mid-context. One extra LLM call here reads the
    chunk ONCE, extracts only what answers the user's question, and the loop
    carries that instead. Below the threshold the observation passes through
    untouched (zero added cost on the common path).

    Security: the chunk is untrusted data, and a hostile page that cannot
    inject the agent must not get to inject the summarizer instead — so the
    compressor receives the same SPOTLIGHT_INSTRUCTIONS, and its output (a
    derivative of untrusted text) is RE-WRAPPED with the turn's nonce before
    it enters the history. Compression failure falls back to the raw
    observation: a logging-adjacent optimization must never kill the turn.
    """
    if REACT_OBS_COMPRESS_THRESHOLD <= 0:
        return observation
    if len(observation) <= REACT_OBS_COMPRESS_THRESHOLD:
        return observation

    question = _user_question(messages)
    try:
        response = _get_client().chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SPOTLIGHT_INSTRUCTIONS},
                {"role": "system",
                 "content": OBSERVATION_COMPRESSION_PROMPT.format(question=question)},
                {"role": "user", "content": observation},
            ],
        )
        log_token_usage(response, "react_obs_compress", trace_id=trace_id)
        summary = (response.choices[0].message.content or "").strip()
    except Exception as e:
        logger.warning("react: observation compression failed, keeping raw "
                       "observation: %s", e)
        log_event("react_obs_compress_failed", {"error": str(e)},
                  trace_id=trace_id)
        return observation

    if not summary:
        return observation
    log_event("react_obs_compressed", {
        "original_len": len(observation),
        "summary_len": len(summary),
    }, trace_id=trace_id)
    return wrap_untrusted(summary, source="compressed tool output",
                          nonce=gw.nonce)


def _trace_tool_input(tool_input, trace_id: str = None) -> dict:
    """The `react_step` fields describing a tool step's ARGUMENTS (Step 21b.1).

    WHY: before this, a trace recorded WHICH tool ran but not WHAT it was called
    with. That is enough to score tool *selection* and no more — a trajectory
    could not tell "right tool, wrong argument" from "right tool, bad source"
    (exactly the ambiguity in issue #36). Logging the arguments makes the trace
    complete enough to diagnose tool-call QUALITY, which is what multi-agent
    failure attribution (Step 21b) needs.

    The arguments are recorded at FULL VALUE, guardrail-scrubbed — deliberately
    different from the gateway's security audit log, which records key names
    only. Different logs, different jobs: the audit log answers "what was
    attempted" for a reviewer and must minimise what it retains; the trace is a
    debugging/eval artifact where the value IS the evidence. Both are local,
    gitignored files. The scrub is the regex floor (see ``scan_structured``), so
    a per-step model forward pass is never added to the loop.

    Never raises: a trace field must not be able to kill a turn. If the scan
    fails it degrades to the audit log's key-names-only shape and flags itself,
    which is strictly less information but never a wrong value.
    """
    if REACT_TRACE_ARG_MAX_CHARS <= 0:  # argument logging switched off
        return {}

    args = {} if tool_input is None else tool_input
    try:
        scanned = scan_structured(args, max_chars=REACT_TRACE_ARG_MAX_CHARS)
    except Exception as e:
        logger.warning("react: tool-argument scrub failed, logging key names "
                       "only: %s", e)
        log_event("react_trace_args_failed", {"error": str(e)}, trace_id=trace_id)
        return {
            "tool_input_keys": sorted(args.keys()) if isinstance(args, dict) else [],
            "tool_input_scan_failed": True,
        }

    fields = {"tool_input": scanned.value}
    # Categories only, never values — same rule as the reply guardrail. Present
    # only when something was redacted, so a clean step stays visually clean.
    if scanned.types:
        fields["tool_input_redacted"] = scanned.types
    if scanned.truncated:
        fields["tool_input_truncated"] = True
    return fields


async def _think_node(state: _ReactState) -> dict:
    """One reasoning step: the LLM proposes the next action.

    Holds the turn's non-deterministic work and NEVER pauses, so a resume
    never re-runs it — its output is checkpointed before ``act`` starts.
    Ends the turn (sets ``reply``) on a final answer, an error, or when the
    step budget is spent. ``model`` (Step 28) is the routed model for the
    per-step reasoning calls; None -> OPENAI_MODEL.
    """
    step = state["step"]
    trace_id = state["trace_id"]
    if step >= state["max_steps"]:
        log_event("react_end", {"steps_taken": state["max_steps"], "reply_length": 0,
                                "stopped": "max_steps"}, trace_id=trace_id)
        return {"reply": _MAX_STEPS_REPLY}

    try:
        response = _get_client().chat.completions.create(
            model=state["model"] or OPENAI_MODEL,
            messages=state["messages"],
            # Constrained decoding: same as classify_intent (Step 6).
            # The ReAct schema is more complex (nested action object,
            # multiple fields) so parse failures are more likely without
            # this — and more costly, since they abort a reasoning chain.
            response_format={"type": "json_object"},
        )
        log_token_usage(response, f"react_step_{step + 1}", trace_id=trace_id)
        raw = response.choices[0].message.content
    except Exception as e:
        logger.error("react_loop: LLM call failed on step %d: %s", step + 1, e, exc_info=True)
        return {"reply": "I ran into an error during planning. Please try again."}

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Should be extremely rare now that response_format is set.
        # Kept as a safety net for empty or malformed responses.
        logger.warning("react_loop: JSONDecodeError on step %d despite response_format.", step + 1)
        return {"reply": "Agent error: Invalid JSON response."}

    thought = data.get("thought", "")
    action = data.get("action", {})

    print(f"\n[THOUGHT {step+1}] {thought}")
    step_payload = {
        "step": step + 1,
        "thought": thought,
        "action_type": action.get("type"),
        "tool_name": action.get("tool_name"),
    }
    if action.get("type") == "tool":
        step_payload.update(_trace_tool_input(action.get("tool_input"), trace_id))
    log_event("react_step", step_payload, trace_id=trace_id)

    # ---------- FINAL ----------
    if action.get("type") == "final":
        if data.get("store_memory") and data.get("memory_text"):
            store_memory(state["user_id"], data["memory_text"])
        # Always a string (issue #49): an object reply becomes text, and a null
        # reply becomes "" so it still ENDs the graph (None means "not set").
        reply = reply_text(data.get("reply"), source="react", trace_id=trace_id)
        log_event("react_end", {"steps_taken": step + 1, "reply_length": len(reply)},
                  trace_id=trace_id)
        return {"step": step + 1, "reply": reply}

    # ---------- TOOL ----------
    if action.get("type") == "tool":
        return {"step": step + 1, "tool_pending": True,
                "tool_name": action.get("tool_name"),
                "tool_input": action.get("tool_input", {}),
                "raw": raw}

    # Unrecognised action type: spend the step and think again (unchanged
    # from the pre-graph loop).
    return {"step": step + 1}


async def _act_node(state: _ReactState, runtime: Runtime[_ReactContext]) -> dict:
    """Run the tool call ``think`` proposed, through the MCP gateway.

    This node can PAUSE: the gateway's approval gate may call ``interrupt()``.
    On resume LangGraph re-runs it from the top, so everything before
    ``gw.call`` reads checkpointed state only — the re-run reaches the same
    approval request, and the decision settles exactly that call.
    """
    gw = runtime.context.gw
    trace_id = state["trace_id"]
    tool_name = state["tool_name"]
    # Turn-scoped grants (issue #6) live in checkpointed state; the gateway
    # is rebuilt on resume, so seed it here and write it back below.
    gw.granted.update(tuple(g) for g in state["granted"])

    # Dispatch over MCP. gw.call returns a readable error string for
    # unknown/hallucinated tools (it does not raise), so the model can
    # observe the error and recover on the next step. gw.call also
    # wraps the tool output as untrusted data with this turn's nonce
    # (Step 17e) — the spotlight rule in the prompt tells the model to
    # treat anything inside those markers as data, never instructions.
    observation = await gw.call(tool_name, state["tool_input"])

    # The ReAct loop speaks the prompt-based JSON protocol: the model
    # returns {thought, action} as assistant *content*, not native
    # tool_calls. So the observation must go back as an ordinary
    # conversation turn — a user message — NOT a role:"tool" message.
    # The OpenAI API only accepts a "tool" message immediately after an
    # assistant message that carried tool_calls (the native
    # function-calling path); feeding an orphan "tool" message here
    # makes the next request fail with a 400. (Issue #1.)
    #
    # `observation` is already wrapped as <untrusted_data_<nonce>> by
    # gw.call, so spotlighting (Step 17e) still holds inside the user
    # turn — the spotlight rules tell the model to treat anything
    # between those markers as data, never instructions.
    messages = [*state["messages"], {"role": "assistant", "content": state["raw"]}]

    # Oversized observations are compressed relative to the user's
    # question before entering the history (issue #8) — otherwise
    # every later step re-sends the raw chunk.
    observation = _maybe_compress_observation(observation, messages, gw,
                                              trace_id=trace_id)
    messages.append({
        "role": "user",
        "content": f"Observation from tool '{tool_name}':\n{observation}"
    })
    return {"messages": messages,
            "granted": sorted([list(g) for g in gw.granted]),
            "tool_pending": False, "tool_name": None, "tool_input": None,
            "raw": None}


def _route_after_think(state: _ReactState) -> str:
    """Pure routing: a reply ends the turn; a proposed tool call goes to act."""
    if state["reply"] is not None:
        return END
    if state["tool_pending"]:
        return "act"
    return "think"
