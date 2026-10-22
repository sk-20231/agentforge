"""Supervisor / planner-worker orchestration (Step 21.2).

When ``AGENT_SUPERVISOR_ENABLED`` is on, a planner replaces the intent
classifier. The classifier was already a one-shot supervisor: it picked ONE
pipeline and never saw the result, so a request with two parts ("what does my
handbook say about X, and what is the latest news about Y?") lost a part.

    plan_turn  ── one LLM call, BEFORE any graph. Each worker is a tool; the
                  parallel tool calls in its reply are the plan (the
                  "supervisor via tools" pattern).
      no task   → the planner's own short reply (small talk)
      one task  → today's pipeline for that worker, run by main.py unchanged
                  (original message, history, streaming, react-… resume)
      2+ tasks  → the supervisor graph below:

    START → run_worker ⟲ (one worker per run) → combine → END

Workers are the existing pipelines behind a thin wrapper that returns one
result shape — ``{worker, planned_worker, brief, status, summary,
tools_called, tokens_used, error, trust}``:

    remember             → semantic.store_memory            (trusted)
    research             → the REACT graph, nested here      (untrusted if a tool ran)
    read_user_documents  → rag.qa.answer_from_docs           (untrusted — outside text)
    answer               → one task: memory.response.answer_with_memory (via main.py)
                           2+ tasks: runs as research (see below)

There is ONE tool worker, backed by REACT, because the planner measurably
cannot tell dependent parts from independent ones (0/10 on gpt-4o-mini,
2026-10-17). REACT looks again after every tool result, so a wrongly split
chain still finishes; the planner never has to make that call.

In a plan with two or more tasks, an ``answer`` task ALSO runs on REACT. The
planner swaps ``answer`` and ``research`` on requests that mix general
knowledge with a lookup ("define X, and get the weather in Y" -> the lookup
went to ``answer``, which has no tools): 10 of 138 plans, 2026-10-18. REACT
answers from knowledge with zero tool calls, so the swap is harmless. A
one-task plan keeps the fast, streaming ``answer`` pipeline — the swap needs
two tasks.

All research tasks of a plan then MERGE into one research task — one REACT
loop with every brief as a numbered part — instead of one loop each. In the
2026-10-22 on/off comparison REACT steps were 64% of the supervisor's tokens,
and 24 of 46 turns ran as multi-task plans. The merged task gets 2 more steps
per extra part (one lookup and one retry). If it is the only task left, its
reply IS the turn's reply: no combine call.

Security: the planner plans once, before any worker runs, so it never reads
worker output — an injection in a fetched page cannot change the plan
(plan-then-execute). ``combine`` has no tools; untrusted results reach it
nonce-wrapped, and main.py runs the output guardrail on its reply. ``trust``
is set here, by code, never by an LLM.

Pause/resume: a research worker can pause for approval. Its REACT graph runs
NESTED in ``run_worker``, so the pause pauses the whole supervisor turn under
one ``sup-…`` thread. On resume LangGraph re-runs ``run_worker`` from its
first line and the nested graph continues from its own checkpoint (measured
in a spike: each tool call ran once, the worker's LLM node did not re-run).
So ``run_worker`` does nothing with a side effect before it calls the worker,
and one run calls one worker.

One MCP gateway per turn, opened AROUND the graph (the Step 21.1 rule) and
only when a task needs it: the servers start once, all research workers share
them, and a pause inside a worker reaches LangGraph bare — not wrapped in the
gateway's task-group ExceptionGroup, which LangGraph would not recognise.
"""
import asyncio
import contextlib
import json
import logging
import operator
import time
from dataclasses import dataclass
from typing import Annotated, Any, Optional, TypedDict

from langgraph.errors import GraphBubbleUp
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command
from openai import OpenAI

from agentforge.approval import decision_label, run_interruptible
from agentforge.config import (
    AGENT_SUPERVISOR_MAX_WORKERS,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
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
from agentforge.logger import log_event, log_token_usage, token_meter
from agentforge.mcp_client import mcp_gateway
from agentforge.memory.semantic import store_memory
from agentforge.prompts import (
    SPOTLIGHT_INSTRUCTIONS,
    SUPERVISOR_COMBINE_PROMPT,
    SUPERVISOR_MERGED_RESEARCH_BRIEF,
    SUPERVISOR_PLANNER_PROMPT,
    SUPERVISOR_WORKER_NAMES,
    SUPERVISOR_WORKER_TOOLS,
)
from agentforge.rag.qa import answer_from_docs
from agentforge.reasoning.react_engine import (
    ReactContext,
    build_react_input,
    react_graph,
    tools_called,
)
from agentforge.router import choose_model
from agentforge.safety import new_spotlight_nonce, wrap_untrusted
from agentforge.tools import tool_catalog_for_classifier

logger = logging.getLogger(__name__)

PIPELINE = "sup"

# The intent each worker stands for: main.py runs a one-task plan through that
# intent's pipeline, and model routing (Step 28) sizes each worker by it.
WORKER_INTENT = {
    "remember": "REMEMBER",
    "research": "REACT",
    "read_user_documents": "DOCS_QA",
    "answer": "ANSWER",
}

# REACT steps for a research task: the REACT default, plus this many for each
# extra part merged into it (REACT makes one tool call per step).
RESEARCH_STEPS = 5
RESEARCH_STEPS_PER_EXTRA_PART = 2

# Shown when the planner calls no worker and writes no text either.
DIRECT_REPLY_FALLBACK = "Okay 🙂"
# Shown when every worker of a multi-task turn failed (no combine call is made).
ALL_FAILED_REPLY = "Sorry, I could not complete any part of that request. Please try again."

_client = None  # created on first API call, not at import time


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(base_url=OPENAI_BASE_URL) if OPENAI_BASE_URL else OpenAI()
    return _client


# ------------------------------------ planning ------------------------------------

@dataclass(frozen=True)
class Plan:
    """The planner's decision for one turn. ``tasks`` is a list of
    ``{"worker", "brief"}`` dicts; ``direct_reply`` is set only when there is
    no task (small talk)."""
    tasks: list
    direct_reply: Optional[str] = None


def plan_turn(user_input: str, history: list = None, trace_id: str = None) -> Optional[Plan]:
    """Split the request into worker tasks with one tool-calling LLM call.

    Returns None when the plan is unusable — the LLM call failed, a worker
    name is unknown, a brief is empty, or there are more tasks than
    ``AGENT_SUPERVISOR_MAX_WORKERS`` — so the caller falls back to the intent
    classifier rather than running a broken or truncated plan.

    The planner sees the (already trimmed) history so it can turn "there" or
    "it" into real names inside each brief; the workers never see it.
    """
    system = SUPERVISOR_PLANNER_PROMPT.format(
        max_workers=AGENT_SUPERVISOR_MAX_WORKERS,
        tool_catalog=tool_catalog_for_classifier())
    messages = [{"role": "system", "content": system}, *(history or []),
                {"role": "user", "content": user_input}]
    start = time.perf_counter()
    try:
        response = _get_client().chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            tools=SUPERVISOR_WORKER_TOOLS,
            # Planning should be repeatable: the same request, the same plan.
            temperature=0,
        )
        log_token_usage(response, "supervisor_plan", trace_id=trace_id)
        message = response.choices[0].message
    except Exception as e:
        logger.warning("supervisor: planner call failed: %s", e, exc_info=True)
        return _invalid_plan("llm_error", trace_id)

    tasks = []
    for call in message.tool_calls or []:
        function = getattr(call, "function", None)  # None for a non-function tool call
        name = getattr(function, "name", None)
        if name not in SUPERVISOR_WORKER_NAMES:
            return _invalid_plan("unknown_worker", trace_id, worker=name)
        try:
            args = json.loads(function.arguments or "{}")
        except json.JSONDecodeError:
            return _invalid_plan("bad_arguments", trace_id, worker=name)
        brief = args.get("brief") if isinstance(args, dict) else None
        if not isinstance(brief, str) or not brief.strip():
            return _invalid_plan("empty_brief", trace_id, worker=name)
        tasks.append({"worker": name, "brief": brief.strip()})
    if len(tasks) > AGENT_SUPERVISOR_MAX_WORKERS:
        return _invalid_plan("too_many_tasks", trace_id, tasks=len(tasks))

    direct = None
    if not tasks:
        direct = (message.content or "").strip() or DIRECT_REPLY_FALLBACK
    # Worker names and brief LENGTHS only — a brief is derived from user text.
    log_event("supervisor_plan", {
        "workers": [t["worker"] for t in tasks],
        "brief_chars": [len(t["brief"]) for t in tasks],
        "direct_reply": not tasks,
    }, trace_id=trace_id, duration_ms=(time.perf_counter() - start) * 1000)
    return Plan(tasks=tasks, direct_reply=direct)


def _invalid_plan(reason: str, trace_id: Optional[str], **details) -> None:
    log_event("supervisor_plan_invalid", {"reason": reason, **details}, trace_id=trace_id)
    return None


# ------------------------------------- the graph -------------------------------------

class _SupState(TypedDict):
    """Checkpointed turn state — plain data only."""
    user_id: str
    user_input: str           # the original request, for combine
    trace_id: Optional[str]
    tasks: list               # [{"worker", "brief"}], fixed for the turn
    # One result per finished worker. LangGraph merges the lists, so Step 21.3
    # can run workers in parallel (``Send``) without changing this state.
    results: Annotated[list, operator.add]
    workers_done: int         # logged at a pause (run_graph's progress_key)
    reply: Optional[str]


@dataclass
class _SupContext:
    """Per-invocation, never-checkpointed dependencies: the turn's MCP
    gateway, or None when no task needs one."""
    gw: Any = None


_GRAPH = None


def _graph():
    """The compiled supervisor graph (built once, on first use)."""
    global _GRAPH
    if _GRAPH is None:
        builder = StateGraph(_SupState, context_schema=_SupContext)
        builder.add_node("run_worker", _run_worker_node)
        builder.add_node("combine", _combine_node)
        builder.add_edge(START, "run_worker")
        builder.add_conditional_edges("run_worker", _route_after_worker,
                                      ["run_worker", "combine"])
        builder.add_edge("combine", END)
        _GRAPH = builder.compile(checkpointer=CHECKPOINTER)
    return _GRAPH


def _recursion_limit(tasks: list) -> int:
    # The nested REACT graph inherits this limit, so it must also cover the
    # longest REACT turn (2 * max_steps + 5; a merged task has more steps).
    steps = max((t.get("max_steps", RESEARCH_STEPS) for t in tasks), default=RESEARCH_STEPS)
    return max(2 * steps + 5, len(tasks) + 2) + 10


def _route_after_worker(state: _SupState) -> str:
    """Pure routing: run the next worker until every task has a result."""
    return "run_worker" if len(state["results"]) < len(state["tasks"]) else "combine"


async def _run_worker_node(state: _SupState, runtime: Runtime[_SupContext]) -> dict:
    """Run ONE worker — the next task without a result.

    This node can pause (a research worker's approval gate) and is re-run
    from its first line on resume, so nothing before the worker call has a
    side effect: the task is read from checkpointed state.
    """
    index = len(state["results"])
    result = await _run_worker(state["tasks"][index], index, state, runtime.context)
    return {"results": [result], "workers_done": index + 1}


async def _run_worker(task: dict, index: int, state: _SupState, ctx: _SupContext) -> dict:
    worker, brief = task["worker"], task["brief"]
    trace_id = state["trace_id"]
    model = choose_model(WORKER_INTENT[worker]).model
    start = time.perf_counter()
    tools = []
    status, summary, error = "ok", "", None
    with token_meter() as meter:
        try:
            if worker == "research":
                summary, tools = await _research(
                    brief, state, ctx.gw, model, task.get("max_steps", RESEARCH_STEPS))
            elif worker == "read_user_documents":
                summary = await asyncio.to_thread(
                    answer_from_docs, brief, stream=False, trace_id=trace_id, model=model,
                    end_trace=False)
            else:  # remember — the brief IS the fact (answer tasks run as research)
                await asyncio.to_thread(store_memory, state["user_id"], brief)
                summary = f"Saved this fact about the user: {brief}"
        except GraphBubbleUp:
            # A pause, not a failure: let LangGraph checkpoint the turn.
            log_event("worker_paused", {"worker": worker, "index": index},
                      trace_id=trace_id)
            raise
        except Exception as e:
            # One failed worker must not stop the others (the tasks are
            # independent); combine tells the user which part failed.
            logger.error("supervisor: worker %s (task %d) failed: %s",
                         worker, index, e, exc_info=True)
            status, summary, error = "error", "", type(e).__name__
    result = {
        "worker": worker, "planned_worker": task.get("planned_worker", worker),
        "brief": brief, "status": status,
        "summary": summary if isinstance(summary, str) else str(summary),
        "tools_called": tools, "tokens_used": meter.total, "error": error,
        "trust": _trust(worker, tools),
    }
    # Named after the OpenTelemetry GenAI agent span (Step 30 / 21b).
    log_event("invoke_agent", {
        "gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": worker,
        "planned_worker": result["planned_worker"],
        "index": index, "status": status, "error": error, "trust": result["trust"],
        "tools_called": tools, "tokens_used": meter.total,
    }, trace_id=trace_id, duration_ms=(time.perf_counter() - start) * 1000)
    return result


def _trust(worker: str, tools: list) -> str:
    """Set by code, never by an LLM. Outside text (a tool result, an uploaded
    document) makes a result untrusted; it then reaches combine nonce-wrapped."""
    if worker == "read_user_documents":
        return "untrusted"
    if worker == "research" and tools:
        return "untrusted"
    return "trusted"


async def _research(brief: str, state: _SupState, gw, model: str,
                    max_steps: int = RESEARCH_STEPS):
    """Run the REACT graph nested in this node, on the turn's shared gateway.
    A pause inside it propagates as GraphInterrupt and pauses this turn."""
    trace_id = state["trace_id"]
    graph_input = build_react_input(state["user_id"], brief, gw, max_steps=max_steps,
                                    trace_id=trace_id, model=model)
    out = await react_graph().ainvoke(graph_input, context=ReactContext(gw=gw))
    return out["reply"], tools_called(out["messages"])


async def _combine_node(state: _SupState) -> dict:
    """Write the final reply from all results. Never pauses; has no tools."""
    results = state["results"]
    trace_id = state["trace_id"]
    if not any(r["status"] == "ok" for r in results):
        log_event("supervisor_combine", {"skipped": "all_failed", "workers": len(results)},
                  trace_id=trace_id)
        return {"reply": ALL_FAILED_REPLY}
    if len(results) == 1:
        # One task left after the research merge: its reply already answers
        # every part, so a combine call would only rewrite it.
        log_event("supervisor_combine", {"skipped": "single_result"}, trace_id=trace_id)
        return {"reply": results[0]["summary"]}

    nonce = new_spotlight_nonce()
    parts = []
    for n, r in enumerate(results, 1):
        if r["status"] != "ok":
            parts.append(f"Part {n} - task: {r['brief']}\nResult: FAILED ({r['error']})")
            continue
        body = (r["summary"] if r["trust"] == "trusted"
                else wrap_untrusted(r["summary"], source=f"worker:{r['worker']}", nonce=nonce))
        parts.append(f"Part {n} - task: {r['brief']}\nResult:\n{body}")
    messages = [
        {"role": "system", "content": SUPERVISOR_COMBINE_PROMPT},
        {"role": "system", "content": SPOTLIGHT_INSTRUCTIONS},
        {"role": "user", "content": f"User's request:\n{state['user_input']}\n\n"
                                    + "\n\n".join(parts)},
    ]
    start = time.perf_counter()
    try:
        response = await asyncio.to_thread(
            _get_client().chat.completions.create, model=OPENAI_MODEL, messages=messages)
        log_token_usage(response, "supervisor_combine", trace_id=trace_id)
        reply = (response.choices[0].message.content or "").strip()
    except Exception as e:
        logger.error("supervisor: combine call failed: %s", e, exc_info=True)
        reply = ""
    if not reply:
        # Degrade without an LLM: the successful results, one per paragraph.
        reply = "\n\n".join(r["summary"] for r in results if r["status"] == "ok")
        log_event("supervisor_combine", {"fallback": "joined_results"}, trace_id=trace_id)
    else:
        log_event("supervisor_combine", {
            "workers": len(results),
            "failed": sum(r["status"] != "ok" for r in results),
            "untrusted": sum(r["trust"] == "untrusted" for r in results),
            "reply_length": len(reply),
        }, trace_id=trace_id, duration_ms=(time.perf_counter() - start) * 1000)
    return {"reply": reply}


# ---------------------------------- entry points ----------------------------------

def run_multi(user_id: str, user_input: str, tasks: list, trace_id: str = None,
              approval_handler=None) -> str:
    """Run a plan with two or more tasks; return the combined reply.

    A research worker that needs approval pauses the whole turn: an
    ``ApprovalRequired`` carrying a ``sup-…`` thread ID propagates out, and
    ``resume`` finishes the turn with the human's decision.
    """
    return run_interruptible(_run_multi_async(user_id, user_input, tasks, trace_id,
                                              approval_handler))


def _multi_task(task: dict) -> dict:
    """An ``answer`` task in a multi-task plan runs on REACT (see the module
    docstring); the planner's choice is kept for the logs."""
    if task["worker"] == "answer":
        return {"worker": "research", "brief": task["brief"], "planned_worker": "answer"}
    return dict(task)


def _merge_research(tasks: list, user_input: str) -> list:
    """All research tasks become ONE, at the first one's position (see the
    module docstring). Other tasks keep their order. Fewer than two research
    tasks: the plan is returned unchanged."""
    research = [t for t in tasks if t["worker"] == "research"]
    if len(research) < 2:
        return tasks
    parts = "\n".join(f"{n}. {t['brief']}" for n, t in enumerate(research, 1))
    merged = {
        "worker": "research",
        "brief": SUPERVISOR_MERGED_RESEARCH_BRIEF.format(user_input=user_input, parts=parts),
        "planned_worker": ",".join(t.get("planned_worker", "research") for t in research),
        "merged_parts": len(research),
        "max_steps": RESEARCH_STEPS + RESEARCH_STEPS_PER_EXTRA_PART * (len(research) - 1),
    }
    out = []
    for t in tasks:
        if t["worker"] != "research":
            out.append(t)
        elif t is research[0]:
            out.append(merged)
    return out


def _turn_gateway(tasks: list, trace_id: Optional[str], handler):
    """The turn's MCP gateway if any task needs one, else a no-op context."""
    if any(t["worker"] == "research" for t in tasks):
        return mcp_gateway(trace_id, approval_handler=handler)
    return contextlib.nullcontext(None)


async def _run_multi_async(user_id, user_input, tasks, trace_id, approval_handler) -> str:
    planned = len(tasks)
    tasks = _merge_research([_multi_task(t) for t in tasks], user_input)
    thread_id = new_thread_id(PIPELINE)
    log_event("supervisor_start", {"thread_id": thread_id,
                                   "workers": [t["worker"] for t in tasks],
                                   "planned_tasks": planned,
                                   "merged_research_parts": max(
                                       (t.get("merged_parts", 1) for t in tasks
                                        if t["worker"] == "research"), default=0)},
              trace_id=trace_id)
    initial = {"user_id": user_id, "user_input": user_input, "trace_id": trace_id,
               "tasks": tasks, "results": [], "workers_done": 0, "reply": None}
    # No front-end handler -> None, so the gateway denies by default.
    handler = (GraphApprovalHandler(approval_handler)
               if approval_handler is not None else None)
    async with _turn_gateway(tasks, trace_id, handler) as gw:
        return await run_graph(
            _graph(), initial, thread_id=thread_id,
            recursion_limit=_recursion_limit(tasks),
            context=_SupContext(gw=gw),
            pipeline=PIPELINE, trace_id=trace_id, output_key="reply",
            progress_key="workers_done")


def resume(thread_id: str, decision, approval_handler=None) -> str:
    """Resume a paused supervisor turn with the human's decision for the
    stored call. Later gated calls go back to ``approval_handler``."""
    return run_interruptible(_resume_async(thread_id, decision, approval_handler))


async def _resume_async(thread_id: str, decision, fallback) -> str:
    pending = await apending_approval(_graph(), thread_id, PIPELINE)
    trace_id = pending.trace_id
    log_event("supervisor_resume", {
        "thread_id": thread_id,
        "workers_done": pending.values.get("workers_done"),
        "tool": pending.request.tool,
        "decision": decision_label(decision),
    }, trace_id=trace_id)
    tasks = pending.values.get("tasks", [])
    # A fresh gateway (the original's servers died when the pause unwound the
    # turn). Its first gated call is the re-run of the stored call, so
    # ``resuming=True`` hands it the decision; later calls ask ``fallback``.
    handler = GraphApprovalHandler(fallback, resuming=True)
    async with _turn_gateway(tasks, trace_id, handler) as gw:
        return await run_graph(
            _graph(), Command(resume=decision), thread_id=thread_id,
            recursion_limit=_recursion_limit(tasks),
            context=_SupContext(gw=gw),
            pipeline=PIPELINE, trace_id=trace_id, output_key="reply",
            progress_key="workers_done")


def pending_supervisor_approval(thread_id: str) -> PendingApproval:
    """The paused supervisor turn ``thread_id``; ValueError if no longer pending."""
    return pending_approval(_graph(), thread_id, PIPELINE)
