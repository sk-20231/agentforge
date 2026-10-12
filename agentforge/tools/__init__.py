"""Tool orchestration for the ACT pipeline.

As of Step 17c.2 the agent has **zero hardcoded tools**. Both ACT
(``run_llm_with_tools``) and ReAct (``reasoning.react_engine``) reach every tool
over MCP via the shared gateway in ``agentforge.mcp_client``: tools are discovered
at runtime from the servers in ``MCP_SERVERS`` (via ``tools/list``) and dispatched
through an open session. Adding a tool means adding a server to ``MCP_SERVERS`` —
no edits here. There is no in-process tool registry any more.

The tool *implementation* modules (``wikipedia``, ``weather``, ``news``) still
live under this package, but only because the MCP servers in ``mcp_servers/``
delegate to them. The agent itself imports none of them as callable tools.
"""
import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command

from openai import OpenAI

from agentforge.approval import decision_label, run_interruptible
from agentforge.config import OPENAI_BASE_URL, OPENAI_MODEL
from agentforge.hitl_graph import (
    CHECKPOINTER,
    GraphApprovalHandler,
    PendingApproval,
    apending_approval,
    new_thread_id,
    pending_approval,
    run_graph,
)
from agentforge.logger import log_event, log_token_usage
from agentforge.mcp_client import mcp_gateway
from agentforge.memory.semantic import get_relevant_memories
from agentforge.prompts import (
    MEMORY_INSTRUCTIONS,
    OUTPUT_SCHEMA,
    SPOTLIGHT_INSTRUCTIONS,
    SYSTEM_PROMPT,
)
from agentforge.tools import news, weather, wikipedia

logger = logging.getLogger(__name__)

_client = None  # created on first API call, not at import time


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(base_url=OPENAI_BASE_URL) if OPENAI_BASE_URL else OpenAI()
    return _client


# -------------------- CLASSIFIER TOOL CATALOG (from MCP discovery) --------------------
# The intent classifier needs the list of available tools to route ACT queries.
# Since ACT is MCP-only, the catalog is sourced from MCP discovery, not the local
# registry. It is discovered once and cached (primed at app startup — see
# prime_tool_catalog). None means "not yet primed".
_TOOL_CATALOG_CACHE: list = None  # list of {"name": str, "description": str}


def tool_catalog_for_classifier() -> str:
    """Render the MCP-discovered tools as a bullet list for the intent classifier.

    Single source of truth is now each MCP server's own tool schema (FastMCP
    derives it from the wrapper's type hints + docstring). Adding a server to
    MCP_SERVERS updates the classifier automatically — no drift, no edits here.

    The catalog is primed once at startup (prime_tool_catalog). If something
    calls this before priming, we prime lazily on first use.

    Recovery: we re-discover when the cache is None OR empty. An empty cache
    means a previous discovery found nothing — usually a transient failure at
    startup (servers/network briefly down). Re-discovering on the next call lets
    the agent recover once the servers are reachable, instead of staying blind to
    every tool for the rest of the session. (If MCP_SERVERS is genuinely empty,
    the retry is cheap — there are no subprocesses to spawn.)
    """
    if not _TOOL_CATALOG_CACHE:  # None or [] -> (re)discover
        prime_tool_catalog(force=True)
    return "\n".join(
        f"    - {t['name']}: {t['description']}" for t in _TOOL_CATALOG_CACHE
    )


# Re-export individual tool functions so existing imports keep working.
wikipedia_lookup = wikipedia.wikipedia_lookup
get_weather = weather.get_weather
get_top_news = news.get_top_news


# -------------------- PROMPT BUILDER --------------------

def build_messages(user_id: str, user_input: str):
    memories = get_relevant_memories(user_id, user_input)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": SPOTLIGHT_INSTRUCTIONS},
        {
            "role": "system",
            "content": f"{MEMORY_INSTRUCTIONS}\n\nRelevant memories:\n{memories}",
        },
        {"role": "user", "content": user_input},
        {"role": "system", "content": OUTPUT_SCHEMA},
    ]


# -------------------- MCP INTEGRATION --------------------
# Discovery + dispatch live in agentforge.mcp_client (the shared gateway). Both
# this ACT pipeline and the ReAct pipeline open one gateway per turn, so the
# protocol plumbing and the untrusted-output boundary exist in exactly one place.


async def _discover_catalog_async() -> list:
    """Discover {name, description} per tool for the classifier catalog.

    Lightweight discovery: open a gateway, read its catalog, tear it down. Used to
    prime the intent classifier's tool list (see prime_tool_catalog). Reuses the
    same gateway as the ACT/ReAct loops — one discovery code path, no drift.
    """
    async with mcp_gateway() as gw:
        return [{"name": t["name"], "description": t["description"]} for t in gw.catalog]


def prime_tool_catalog(force: bool = False) -> list:
    """Discover the MCP tool catalog once and cache it for the classifier.

    Call this at application startup (run.py / app.py) so the discovery cost is
    paid up front rather than on the first user message. Idempotent: returns the
    cached catalog unless ``force=True``.
    """
    global _TOOL_CATALOG_CACHE
    if _TOOL_CATALOG_CACHE is not None and not force:
        return _TOOL_CATALOG_CACHE
    _TOOL_CATALOG_CACHE = asyncio.run(_discover_catalog_async())
    log_event("mcp_catalog_primed", {"tools": [t["name"] for t in _TOOL_CATALOG_CACHE]})
    return _TOOL_CATALOG_CACHE


# -------------------- ACT GRAPH (Step 21.1) --------------------
#
#     START -> pick --(tool calls)--> dispatch --(calls left)--> dispatch
#               |                         +------(all done)----> followup -> END
#               +--(no tools / error / no tool calls)--> END
#
# ``pick`` and ``followup`` are the LLM calls; they never pause. ``dispatch``
# runs exactly ONE tool call per graph step and can pause on it. LangGraph
# saves state only BETWEEN nodes and re-runs a paused node from its first line,
# so each finished call (its tool message and the advanced index) is saved
# before the next call can pause. A resume re-runs only the paused call. A loop
# over all calls inside ONE node would re-run every earlier call on each later
# resume — with two gated calls, the first tool would run twice.

_NO_TOOLS_REPLY = json.dumps({
    "reply": "I don't have any tools available right now to do that. Please try again shortly.",
    "store_memory": False,
    "memory_text": "",
})
_SERVICE_ERROR_REPLY = json.dumps({
    "reply": "I couldn't complete that request due to a service error. Please try again.",
    "store_memory": False,
    "memory_text": "",
})
_FOLLOWUP_ERROR_REPLY = json.dumps({
    "reply": "I ran into an error after using the tool. Please try again.",
    "store_memory": False,
    "memory_text": "",
})

# Graph steps per turn = tool calls + 2 (pick, followup). ``index`` strictly
# increases, so the graph cannot loop forever; LangGraph's default limit (25)
# would only cut off a response with more than 23 tool calls.
_ACT_RECURSION_LIMIT = 256


class _ActState(TypedDict):
    """Checkpointed turn state. Plain data only — the checkpointer serializes
    it. Live objects (the gateway) travel in ``_ActContext`` instead."""
    user_id: str
    messages: list
    model: Optional[str]      # routed model (Step 28) — survives a resume
    trace_id: Optional[str]   # issue #7 — the resumed half logs on the same trace
    tool_calls: list          # [{"id", "name", "arguments"}] — not SDK objects
    index: int                # next tool call to dispatch
    tool_messages: list       # one role:"tool" message per finished call
    granted: list             # APPROVE_TURN grants as [server, tool] pairs (issue #6)
    output: Optional[str]     # the loop's JSON-string result


@dataclass
class _ActContext:
    """Per-invocation, never-checkpointed dependencies for the nodes."""
    gw: Any


async def _pick_node(state: _ActState, runtime: Runtime[_ActContext]) -> dict:
    """First LLM call: the model must pick one or more tools.

    Never pauses. Converts the SDK's tool-call objects to plain dicts so the
    checkpointer can save them, or ends the turn early (sets ``output``).
    """
    gw = runtime.context.gw
    trace_id = state["trace_id"]

    # ACT is MCP-only: tools come entirely from discovery.
    if not gw.has_tools:
        # No MCP tools discovered (all servers down / misconfigured). With
        # tool_choice="required" and an empty tool list the API would error,
        # so bail out gracefully instead of crashing the turn.
        log_event("act_no_tools_available", {}, trace_id=trace_id)
        return {"output": _NO_TOOLS_REPLY}

    try:
        response = _get_client().chat.completions.create(
            model=state["model"] or OPENAI_MODEL,
            messages=state["messages"],
            tools=gw.openai_schemas,
            tool_choice="required",
        )
        log_token_usage(response, "act_tool_call", trace_id=trace_id)
    except Exception as e:
        logger.error("act: tool-pick LLM call failed: %s", e, exc_info=True)
        return {"output": _SERVICE_ERROR_REPLY}

    message = response.choices[0].message
    if not message.tool_calls:
        return {"output": message.content}

    calls = [{"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
             for tc in message.tool_calls]
    # Tool NAMES only — arguments can hold user data (Step 17e gap A).
    log_event("act_tools_picked", {"tools": [c["name"] for c in calls]},
              trace_id=trace_id)
    return {"tool_calls": calls, "index": 0}


async def _dispatch_node(state: _ActState, runtime: Runtime[_ActContext]) -> dict:
    """Run the ONE tool call at ``index`` through the MCP gateway.

    This node can PAUSE: the gateway's approval gate may call ``interrupt()``.
    On resume LangGraph re-runs it from the top, so everything before
    ``gw.call`` reads checkpointed state only — the re-run reaches the same
    approval request, and earlier calls are already saved, never repeated.
    gw.call handles unknown/hallucinated tool names by returning a readable
    error string (it does not raise) — a recoverable observation.
    """
    gw = runtime.context.gw
    index = state["index"]
    call = state["tool_calls"][index]
    # Turn-scoped grants (issue #6) live in checkpointed state; the gateway
    # is rebuilt on resume, so seed it here and write it back below.
    gw.granted.update(tuple(g) for g in state["granted"])

    try:
        arguments = json.loads(call["arguments"])
    except json.JSONDecodeError:
        logger.warning("act: tool call %d (%s) had invalid JSON arguments",
                       index, call["name"])
        content = "Error: Tool arguments were invalid."
    else:
        content = await gw.call(call["name"], arguments)

    return {
        "tool_messages": [*state["tool_messages"],
                          {"role": "tool", "tool_call_id": call["id"], "content": content}],
        "index": index + 1,
        "granted": sorted([list(g) for g in gw.granted]),
    }


async def _followup_node(state: _ActState) -> dict:
    """Second LLM call: turn the tool results into the final JSON answer.
    Never pauses."""
    trace_id = state["trace_id"]
    assistant = {
        "role": "assistant",
        "tool_calls": [{"id": c["id"], "type": "function",
                        "function": {"name": c["name"], "arguments": c["arguments"]}}
                       for c in state["tool_calls"]],
    }
    try:
        followup = _get_client().chat.completions.create(
            model=state["model"] or OPENAI_MODEL,
            messages=[
                *state["messages"],
                assistant,
                *state["tool_messages"],
                {
                    "role": "system",
                    "content": (
                        "You have received the tool result above. "
                        "Now produce your FINAL response. "
                        "Set action.type to 'final', write the answer to the user "
                        "in the 'reply' field, and do NOT call any more tools."
                    ),
                },
            ],
        )
        log_token_usage(followup, "act_tool_followup", trace_id=trace_id)
        return {"output": followup.choices[0].message.content}
    except Exception as e:
        logger.error("act: follow-up LLM call failed: %s", e, exc_info=True)
        return {"output": _FOLLOWUP_ERROR_REPLY}


def _route_after_pick(state: _ActState) -> str:
    """Tool calls go to dispatch; an early ``output`` ends the turn."""
    return "dispatch" if state["tool_calls"] else END


def _route_after_dispatch(state: _ActState) -> str:
    """One more dispatch per remaining call, then the follow-up."""
    return "dispatch" if state["index"] < len(state["tool_calls"]) else "followup"


_ACT_GRAPH = None


def _act_graph():
    """The compiled ACT graph (built once, on first use)."""
    global _ACT_GRAPH
    if _ACT_GRAPH is None:
        builder = StateGraph(_ActState, context_schema=_ActContext)
        builder.add_node("pick", _pick_node)
        builder.add_node("dispatch", _dispatch_node)
        builder.add_node("followup", _followup_node)
        builder.add_edge(START, "pick")
        builder.add_conditional_edges("pick", _route_after_pick, ["dispatch", END])
        builder.add_conditional_edges("dispatch", _route_after_dispatch,
                                      ["dispatch", "followup"])
        builder.add_edge("followup", END)
        _ACT_GRAPH = builder.compile(checkpointer=CHECKPOINTER)
    return _ACT_GRAPH


async def _run_act(graph_input, thread_id: str, gw,
                   trace_id: Optional[str]) -> str:
    return await run_graph(
        _act_graph(), graph_input, thread_id=thread_id,
        recursion_limit=_ACT_RECURSION_LIMIT, context=_ActContext(gw=gw),
        pipeline="act", trace_id=trace_id,
        output_key="output", progress_key="index")


async def _run_tool_loop(messages: list, trace_id: str = None, approval_handler=None,
                         user_id: str = "", model: str = None) -> str:
    """Open the shared MCP gateway, then run the ACT graph.

    ACT is MCP-only (Step 17c.1): every tool is discovered from the gateway; there
    is no in-process dispatch. The gateway holds one session per server open for
    the whole turn, around the graph, so the tool calls and the follow-up share
    connections.

    ``approval_handler`` is the front-end's human-in-the-loop callback (Step 17f).
    The gate lives in the gateway, and this loop stays security-ignorant; the
    handler reaches the gateway through the LangGraph bridge, so a handler that
    cannot answer synchronously pauses the graph (surfaced as ApprovalRequired).
    """
    # No front-end handler -> pass None, so the gateway denies by default and
    # audits it as "no approval handler configured".
    handler = (GraphApprovalHandler(approval_handler)
               if approval_handler is not None else None)
    async with mcp_gateway(trace_id, approval_handler=handler) as gw:
        initial = {
            "user_id": user_id, "messages": messages, "model": model,
            "trace_id": trace_id, "tool_calls": [], "index": 0,
            "tool_messages": [], "granted": [], "output": None,
        }
        return await _run_act(initial, new_thread_id("act"), gw, trace_id)


async def _resume_tool_loop_async(thread_id: str, decision, fallback=None) -> str:
    """Resume a paused ACT graph with the human's decision (Step 17f / 21.1).

    Opens a fresh gateway (the original's server subprocesses died when the
    interrupt unwound the turn) and resumes the checkpointed thread with
    ``Command(resume=decision)``. LangGraph re-runs ``dispatch`` for the paused
    call only; the remaining calls and the follow-up continue from there.
    Later gated calls go to ``fallback`` and can pause again.
    """
    pending = await apending_approval(_act_graph(), thread_id, "act")
    trace_id = pending.trace_id
    log_event("act_resume", {
        "thread_id": thread_id,
        "index": pending.values.get("index"),
        "tool": pending.request.tool,
        "decision": decision_label(decision),
    }, trace_id=trace_id)

    handler = GraphApprovalHandler(fallback, resuming=True)
    async with mcp_gateway(trace_id, approval_handler=handler) as gw:
        return await _run_act(Command(resume=decision), thread_id, gw, trace_id)


# -------------------- MAIN ENTRY --------------------

def run_llm_with_tools(user_id: str, user_input: str, trace_id: str = None,
                       approval_handler=None, model: str = None) -> str:
    """Discover MCP tools at runtime and execute the (MCP-only) tool loop.

    ``model`` (Step 28) is the routed model for this turn; None → OPENAI_MODEL,
    so pre-routing callers are unaffected. ACT is a routine intent (small tier),
    but a low-confidence route can still hand the frontier model here.
    """
    messages = build_messages(user_id, user_input)
    return run_interruptible(_run_tool_loop(messages, trace_id,
                                            approval_handler=approval_handler,
                                            user_id=user_id, model=model))


def resume_tool_loop(thread_id: str, decision,
                     approval_handler=None) -> str:
    """Resume an ACT turn that was interrupted for human approval (Step 17f).

    ``thread_id`` is the paused graph's handle (``ApprovalRequired.thread_id``,
    Step 21.1). ``decision`` is the human's Deny (False) / Allow once (True) / APPROVE_TURN
    (allow this tool for the rest of the turn — issue #6) for the stored
    pending call. Resume — not replay — so the decision settles the EXACT call
    the human looked at; the LLM is never asked to regenerate it. Returns the
    same JSON-string contract as ``run_llm_with_tools`` (main.resume_agent
    parses it).
    """
    return run_interruptible(
        _resume_tool_loop_async(thread_id, decision, approval_handler))


def pending_tool_approval(thread_id: str) -> PendingApproval:
    """The paused ACT turn ``thread_id``; ValueError if no longer pending."""
    return pending_approval(_act_graph(), thread_id, "act")
