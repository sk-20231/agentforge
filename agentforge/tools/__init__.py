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

from openai import OpenAI

from agentforge.approval import (
    ApprovalRequired,
    make_resume_handler,
    run_interruptible,
)
from agentforge.config import OPENAI_BASE_URL, OPENAI_MODEL
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


async def _run_tool_loop(messages: list, trace_id: str = None, approval_handler=None,
                         user_id: str = "") -> str:
    """Open the shared MCP gateway, then run the single-pick ACT tool loop.

    ACT is MCP-only (Step 17c.1): every tool is discovered from the gateway; there
    is no in-process dispatch. The gateway holds one session per server open for
    the whole turn, so the model's tool call and the follow-up share connections.

    ``approval_handler`` is the front-end's human-in-the-loop callback (Step 17f),
    passed through untouched — the gate lives in the gateway, and this loop stays
    security-ignorant. If the handler raises ApprovalRequired it propagates out of
    this loop by design (the front-end catches it and replays the turn).
    """
    async with mcp_gateway(trace_id, approval_handler=approval_handler) as gw:

        # ---- 1. ACT is MCP-only: tools come entirely from discovery ----
        if not gw.has_tools:
            # No MCP tools discovered (all servers down / misconfigured). With
            # tool_choice="required" and an empty tool list the API would error,
            # so bail out gracefully instead of crashing the turn.
            log_event("act_no_tools_available", {}, trace_id=trace_id)
            return json.dumps({
                "reply": "I don't have any tools available right now to do that. Please try again shortly.",
                "store_memory": False,
                "memory_text": "",
            })

        # ---- 2. First LLM call: let the model pick a tool ----
        try:
            response = _get_client().chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                tools=gw.openai_schemas,
                tool_choice="required",
            )
            log_token_usage(response, "act_tool_call", trace_id=trace_id)
        except Exception:
            return json.dumps({
                "reply": "I couldn't complete that request due to a service error. Please try again.",
                "store_memory": False,
                "memory_text": "",
            })

        message = response.choices[0].message
        if not message.tool_calls:
            return message.content

        # ---- 3 + 4. Dispatch the tool calls, then the follow-up LLM call ----
        return await _dispatch_and_followup(
            gw, messages, message.tool_calls, [], 0, trace_id, user_id)


async def _dispatch_and_followup(gw, messages: list, tool_calls, tool_messages: list,
                                 start_index: int, trace_id: str, user_id: str) -> str:
    """Dispatch ``tool_calls[start_index:]`` and run the follow-up LLM call.

    Shared by the fresh ACT path (``start_index=0``) and the resume path (which
    passes the interrupted call's index and the tool messages gathered before
    the interrupt). gw.call handles unknown/hallucinated tool names by returning
    a readable error string (it does not raise) — same recoverable-observation
    contract the loop already relies on.
    """
    for i in range(start_index, len(tool_calls)):
        tool_call = tool_calls[i]
        tool_name = tool_call.function.name
        try:
            arguments = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            tool_messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": "Error: Tool arguments were invalid.",
            })
            continue

        try:
            tool_result = await gw.call(tool_name, arguments)
        except ApprovalRequired as exc:
            # CHECKPOINT (Step 17f): freeze this loop's state on the exception
            # so resume_tool_loop can finish the turn after the human decides.
            # The loop, not the gateway, owns the loop state.
            exc.continuation = {
                "pipeline": "act",
                "user_id": user_id,
                "messages": messages,
                "tool_calls": tool_calls,
                "tool_messages": list(tool_messages),
                "pending_index": i,
                "trace_id": trace_id,
            }
            raise
        tool_messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": tool_result,
        })

    # ---- Follow-up LLM call: turn tool results into a final answer ----
    try:
        followup = _get_client().chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                *messages,
                {"role": "assistant", "tool_calls": tool_calls},
                *tool_messages,
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
        return followup.choices[0].message.content
    except Exception:
        return json.dumps({
            "reply": "I ran into an error after using the tool. Please try again.",
            "store_memory": False,
            "memory_text": "",
        })


async def _resume_tool_loop_async(interrupt: ApprovalRequired, decision: bool,
                                  fallback=None) -> str:
    """Re-enter an interrupted ACT turn mid-flight (Step 17f resume).

    Opens a fresh gateway (the original's server subprocesses died when the
    interrupt unwound the turn), settles the STORED pending call with the
    human's decision via a one-shot handler, then finishes the remaining tool
    calls and the follow-up. Later gated calls go to ``fallback`` and can
    interrupt again.
    """
    cont = interrupt.continuation
    handler = make_resume_handler(decision, interrupt.request, fallback)
    log_event("act_resume", {
        "tool": interrupt.request.tool,
        "decision": "approved" if decision else "denied",
    }, trace_id=cont["trace_id"])

    async with mcp_gateway(cont["trace_id"], approval_handler=handler) as gw:
        return await _dispatch_and_followup(
            gw, cont["messages"], cont["tool_calls"], cont["tool_messages"],
            cont["pending_index"], cont["trace_id"], cont["user_id"])


# -------------------- MAIN ENTRY --------------------

def run_llm_with_tools(user_id: str, user_input: str, trace_id: str = None,
                       approval_handler=None) -> str:
    """Discover MCP tools at runtime and execute the (MCP-only) tool loop."""
    messages = build_messages(user_id, user_input)
    return run_interruptible(_run_tool_loop(messages, trace_id,
                                            approval_handler=approval_handler,
                                            user_id=user_id))


def resume_tool_loop(interrupt: ApprovalRequired, decision: bool,
                     approval_handler=None) -> str:
    """Resume an ACT turn that was interrupted for human approval (Step 17f).

    ``interrupt`` is the ApprovalRequired the front-end caught — its
    ``continuation`` holds the frozen loop state. ``decision`` is the human's
    Allow (True) / Deny (False) for the stored pending call. Resume — not
    replay — so the decision settles the EXACT call the human looked at; the
    LLM is never asked to regenerate it. Returns the same JSON-string contract
    as ``run_llm_with_tools`` (main.resume_agent parses it).
    """
    return run_interruptible(
        _resume_tool_loop_async(interrupt, decision, approval_handler))
