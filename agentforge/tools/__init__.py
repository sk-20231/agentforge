"""Tool registry and orchestration.

The ACT pipeline is **MCP-only** as of Step 17c.1: ``run_llm_with_tools``
discovers every tool at runtime from the servers in ``MCP_SERVERS`` (via
``tools/list``) and never dispatches to an in-process tool. Adding a tool
means adding a server to ``MCP_SERVERS`` — no edits here.

The local ``TOOL_MODULES`` / ``TOOL_REGISTRY`` / ``execute_tool`` machinery
below is **retained only for the ReAct pipeline** (``reasoning.react_engine``),
which still dispatches tools in-process. Step 17c.2 migrates ReAct to MCP and
deletes this machinery — at which point the agent has *zero* hardcoded tools.

Each local tool module under ``agentforge.tools`` exports two names:
- ``TOOL_FUNCTION`` — the callable, signature ``(**kwargs) -> str``
- ``TOOL_SCHEMA``   — the OpenAI function-calling schema for the tool
"""
import asyncio
import contextlib
import json
import logging
import sys
from typing import Any, Callable, Dict

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from openai import OpenAI

from agentforge.config import MCP_SERVERS, OPENAI_BASE_URL, OPENAI_MODEL
from agentforge.logger import log_event, log_token_usage
from agentforge.memory.semantic import get_relevant_memories
from agentforge.prompts import MEMORY_INSTRUCTIONS, OUTPUT_SCHEMA, SYSTEM_PROMPT
from agentforge.tools import news, weather, wikipedia

logger = logging.getLogger(__name__)

_client = None  # created on first API call, not at import time


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(base_url=OPENAI_BASE_URL) if OPENAI_BASE_URL else OpenAI()
    return _client


# -------------------- LOCAL TOOL REGISTRY (ReAct only) --------------------
# Retained ONLY for the ReAct pipeline, which still dispatches tools in-process
# via execute_tool. The ACT pipeline no longer uses any of these (Step 17c.1).
# Step 17c.2 migrates ReAct to MCP and removes this whole block.
TOOL_MODULES = [wikipedia, weather, news]

TOOL_REGISTRY: Dict[str, Callable] = {
    m.TOOL_FUNCTION.__name__: m.TOOL_FUNCTION for m in TOOL_MODULES
}

TOOLS_SCHEMA = [m.TOOL_SCHEMA for m in TOOL_MODULES]


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
        {
            "role": "system",
            "content": f"{MEMORY_INSTRUCTIONS}\n\nRelevant memories:\n{memories}",
        },
        {"role": "user", "content": user_input},
        {"role": "system", "content": OUTPUT_SCHEMA},
    ]


# -------------------- TOOL EXECUTION --------------------

def execute_tool(name: str, arguments: Dict[str, Any]) -> str:
    log_event("tool_call", {"tool": name, "arguments": arguments})

    if name not in TOOL_REGISTRY:
        return f"Error: Unknown tool '{name}'"

    try:
        result = TOOL_REGISTRY[name](**arguments)
        log_event("tool_result", {"tool": name, "result": result})
        return result
    except Exception as e:
        return f"Tool execution error: {e}"


# -------------------- MCP INTEGRATION --------------------

def _mcp_tool_to_openai_schema(tool) -> dict:
    """Convert an MCP tool definition to the OpenAI function-calling schema shape."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema,
        },
    }


async def _call_mcp_tool(
    session,
    tool_name: str,
    arguments: Dict[str, Any],
    trace_id: str = None,
) -> str:
    """Call one tool via an already-open MCP session; return its text result."""
    try:
        result = await session.call_tool(tool_name, arguments)
        text = " ".join(c.text for c in result.content if hasattr(c, "text"))
        log_event("mcp_tool_call", {"tool": tool_name, "is_error": result.isError}, trace_id=trace_id)
        return f"Tool error: {text}" if result.isError else text
    except Exception as exc:
        return f"MCP tool error: {exc}"


async def _discover_catalog_async() -> list:
    """Connect to each MCP server once and collect {name, description} per tool.

    This is the lightweight discovery used to build the classifier catalog — it
    only needs names + descriptions, so the sessions are opened and torn down
    immediately (unlike _run_tool_loop, which keeps them open to *call* tools).
    """
    catalog: list = []
    async with contextlib.AsyncExitStack() as stack:
        for server_path in MCP_SERVERS:
            params = StdioServerParameters(command=sys.executable, args=[server_path])
            try:
                read, write = await stack.enter_async_context(stdio_client(params))
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                tools_result = await session.list_tools()
                for tool in tools_result.tools:
                    catalog.append({"name": tool.name, "description": tool.description or ""})
            except Exception as exc:
                logger.warning("Catalog discovery failed for %s: %s", server_path, exc)
    return catalog


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


async def _run_tool_loop(messages: list, trace_id: str = None) -> str:
    """Connect to configured MCP servers, discover their tools, run the tool loop.

    Session lifecycle: one subprocess spawn per MCP server per run_llm_with_tools
    call. All tool calls within a single agent turn share the same open sessions —
    no reconnect overhead between tool calls.
    """
    mcp_tool_registry: Dict[str, str] = {}  # tool_name → server_path
    mcp_schemas: list = []
    open_sessions: Dict[str, Any] = {}       # server_path → live ClientSession

    async with contextlib.AsyncExitStack() as stack:

        # ---- 1. Discovery: connect to each server, ask tools/list ----
        for server_path in MCP_SERVERS:
            params = StdioServerParameters(command=sys.executable, args=[server_path])
            try:
                read, write = await stack.enter_async_context(stdio_client(params))
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                open_sessions[server_path] = session

                tools_result = await session.list_tools()
                for tool in tools_result.tools:
                    mcp_tool_registry[tool.name] = server_path
                    mcp_schemas.append(_mcp_tool_to_openai_schema(tool))

                log_event(
                    "mcp_discovery",
                    {"server": server_path, "tools": [t.name for t in tools_result.tools]},
                    trace_id=trace_id,
                )
            except Exception as exc:
                logger.warning("MCP discovery failed for %s: %s", server_path, exc)

        # ---- 2. ACT is MCP-only: tools come entirely from discovery ----
        all_schemas = mcp_schemas
        if not all_schemas:
            # No MCP tools discovered (all servers down / misconfigured). With
            # tool_choice="required" and an empty tool list the API would error,
            # so bail out gracefully instead of crashing the turn.
            log_event("act_no_tools_available", {}, trace_id=trace_id)
            return json.dumps({
                "reply": "I don't have any tools available right now to do that. Please try again shortly.",
                "store_memory": False,
                "memory_text": "",
            })

        # ---- 3. First LLM call: let the model pick a tool ----
        try:
            response = _get_client().chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                tools=all_schemas,
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

        # ---- 4. Dispatch each tool call: MCP registry first, local second ----
        tool_messages = []
        for tool_call in message.tool_calls:
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

            if tool_name in mcp_tool_registry:
                tool_result = await _call_mcp_tool(
                    open_sessions[mcp_tool_registry[tool_name]],
                    tool_name,
                    arguments,
                    trace_id,
                )
            else:
                # ACT is MCP-only — there is no local dispatch. A tool name that
                # isn't in the discovered registry means the model hallucinated it.
                log_event("act_unknown_tool", {"tool": tool_name}, trace_id=trace_id)
                tool_result = f"Error: Unknown tool '{tool_name}'"

            tool_messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result,
            })

        # ---- 5. Follow-up LLM call: turn tool results into a final answer ----
        try:
            followup = _get_client().chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    *messages,
                    {"role": "assistant", "tool_calls": message.tool_calls},
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


# -------------------- MAIN ENTRY --------------------

def run_llm_with_tools(user_id: str, user_input: str, trace_id: str = None) -> str:
    """Discover MCP tools at runtime and execute the (MCP-only) tool loop."""
    messages = build_messages(user_id, user_input)
    return asyncio.run(_run_tool_loop(messages, trace_id))
