"""Tool registry and orchestration.

Each tool lives in its own module under ``agentforge.tools`` and must
export two names:

- ``TOOL_FUNCTION`` — the callable, signature ``(**kwargs) -> str``
- ``TOOL_SCHEMA``   — the OpenAI function-calling schema for the tool

To register a new tool, create a module under ``agentforge.tools`` and
append it to ``TOOL_MODULES`` below. ``TOOL_REGISTRY`` and
``TOOLS_SCHEMA`` are built automatically.

MCP tools are discovered at runtime (Step 17b): the agent connects to
each server in ``MCP_SERVERS``, calls ``tools/list``, and merges the
returned schemas with ``TOOLS_SCHEMA`` before the first LLM call.
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


# -------------------- TOOL REGISTRY --------------------
# Register every tool module here. Each must export TOOL_FUNCTION + TOOL_SCHEMA.
TOOL_MODULES = [wikipedia, weather, news]

TOOL_REGISTRY: Dict[str, Callable] = {
    m.TOOL_FUNCTION.__name__: m.TOOL_FUNCTION for m in TOOL_MODULES
}

TOOLS_SCHEMA = [m.TOOL_SCHEMA for m in TOOL_MODULES]


def tool_catalog_for_classifier() -> str:
    """Render the registered tools as a bullet list for the intent classifier.

    Single source of truth: tool name + description live in each tool module's
    TOOL_SCHEMA. The classifier reads from here, so adding a tool to
    TOOL_MODULES updates the classifier prompt automatically — no drift.
    """
    lines = []
    for m in TOOL_MODULES:
        fn = m.TOOL_SCHEMA["function"]
        lines.append(f"    - {fn['name']}: {fn['description']}")
    return "\n".join(lines)


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

        # ---- 2. Merge local + MCP schemas ----
        all_schemas = TOOLS_SCHEMA + mcp_schemas

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
                tool_result = execute_tool(tool_name, arguments)

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
    """Discover MCP tools, merge with local schema, execute the tool loop."""
    messages = build_messages(user_id, user_input)
    return asyncio.run(_run_tool_loop(messages, trace_id))
