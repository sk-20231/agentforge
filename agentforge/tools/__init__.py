"""Tool registry and orchestration.

Each tool lives in its own module under ``agentforge.tools`` and must
export two names:

- ``TOOL_FUNCTION`` — the callable, signature ``(**kwargs) -> str``
- ``TOOL_SCHEMA``   — the OpenAI function-calling schema for the tool

To register a new tool, create a module under ``agentforge.tools`` and
append it to ``TOOL_MODULES`` below. ``TOOL_REGISTRY`` and
``TOOLS_SCHEMA`` are built automatically.
"""
import json
import logging
from typing import Any, Callable, Dict

from openai import OpenAI

from agentforge.config import OPENAI_BASE_URL, OPENAI_MODEL
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


# -------------------- MAIN ENTRY --------------------

def run_llm_with_tools(user_id: str, user_input: str, trace_id: str = None) -> str:
    """Execute the LLM call, handle tool calls, return final model output."""
    try:
        response = _get_client().chat.completions.create(
            model=OPENAI_MODEL,
            messages=build_messages(user_id, user_input),
            tools=TOOLS_SCHEMA,
            # "required" forces the model to call a tool even for trivially simple
            # prompts it could answer mentally. Without this, the model can skip
            # the tool and return reply="" (OUTPUT_SCHEMA requires reply empty
            # when action.type=tool), showing a blank response to the user.
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

    # ---------- TOOL CALL ----------
    if message.tool_calls:
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

            tool_result = execute_tool(tool_name, arguments)
            tool_messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result,
            })

        try:
            followup = _get_client().chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    *build_messages(user_id, user_input),
                    {"role": "assistant", "tool_calls": message.tool_calls},
                    *tool_messages,
                    # Explicit finalise: the model now has the tool result and
                    # must produce a final JSON reply, not another tool call.
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

    # ---------- NO TOOL ----------
    return response.choices[0].message.content
