# agent/reasoning/react_engine.py

import asyncio
import json
import logging
from openai import OpenAI

from agentforge.config import OPENAI_MODEL, OPENAI_BASE_URL
from agentforge.prompts import build_prompt, render_tool_catalog, OUTPUT_SCHEMA
from agentforge.mcp_client import mcp_gateway
from agentforge.memory.semantic import get_relevant_memories, store_memory
from agentforge.logger import log_event, log_token_usage

_client = None  # created on first API call, not at import time

def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(base_url=OPENAI_BASE_URL) if OPENAI_BASE_URL else OpenAI()
    return _client

logger = logging.getLogger(__name__)


def react_loop(user_id: str, user_input: str, max_steps: int = 5) -> str:
    """Core ReAct reasoning loop (think → act → observe → repeat).

    Public entry point. The signature is unchanged from before Step 17c.2, so
    callers (main.py) are unaffected; internally it now runs an async loop because
    tools are reached over MCP. One MCP gateway is opened for the whole turn (see
    ``_react_loop_async``).
    """
    return asyncio.run(_react_loop_async(user_id, user_input, max_steps))


async def _react_loop_async(user_id: str, user_input: str, max_steps: int = 5) -> str:
    """Async ReAct loop with tools served over MCP (Step 17c.2).

    What changed from the pre-MCP loop:
    - Tools are no longer dispatched in-process via a local registry. We open one
      shared MCP gateway for the whole turn and call ``gw.call(...)`` for each
      tool step, so every reasoning step reuses the same warm server sessions.
    - The available tools are discovered at runtime and rendered into the prompt
      (render_tool_catalog), instead of a hardcoded tool name. The agent has zero
      hardcoded tools — it can only call what the servers actually expose.

    Each step uses response_format={"type": "json_object"} (constrained decoding)
    so the API guarantees valid JSON on every step. A single JSONDecodeError
    anywhere in the loop kills the whole reasoning chain — there is no partial
    recovery mid-loop — so the json.JSONDecodeError guard is kept as a last resort.
    Shape validation (action.get("type") etc.) stays too: valid JSON does not mean
    correct keys or values.
    """
    memory_chunks = get_relevant_memories(user_id, user_input)
    log_event("react_start", {"user_input": user_input, "max_steps": max_steps})

    async with mcp_gateway() as gw:
        # Build the prompt once, injecting the live tool catalog so the model can
        # only reference tools that are actually served this turn.
        messages = build_prompt(user_input, memory_chunks)
        messages.append({"role": "system", "content": render_tool_catalog(gw.catalog)})
        messages.append({"role": "system", "content": OUTPUT_SCHEMA})
        log_event("react_tools_discovered", {"tools": [t["name"] for t in gw.catalog]})

        for step in range(max_steps):
            try:
                response = _get_client().chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=messages,
                    # Constrained decoding: same as classify_intent (Step 6).
                    # The ReAct schema is more complex (nested action object,
                    # multiple fields) so parse failures are more likely without
                    # this — and more costly, since they abort a reasoning chain.
                    response_format={"type": "json_object"},
                )
                log_token_usage(response, f"react_step_{step + 1}")
                raw = response.choices[0].message.content
            except Exception as e:
                logger.error("react_loop: LLM call failed on step %d: %s", step + 1, e, exc_info=True)
                return "I ran into an error during planning. Please try again."

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                # Should be extremely rare now that response_format is set.
                # Kept as a safety net for empty or malformed responses.
                logger.warning("react_loop: JSONDecodeError on step %d despite response_format.", step + 1)
                return "Agent error: Invalid JSON response."

            thought = data.get("thought", "")
            action = data.get("action", {})

            print(f"\n[THOUGHT {step+1}] {thought}")
            log_event("react_step", {
                "step": step + 1,
                "thought": thought,
                "action_type": action.get("type"),
                "tool_name": action.get("tool_name"),
            })

            # ---------- FINAL ----------
            if action.get("type") == "final":
                if data.get("store_memory") and data.get("memory_text"):
                    store_memory(user_id, data["memory_text"])

                reply = data.get("reply", "")
                log_event("react_end", {"steps_taken": step + 1, "reply_length": len(reply)})
                return reply

            # ---------- TOOL ----------
            if action.get("type") == "tool":
                tool_name = action.get("tool_name")
                tool_input = action.get("tool_input", {})

                # Dispatch over MCP. gw.call returns a readable error string for
                # unknown/hallucinated tools (it does not raise), so the model can
                # observe the error and recover on the next step. gw.call also
                # wraps the tool output as untrusted data with this turn's nonce
                # (Step 17e) — the spotlight rule in the prompt tells the model to
                # treat anything inside those markers as data, never instructions.
                observation = await gw.call(tool_name, tool_input)

                messages.append({
                    "role": "assistant",
                    "content": raw
                })

                messages.append({
                    "role": "tool",
                    "name": tool_name,
                    "content": observation
                })

        log_event("react_end", {"steps_taken": max_steps, "reply_length": 0, "stopped": "max_steps"})
        return "Agent stopped: too many reasoning steps."
