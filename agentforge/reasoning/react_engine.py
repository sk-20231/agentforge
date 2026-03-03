# agent/reasoning/react_engine.py

import json
from openai import OpenAI

from agentforge.config import OPENAI_MODEL, OPENAI_BASE_URL
from agentforge.prompts import build_prompt, OUTPUT_SCHEMA
from agentforge.tools import execute_tool
from agentforge.memory.semantic import get_relevant_memories, store_memory

client = OpenAI(base_url=OPENAI_BASE_URL) if OPENAI_BASE_URL else OpenAI()


def react_loop(user_id: str, user_input: str, max_steps: int = 5) -> str:
    """
    Core ReAct reasoning loop.
    Allows the model to think, act, observe, and repeat.
    """

    memory_chunks = get_relevant_memories(user_id, user_input)

    messages = build_prompt(user_input, memory_chunks)

    for step in range(max_steps):
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages
            )
            raw = response.choices[0].message.content
        except Exception:
            return "I ran into an error during planning. Please try again."

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return "Agent error: Invalid JSON response."

        thought = data.get("thought", "")
        action = data.get("action", {})
        plan = data.get("plan", {})

        print(f"\n[THOUGHT {step+1}] {thought}")

        # ---------- FINAL ----------
        if action.get("type") == "final":
            if data.get("store_memory") and data.get("memory_text"):
                store_memory(user_id, data["memory_text"])

            return data.get("reply", "")

        # ---------- TOOL ----------
        if action.get("type") == "tool":
            tool_name = action.get("tool_name")
            tool_input = action.get("tool_input", {})

            observation = execute_tool(tool_name, tool_input)

            messages.append({
                "role": "assistant",
                "content": raw
            })

            messages.append({
                "role": "tool",
                "name": tool_name,
                "content": observation
            })

    return "Agent stopped: too many reasoning steps."
