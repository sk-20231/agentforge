from openai import OpenAI
import json

from agentforge.config import OPENAI_MODEL, OPENAI_BASE_URL, HISTORY_TOKEN_BUDGET
from agentforge.tools import run_llm_with_tools, TOOLS_SCHEMA, execute_tool
from agentforge.memory.semantic import (
    load_memory,
    save_memory,
    get_embedding,
    store_memory,
)
from agentforge.memory.response import answer_with_memory
from agentforge.rag.qa import answer_from_docs
from agentforge.logger import log_event
from agentforge.reasoning.react_engine import react_loop
from agentforge.conversation import trim_history, count_tokens

client = OpenAI(base_url=OPENAI_BASE_URL) if OPENAI_BASE_URL else OpenAI()


def simple_llm_answer(user_input: str) -> str:
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "user", "content": user_input}
        ]
    )

    return response.choices[0].message.content


def run_react_agent(user_id: str, user_input: str, max_steps: int = 5) -> str:
    """Delegates to the memory-aware ReAct loop."""
    return react_loop(user_id, user_input, max_steps)

VALID_INTENTS = frozenset({"REMEMBER", "ACT", "REACT", "ANSWER", "IGNORE", "RESPOND_WITH_MEMORY", "DOCS_QA"})


def run_agent(user_id: str, session_id: str, user_input: str, history: list[dict] = None) -> str:
    # Trim history to the configured token budget before any LLM call.
    # We trim here — once, at the entry point — so every pipeline (ANSWER,
    # DOCS_QA, etc.) automatically receives a history that fits the budget.
    # The caller's original list is NOT mutated; trim_history returns a new list.
    safe_history = trim_history(history or [], HISTORY_TOKEN_BUDGET)
    log_event("history_trimmed", {
        "original_turns": len(history) // 2 if history else 0,
        "kept_turns": len(safe_history) // 2,
        "estimated_tokens": count_tokens(safe_history),
        "budget": HISTORY_TOKEN_BUDGET,
    })

    try:
        intent_data = classify_intent(user_input)
    except Exception:
        return "Something went wrong while understanding your request. Please try again."
    log_event("intent_classification", {"intent": intent_data["intent"], "memory_candidate": intent_data["memory_candidate"], "reason": intent_data["reason"]})

    intent = intent_data["intent"]
    memory_candidate = intent_data["memory_candidate"]
    reason = intent_data["reason"]

    print(f"\n[Intent] {intent} — {reason}")

    # -------------------------------
    # REMEMBER
    # -------------------------------
    if intent == "REMEMBER":
        if memory_candidate:
            if isinstance(memory_candidate, dict):
                parts = []
                for key, value in memory_candidate.items():
                    if isinstance(value, list):
                        value = ", ".join(value)
                    parts.append(f"{key}: {value}")
                memory_candidate = ". ".join(parts)
            store_memory(user_id, memory_candidate)
        return "Got it 👍 I'll remember that."

    # -------------------------------
    # ACT (Single tool execution)
    # -------------------------------
    if intent == "ACT":
        tool_output = run_llm_with_tools(user_id, user_input)

        try:
            tool_output = json.loads(tool_output)
        except json.JSONDecodeError:
            return "Agent error: invalid tool response."

        reply = tool_output.get("reply", "")
        store = tool_output.get("store_memory", False)
        memory_text = tool_output.get("memory_text", "")

        if store and memory_text:
            store_memory(user_id, memory_text)

        return reply

    # -------------------------------
    # REACT (Multi-step reasoning + tools)
    # -------------------------------
    if intent == "REACT":
        return run_react_agent(user_id, user_input)

    # -------------------------------
    # DOCS_QA (RAG — answer from ingested documents)
    # -------------------------------
    if intent == "DOCS_QA":
        return answer_from_docs(user_input, history=safe_history)

    # -------------------------------
    # ANSWER / MEMORY-AWARE
    # -------------------------------
    if intent in ("ANSWER", "RESPOND_WITH_MEMORY"):
        return answer_with_memory(user_id, user_input, history=safe_history)

    # -------------------------------
    # IGNORE
    # -------------------------------
    return "Okay 🙂"


# -------------------------------
# Intent Classification (Phase 3.2)
# -------------------------------

def classify_intent(user_input: str) -> dict:
    prompt = f"""
You are an intent classifier for an AI agent.

Classify the user's intent into ONE of the following:
- REMEMBER → stable personal preference or fact
- ACT → requires a tool or calculation
- REACT → complex tasks requiring reasoning, planning, decomposition, or multiple steps
  Examples: planning a trip, scheduling tasks, comparing options, multi-step problem solving
- ANSWER → informational response only
- IGNORE → greetings or small talk
- RESPOND_WITH_MEMORY → answer should use previously stored personal information
- DOCS_QA → user wants an answer from uploaded/ingested documents (e.g. "what does the document say about X", "according to the docs", "search my files for")

Rules for memory:
- Save ONLY long-term stable facts (likes, dislikes, preferences, traits)
- Do NOT save temporary interests or one-time actions
- memory_candidate MUST be a plain text STRING, not a JSON object
- Combine multiple facts into a single sentence if needed

Examples to SAVE (as plain text strings):
- "I like dogs"
- "I prefer Indian food"
- "I work as a backend engineer"
- "I live in MN, USA. I am a caucasian male. I like playing soccer and playing in a band."

Examples NOT to save:
- I watched a movie yesterday
- I am tired today
- I want pizza right now

Return ONLY valid JSON with memory_candidate as a STRING (not an object).

User input:
"{user_input}"

JSON:
{{
  "intent": "",
  "memory_candidate": "<plain text string or empty string>",
  "reason": ""
}}
"""


    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.choices[0].message.content
        if not raw or not raw.strip():
            return _default_intent(user_input)
    except Exception:
        return _default_intent(user_input)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return _default_intent(user_input)

    intent = data.get("intent")
    if intent not in VALID_INTENTS:
        return _default_intent(user_input)
    memory_candidate = data.get("memory_candidate", "")
    if not isinstance(memory_candidate, str):
        memory_candidate = ""
    reason = data.get("reason", "")
    if not isinstance(reason, str):
        reason = ""
    return {"intent": intent, "memory_candidate": memory_candidate, "reason": reason}


def _default_intent(user_input: str) -> dict:
    """Safe fallback when intent classification fails or returns invalid data."""
    return {"intent": "ANSWER", "memory_candidate": "", "reason": "Classification failed or invalid response."}


# -------------------------------
# CLI
# -------------------------------

if __name__ == "__main__":
    print("🤖 AI Agent (type 'exit' to quit)\n")
    user_id = input("Enter user id: ").strip()
    session_id = "default"
    history: list[dict] = []

    while True:
        user_input = input(f"{user_id}> ")
        if user_input.lower() == "exit":
            break

        try:
            log_event("user_input", {"text": user_input})
            result = run_agent(user_id, session_id, user_input, history=history)
            log_event("final_answer", {"text": result})
            print("Agent:", result)

            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": result})
        except Exception as e:
            print("Error:", e)
