from __future__ import annotations

from collections.abc import Iterator
from openai import OpenAI
import json

from agentforge.config import OPENAI_MODEL, OPENAI_BASE_URL, HISTORY_TOKEN_BUDGET
from agentforge.tools import (
    run_llm_with_tools,
    TOOLS_SCHEMA,
    execute_tool,
    tool_catalog_for_classifier,
)
from agentforge.memory.semantic import (
    load_memory,
    save_memory,
    get_embedding,
    store_memory,
)
from agentforge.memory.response import answer_with_memory
from agentforge.rag.qa import answer_from_docs
from agentforge.logger import log_event, generate_trace_id, Span, log_token_usage
from agentforge.reasoning.react_engine import react_loop
from agentforge.conversation import trim_history, count_tokens

_client = None  # created on first API call, not at import time

def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(base_url=OPENAI_BASE_URL) if OPENAI_BASE_URL else OpenAI()
    return _client


def simple_llm_answer(user_input: str, trace_id: str = None) -> str:
    """Non-streaming LLM call. Returns the full response as a single string."""
    response = _get_client().chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": user_input}],
    )
    log_token_usage(response, "simple_llm_answer", trace_id=trace_id)
    return response.choices[0].message.content


def stream_llm_answer(user_input: str, trace_id: str = None) -> Iterator[str]:
    """Stream a basic LLM response token by token.

    Uses stream=True so the API returns one ChatCompletionChunk per token
    instead of waiting for the full response. This function is a Python
    generator — callers iterate over it with `for token in stream_llm_answer(...)`.

    Each chunk's delta.content is one token (or None on the final sentinel
    chunk, which we skip with the `if token` guard).
    """
    response = _get_client().chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": user_input}],
        stream=True,
        stream_options={"include_usage": True},
    )
    token_count = 0
    for chunk in response:
        if chunk.usage:
            log_token_usage(chunk, "stream_llm_answer", trace_id=trace_id)
        if not chunk.choices:
            continue
        token = chunk.choices[0].delta.content
        if token:
            token_count += 1
            yield token
    import logging
    logging.getLogger(__name__).debug("stream_llm_answer: yielded %d tokens.", token_count)


def run_react_agent(user_id: str, user_input: str, max_steps: int = 5) -> str:
    """Delegates to the memory-aware ReAct loop."""
    return react_loop(user_id, user_input, max_steps)

VALID_INTENTS = frozenset({"REMEMBER", "ACT", "REACT", "ANSWER", "IGNORE", "RESPOND_WITH_MEMORY", "DOCS_QA"})


def run_agent(
    user_id: str,
    session_id: str,
    user_input: str,
    history: list[dict] = None,
    stream: bool = False,
    trace_id: str = None,
) -> str | Iterator[str]:
    # Trim history to the configured token budget before any LLM call.
    # We trim here — once, at the entry point — so every pipeline (ANSWER,
    # DOCS_QA, etc.) automatically receives a history that fits the budget.
    # The caller's original list is NOT mutated; trim_history returns a new list.
    tid = trace_id or generate_trace_id()
    log_event("trace_start", {"user_input": user_input, "user_id": user_id}, trace_id=tid)

    safe_history = trim_history(history or [], HISTORY_TOKEN_BUDGET)
    log_event("history_trimmed", {
        "original_turns": len(history) // 2 if history else 0,
        "kept_turns": len(safe_history) // 2,
        "estimated_tokens": count_tokens(safe_history),
        "budget": HISTORY_TOKEN_BUDGET,
    }, trace_id=tid)

    try:
        with Span("intent_classification", trace_id=tid) as span:
            intent_data = classify_intent(user_input, trace_id=tid)
            span.payload = {
                "intent": intent_data["intent"],
                "memory_candidate": intent_data["memory_candidate"],
                "reason": intent_data["reason"],
            }
    except Exception:
        return "Something went wrong while understanding your request. Please try again."

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
        log_event("trace_end", {"intent": intent}, trace_id=tid)
        return "Got it 👍 I'll remember that."

    # -------------------------------
    # ACT (Single tool execution)
    # -------------------------------
    if intent == "ACT":
        with Span("act_tool_pipeline", trace_id=tid) as span:
            tool_output = run_llm_with_tools(user_id, user_input, trace_id=tid)
            try:
                tool_output = json.loads(tool_output)
            except json.JSONDecodeError:
                span.payload = {"error": "invalid_json"}
                log_event("trace_end", {"intent": intent, "error": True}, trace_id=tid)
                return "Agent error: invalid tool response."

            reply = tool_output.get("reply", "")
            store = tool_output.get("store_memory", False)
            memory_text = tool_output.get("memory_text", "")
            span.payload = {"reply_length": len(reply)}

        if store and memory_text:
            store_memory(user_id, memory_text)

        log_event("trace_end", {"intent": intent}, trace_id=tid)
        return reply

    # -------------------------------
    # REACT (Multi-step reasoning + tools)
    # -------------------------------
    if intent == "REACT":
        with Span("react_pipeline", trace_id=tid) as span:
            result = run_react_agent(user_id, user_input)
            span.payload = {"reply_length": len(result)}
        log_event("trace_end", {"intent": intent}, trace_id=tid)
        return result

    # -------------------------------
    # DOCS_QA (RAG — answer from ingested documents)
    # -------------------------------
    if intent == "DOCS_QA":
        log_event("pipeline_start", {"pipeline": "docs_qa"}, trace_id=tid)
        return answer_from_docs(user_input, history=safe_history, stream=stream, trace_id=tid)

    # -------------------------------
    # ANSWER / MEMORY-AWARE
    # -------------------------------
    if intent in ("ANSWER", "RESPOND_WITH_MEMORY"):
        log_event("pipeline_start", {"pipeline": "answer_with_memory"}, trace_id=tid)
        return answer_with_memory(user_id, user_input, history=safe_history, stream=stream, trace_id=tid)

    # -------------------------------
    # IGNORE
    # -------------------------------
    log_event("trace_end", {"intent": intent}, trace_id=tid)
    return "Okay 🙂"


# -------------------------------
# Intent Classification (Phase 3.2)
# -------------------------------

def classify_intent(user_input: str, trace_id: str = None) -> dict:
    # Prompt no longer needs "return ONLY valid JSON" instructions —
    # response_format={"type": "json_object"} enforces that at the API level.
    # The prompt still describes the expected keys and value constraints because
    # structured output guarantees valid JSON but NOT the right shape or values.
    # The available tools are injected dynamically from the tool registry so
    # adding/removing a tool updates the classifier with no prompt edit needed.
    available_tools = tool_catalog_for_classifier()
    prompt = f"""
You are an intent classifier for an AI agent.

Classify the user's intent into ONE of the following:
- REMEMBER → stable personal preference or fact
- ACT → user wants live, current, real-time, or external information (or a deterministic
  action) that one of the registered tools below can provide. Route here whenever the
  honest answer would be "the model can't know that without an external lookup".
  Available tools:
{available_tools}
  ACT examples:
  - "what's the weather in Tokyo" → get_weather (live data, model doesn't know)
  - "latest news about OpenAI" → get_top_news (current events)
  - "who is Ada Lovelace" → wikipedia_lookup (factual entity lookup)
- REACT → complex tasks requiring reasoning, planning, decomposition, or multiple steps
  Examples: planning a trip, scheduling tasks, comparing options, multi-step problem solving
- ANSWER → informational response answerable from the model's own knowledge
  (history, definitions, explanations). Do NOT use ANSWER if the query needs
  current/real-time data — use ACT instead.
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

User input:
"{user_input}"

Respond with a JSON object with exactly these keys:
  "intent"           — one of the intent labels above
  "memory_candidate" — plain text string, or empty string if nothing to save
  "reason"           — one sentence explaining the classification
"""

    try:
        response = _get_client().chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            # Constrained decoding: the API enforces valid JSON at the token level.
            # json.loads will never throw a JSONDecodeError on this response.
            # Key/value validation still happens below — response_format only
            # guarantees structure, not the correct keys or intent values.
            response_format={"type": "json_object"},
        )
        log_token_usage(response, "intent_classification", trace_id=trace_id)
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
            result = run_agent(user_id, session_id, user_input, history=history, stream=True)

            if isinstance(result, str):
                log_event("final_answer", {"text": result, "streamed": False})
                print("Agent:", result)
            else:
                print("Agent: ", end="", flush=True)
                full_text = ""
                for token in result:
                    print(token, end="", flush=True)
                    full_text += token
                print()
                result = full_text
                log_event("final_answer", {"text": result, "streamed": True})

            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": result})
        except Exception as e:
            print("Error:", e)
