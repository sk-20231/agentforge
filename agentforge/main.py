from __future__ import annotations

from collections.abc import Iterator
from openai import OpenAI
import json

from agentforge.config import (
    OPENAI_MODEL,
    OPENAI_BASE_URL,
    HISTORY_TOKEN_BUDGET,
    AGENT_INPUT_GUARDRAIL_ENABLED,
    AGENT_OUTPUT_GUARDRAIL_ENABLED,
)
from agentforge import guardrail
from agentforge import output_guardrail
from agentforge.tools import (
    resume_tool_loop,
    run_llm_with_tools,
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
from agentforge.reasoning.react_engine import react_loop, resume_react_loop
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


def run_react_agent(user_id: str, user_input: str, max_steps: int = 5,
                    approval_handler=None, trace_id: str = None) -> str:
    """Delegates to the memory-aware ReAct loop."""
    return react_loop(user_id, user_input, max_steps,
                      approval_handler=approval_handler, trace_id=trace_id)

VALID_INTENTS = frozenset({"REMEMBER", "ACT", "REACT", "ANSWER", "IGNORE", "RESPOND_WITH_MEMORY", "DOCS_QA"})


def _input_guardrail_block(user_input: str, trace_id: str = None) -> str | None:
    """Scan user input for prompt-injection/jailbreak (issue #22 — INPUT placement point).

    Returns a refusal STRING when the turn must be refused (the classifier flagged the
    input), else None to proceed. UNAVAILABLE fails OPEN (proceed) — consistent with
    the gateway's tool-output guardrail default: the input classifier going down must
    not brick the agent (the model + downstream guards remain). This is the *direct*-
    injection guard; because the user is also the legitimate command channel it false-
    positives on role-play, so it is threshold-tunable (AGENT_GUARDRAIL_THRESHOLD) and
    toggleable (AGENT_INPUT_GUARDRAIL_ENABLED). Mirrors the gateway's _guardrail_block.
    """
    if not AGENT_INPUT_GUARDRAIL_ENABLED:
        return None
    result = guardrail.scan_external_text(user_input, trace_id=trace_id)
    if result.verdict == guardrail.Verdict.BLOCK:
        log_event("input_guardrail_blocked",
                  {"reason": result.reason, "score": result.score}, trace_id=trace_id)
        return ("I couldn't process that message — it looked like an attempt to override "
                "the assistant's instructions. Please rephrase your request.")
    if result.verdict == guardrail.Verdict.UNAVAILABLE:
        # Loud-but-cheap: one audited line; the scanner itself logs once per process
        # when it can't load. Proceed (fail-open).
        log_event("input_guardrail_unavailable", {"reason": result.reason}, trace_id=trace_id)
    return None


def _scan_output(text, trace_id: str = None):
    """Redact structured PII from an outgoing reply (issue #22 — OUTPUT placement point).

    No-op when disabled or the text is empty. On redaction, logs the PII TYPES + count
    (never the values — that would re-leak the PII into the log) and returns the
    redacted text. Applied to run_agent's NON-streaming string returns; scanning
    streamed output can't un-send tokens and is a documented follow-up.
    """
    if not AGENT_OUTPUT_GUARDRAIL_ENABLED or not isinstance(text, str) or not text:
        return text
    result = output_guardrail.scan_output(text)
    if result.found:
        log_event("output_guardrail_redacted",
                  {"types": result.types, "count": result.count}, trace_id=trace_id)
        return result.redacted_text
    return text


def run_agent(
    user_id: str,
    session_id: str,
    user_input: str,
    history: list[dict] = None,
    stream: bool = False,
    trace_id: str = None,
    approval_handler=None,
) -> str | Iterator[str]:
    # approval_handler (Step 17f): the front-end's human-in-the-loop callback for
    # high-impact tool calls — see agentforge.approval. Threaded through to the
    # MCP gateway by the ACT and REACT pipelines (the only tool-calling paths).
    # None (the default, and every pre-17f caller) means gated tools are denied
    # with a readable observation rather than silently auto-approved. If the
    # handler raises ApprovalRequired it propagates out of this function by
    # design — carrying the pipeline's continuation — and the front-end asks
    # the human, then calls resume_agent(exc, decision) to finish the turn.
    # Trim history to the configured token budget before any LLM call.
    # We trim here — once, at the entry point — so every pipeline (ANSWER,
    # DOCS_QA, etc.) automatically receives a history that fits the budget.
    # The caller's original list is NOT mutated; trim_history returns a new list.
    tid = trace_id or generate_trace_id()
    log_event("trace_start", {"user_input": user_input, "user_id": user_id}, trace_id=tid)

    # INPUT guardrail (issue #22): scan the user's message for prompt-injection /
    # jailbreak BEFORE any classification or routing. A flagged message short-circuits
    # the turn with a refusal — we never classify, retrieve, or call a tool on it.
    refusal = _input_guardrail_block(user_input, trace_id=tid)
    if refusal is not None:
        log_event("trace_end", {"intent": "INPUT_BLOCKED"}, trace_id=tid)
        return refusal

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
            tool_output = run_llm_with_tools(user_id, user_input, trace_id=tid,
                                             approval_handler=approval_handler)
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
        return _scan_output(reply, tid)

    # -------------------------------
    # REACT (Multi-step reasoning + tools)
    # -------------------------------
    if intent == "REACT":
        with Span("react_pipeline", trace_id=tid) as span:
            # Issue #7: REACT was the only pipeline run_agent didn't hand the
            # trace ID to, so its token/step/audit logs were orphaned from the
            # turn and per-trace cost reports undercounted REACT turns.
            result = run_react_agent(user_id, user_input,
                                     approval_handler=approval_handler,
                                     trace_id=tid)
            span.payload = {"reply_length": len(result)}
        log_event("trace_end", {"intent": intent}, trace_id=tid)
        return _scan_output(result, tid)

    # -------------------------------
    # DOCS_QA (RAG — answer from ingested documents)
    # -------------------------------
    if intent == "DOCS_QA":
        log_event("pipeline_start", {"pipeline": "docs_qa"}, trace_id=tid)
        result = answer_from_docs(user_input, history=safe_history, stream=stream, trace_id=tid)
        # Output guardrail on the non-streaming return; a streamed Iterator passes
        # through (post-hoc scanning of streamed tokens is a documented follow-up).
        return _scan_output(result, tid) if isinstance(result, str) else result

    # -------------------------------
    # ANSWER / MEMORY-AWARE
    # -------------------------------
    if intent in ("ANSWER", "RESPOND_WITH_MEMORY"):
        log_event("pipeline_start", {"pipeline": "answer_with_memory"}, trace_id=tid)
        result = answer_with_memory(user_id, user_input, history=safe_history, stream=stream, trace_id=tid)
        return _scan_output(result, tid) if isinstance(result, str) else result

    # -------------------------------
    # IGNORE
    # -------------------------------
    log_event("trace_end", {"intent": intent}, trace_id=tid)
    return "Okay 🙂"


def resume_agent(interrupt, decision, approval_handler=None) -> str:
    """Resume a turn that was interrupted for human approval (Step 17f).

    ``interrupt`` is the ApprovalRequired the front-end caught from run_agent;
    its ``continuation`` (attached by the pipeline loop as the exception
    unwound) tells us which pipeline to re-enter and carries its frozen state.
    ``decision`` is the human's choice for ``interrupt.request``: False (deny),
    True (allow once), or ``approval.APPROVE_TURN`` (allow + grant the same
    tool for the rest of this turn — issue #6; the gateway records the grant
    in the turn's ``granted`` set, carried by the continuation). The decision
    settles ONLY that stored call (one-shot); any new gated call the resumed
    turn makes goes back through ``approval_handler`` and can interrupt
    again — each interrupt gets its own decision.

    No intent re-classification happens here — the original classification is
    part of the frozen turn. That (plus not re-running earlier loop steps) is
    why resume is both cheaper and more correct than replaying the turn:
    a replay would ask the non-deterministic LLM to regenerate the approved
    call and hope it matches.
    """
    cont = getattr(interrupt, "continuation", None)
    if not cont:
        raise ValueError("This interrupt carries no continuation and cannot be resumed.")
    tid = cont.get("trace_id")
    log_event("resume_start", {
        "pipeline": cont["pipeline"],
        "tool": interrupt.request.tool,
        "decision": "approved" if decision else "denied",
    }, trace_id=tid)

    if cont["pipeline"] == "react":
        with Span("react_pipeline_resume", trace_id=tid) as span:
            result = resume_react_loop(interrupt, decision,
                                       approval_handler=approval_handler)
            span.payload = {"reply_length": len(result)}
        log_event("trace_end", {"intent": "REACT", "resumed": True}, trace_id=tid)
        return result

    if cont["pipeline"] == "act":
        # Same post-processing contract as run_agent's ACT branch: the loop
        # returns a JSON string with reply / store_memory / memory_text.
        with Span("act_tool_pipeline_resume", trace_id=tid) as span:
            tool_output = resume_tool_loop(interrupt, decision,
                                           approval_handler=approval_handler)
            try:
                parsed = json.loads(tool_output)
            except json.JSONDecodeError:
                span.payload = {"error": "invalid_json"}
                log_event("trace_end", {"intent": "ACT", "resumed": True, "error": True},
                          trace_id=tid)
                return "Agent error: invalid tool response."
            reply = parsed.get("reply", "")
            span.payload = {"reply_length": len(reply)}
        if parsed.get("store_memory") and parsed.get("memory_text"):
            store_memory(cont["user_id"], parsed["memory_text"])
        log_event("trace_end", {"intent": "ACT", "resumed": True}, trace_id=tid)
        return reply

    raise ValueError(f"Unknown pipeline in continuation: {cont['pipeline']!r}")


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
  - "who is Ada Lovelace" → search_wikipedia (factual entity lookup)
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
