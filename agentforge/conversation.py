"""
conversation.py — Context window management and conversation-aware query processing.

Two responsibilities:
  1. Token-count truncation: keep history within a budget before LLM calls.
  2. Query rewriting: rewrite follow-up questions into standalone queries so
     RAG retrieval works correctly even in multi-turn conversations.

Token estimation strategy:
  - 4 chars ≈ 1 token heuristic (no extra dependency).
  - Remove the OLDEST user+assistant pair first (least relevant).
  - Always keep at least the most recent turn (last 2 messages).

Upgrade path for count_tokens(): replace char estimate with tiktoken:
    import tiktoken
    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    total += len(enc.encode(content))
"""

import logging

from openai import OpenAI
from agentforge.config import OPENAI_MODEL, OPENAI_BASE_URL
from agentforge.logger import log_token_usage

client = OpenAI(base_url=OPENAI_BASE_URL) if OPENAI_BASE_URL else OpenAI()
logger = logging.getLogger(__name__)


# ---------- Token estimation ----------

def count_tokens(messages: list[dict]) -> int:
    """
    Estimate the total token count for a list of role/content message dicts.

    Uses the 4 chars ≈ 1 token heuristic plus 4 tokens of overhead per message
    (for role label and formatting that the API adds internally).

    This is intentionally an over-estimate: being slightly conservative means
    we trim a little more aggressively, which is the safer direction (lower cost,
    never hitting the context window limit).

    Args:
        messages: List of {"role": ..., "content": ...} dicts.

    Returns:
        Estimated total token count across all messages.
    """
    total = 0
    for msg in messages:
        content = msg.get("content") or ""
        total += len(content) // 4   # content tokens (4 chars ≈ 1 token)
        total += 4                   # per-message overhead for role + formatting
    return total


# ---------- Trimming ----------

def trim_history(history: list[dict], budget: int) -> list[dict]:
    """
    Return a copy of history trimmed so its estimated token count is within budget.

    Removes oldest user+assistant PAIRS first (2 messages at a time) to keep
    the history coherent — a dangling assistant message with no preceding user
    message would confuse the model.

    Always keeps at least the last 2 messages (the most recent user+assistant
    turn) so the model always has immediate context, even in extreme cases.

    Args:
        history: Full conversation history as a list of role/content dicts.
        budget:  Maximum estimated tokens allowed (from HISTORY_TOKEN_BUDGET).

    Returns:
        A new list (the original is not mutated) that fits within the budget.
    """
    if not history:
        return []

    trimmed = list(history)  # work on a copy — never mutate the caller's list
    original_len = len(trimmed)

    # Remove oldest pairs until we fit within budget or only 1 pair remains
    while count_tokens(trimmed) > budget and len(trimmed) > 2:
        trimmed = trimmed[2:]  # drop the oldest user+assistant pair

    trimmed_count = original_len - len(trimmed)
    if trimmed_count > 0:
        logger.info(
            "trim_history: dropped %d messages (%d turns) to fit within %d token budget. "
            "Remaining: %d messages (~%d tokens).",
            trimmed_count,
            trimmed_count // 2,
            budget,
            len(trimmed),
            count_tokens(trimmed),
        )
    else:
        logger.debug(
            "trim_history: history fits within budget. %d messages (~%d tokens).",
            len(trimmed),
            count_tokens(trimmed),
        )

    return trimmed


# ---------- Query rewriting ----------

def rewrite_query(user_input: str, history: list[dict], trace_id: str = None) -> str:
    """
    Rewrite a potentially ambiguous follow-up question into a standalone query
    suitable for RAG retrieval.

    Problem: embedding a message like "How does it handle citations?" produces
    a poor embedding because "it" has no meaning without the conversation.
    The retrieval step runs before the LLM sees history, so it retrieves
    irrelevant chunks.

    Solution: a small LLM call that resolves pronouns and references using the
    conversation history, producing a self-contained query like
    "How does RAG handle citations?" that embeds and retrieves correctly.

    Design decisions:
    - Only called when history is non-empty (turn 1 is always self-contained).
    - Falls back to the original user_input on any failure so retrieval always
      runs — a rewrite failure should never block the pipeline.
    - The rewritten query is used ONLY for retrieval (search_docs). The original
      user_input is used in the final generation prompt so the user's exact
      wording is preserved and the agent doesn't put words in their mouth.

    Args:
        user_input: The user's latest message (may contain pronouns/references).
        history:    Conversation history as a list of role/content dicts.

    Returns:
        A standalone question string. Falls back to user_input on error.
    """
    if not history:
        # No history means nothing to resolve — original query is already standalone.
        logger.debug("rewrite_query: no history, returning original query.")
        return user_input

    # Format history as readable text for the rewrite prompt
    history_text = "\n".join(
        f"{msg['role'].capitalize()}: {msg['content']}"
        for msg in history
    )

    prompt = f"""You are a query rewriter for a retrieval system.

Given a conversation history and the user's latest message, rewrite the latest \
message as a complete, standalone question that can be understood without the conversation.

Rules:
1. If the message is already self-contained, return it unchanged.
2. Do NOT answer the question. Only rewrite it.
3. Return ONLY the rewritten question — no explanation, no quotes, no punctuation changes.

Conversation history:
{history_text}

Latest message: "{user_input}"

Standalone question:"""

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=128,   # rewrites are always short
            temperature=0,    # deterministic — same input should give same rewrite
        )
        log_token_usage(response, "query_rewrite", trace_id=trace_id)
        rewritten = (response.choices[0].message.content or "").strip()

        if not rewritten:
            logger.warning("rewrite_query: LLM returned empty rewrite, using original.")
            return user_input

        logger.info(
            "rewrite_query: original=%r  rewritten=%r",
            user_input,
            rewritten,
        )
        return rewritten

    except Exception as exc:
        # Never block retrieval because the rewrite failed.
        logger.warning("rewrite_query: failed (%s), falling back to original query.", exc)
        return user_input
