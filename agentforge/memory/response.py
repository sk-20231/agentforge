from __future__ import annotations

import logging
from collections.abc import Iterator

from openai import OpenAI

from agentforge.config import OPENAI_MODEL, OPENAI_BASE_URL
from agentforge.memory.semantic import load_memory
from agentforge.logger import log_token_usage

_client = None  # created on first API call, not at import time

def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(base_url=OPENAI_BASE_URL) if OPENAI_BASE_URL else OpenAI()
    return _client

logger = logging.getLogger(__name__)


def answer_with_memory(
    user_id: str,
    user_input: str,
    history: list[dict] = None,
    stream: bool = False,
    trace_id: str = None,
) -> str | Iterator[str]:
    """Answer a question using the user's stored memory and conversation history.

    Args:
        user_id:  Identifies whose semantic memory to load.
        user_input: The user's question.
        history:  Conversation history (role/content dicts) for multi-turn context.
        stream:   If True, returns an Iterator[str] that yields tokens one by one
                  (stream=True on the OpenAI call). If False (default), returns
                  the full response as a str — fully backward compatible.
    """
    try:
        memory = load_memory(user_id)
    except Exception:
        return "I couldn't load your saved information. Please try again."

    if not memory:
        memory_text = "No stored personal information yet."
    else:
        memory_text = "\n".join(f"- {item['text']}" for item in memory)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Use the user's stored personal "
                "information to answer their question in a natural way. "
                "Do not invent facts."
            ),
        },
        {
            "role": "system",
            "content": f"Known information about the user:\n{memory_text}",
        },
    ]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_input})

    if stream:
        logger.debug("answer_with_memory: starting streaming response for user=%s.", user_id)
        return _stream_tokens(messages, trace_id=trace_id)

    try:
        response = _get_client().chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
        )
        log_token_usage(response, "memory_answer", trace_id=trace_id)
        return response.choices[0].message.content
    except Exception:
        return "I ran into an error answering your question. Please try again."


def _stream_tokens(messages: list[dict], trace_id: str = None) -> Iterator[str]:
    """Internal generator: streams tokens from the OpenAI API.

    Separated from answer_with_memory so the error-handling is clean:
    if the API call itself fails, the exception propagates to the CLI which
    catches it; if a mid-stream chunk is malformed, we skip it and continue.
    """
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        stream=True,
        stream_options={"include_usage": True},
    )
    token_count = 0
    for chunk in response:
        if chunk.usage:
            log_token_usage(chunk, "memory_answer_stream", trace_id=trace_id)
        if not chunk.choices:
            continue
        token = chunk.choices[0].delta.content
        if token:
            token_count += 1
            yield token
    logger.debug("_stream_tokens: streamed %d tokens.", token_count)
