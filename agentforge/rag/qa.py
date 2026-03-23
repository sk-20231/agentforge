# agent/rag/qa.py
"""
RAG answer pipeline: retrieve relevant chunks, generate an answer with citations,
then validate citations (guardrail). Step 7 of the Document Q&A roadmap.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Iterator

from openai import OpenAI
from agentforge.config import OPENAI_MODEL, OPENAI_BASE_URL
from agentforge.rag.document_store import search_docs
from agentforge.logger import log_event, Span, log_token_usage
from agentforge.conversation import rewrite_query

client = OpenAI(base_url=OPENAI_BASE_URL) if OPENAI_BASE_URL else OpenAI()
logger = logging.getLogger(__name__)


# ---------- Prompt ----------

def _build_prompt(user_input: str, chunks: list[dict]) -> str:
    """
    Build the system + user prompt for RAG generation.
    Each chunk is labeled with its id so the model can cite it.
    """
    context_parts = []
    for c in chunks:
        context_parts.append(f"[{c['id']}] (source: {c['source']})\n{c['text']}")
    context_block = "\n\n---\n\n".join(context_parts)

    return f"""You are a helpful assistant that answers questions using ONLY the provided document chunks.

Rules:
1. Answer the question using ONLY the information in the chunks below.
2. Cite your sources by placing the chunk id in square brackets, e.g. [doc_chunk_0].
3. You may cite multiple chunks.
4. If the chunks do not contain enough information to answer, say so honestly.
5. Do NOT invent or hallucinate information that is not in the chunks.

--- DOCUMENT CHUNKS ---

{context_block}

--- END CHUNKS ---

User question: {user_input}
"""


# ---------- Citation guardrail ----------

def _strip_invalid_citations(answer: str, valid_ids: set[str]) -> str:
    """
    Remove any citation like [some_id] from the answer if some_id was NOT in the
    retrieved chunk set. This is a guardrail: the model should only cite chunks
    it was actually given, not hallucinate citation ids.
    """
    def replacer(match):
        cited_id = match.group(1)
        if cited_id in valid_ids:
            return match.group(0)  # keep valid citation
        return ""  # remove invalid citation

    cleaned = re.sub(r"\[([^\]]+)\]", replacer, answer)
    # Clean up any double spaces left after removal
    cleaned = re.sub(r"  +", " ", cleaned).strip()
    return cleaned


# ---------- Streaming generator ----------

def _stream_rag_tokens(
    messages: list[dict],
    valid_ids: set[str],
    user_input: str,
    trace_id: str = None,
) -> Iterator[str]:
    """Stream RAG generation tokens while buffering for the citation guardrail.

    This is the "stream and collect" pattern:
      - Each token is yielded immediately so the caller (CLI or UI) can print it.
      - Every token is also appended to full_text so the guardrail has the
        complete response after the loop ends.
      - After streaming finishes, the guardrail runs on the full buffer.
        If any invalid citation was removed, the corrected version is compared
        and a log entry records what changed. The caller already received the
        raw tokens — we log the diff for observability but don't send a
        correction back (the next turn will use the clean version in history).

    Why separate from answer_from_docs:
      Same reason as _stream_tokens in memory/response.py — keeping the
      generator isolated makes error handling clean and the main function readable.
    """
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        stream=True,
        stream_options={"include_usage": True},
    )

    full_text = ""
    token_count = 0
    for chunk in response:
        if chunk.usage:
            log_token_usage(chunk, "docs_qa_generate_stream", trace_id=trace_id)
        if not chunk.choices:
            continue
        token = chunk.choices[0].delta.content
        if token:
            full_text += token
            token_count += 1
            yield token

    # Guardrail runs on the complete buffered response after streaming ends.
    clean_answer = _strip_invalid_citations(full_text, valid_ids)
    citations_removed = full_text != clean_answer

    log_event("docs_qa_answer", {
        "query": user_input,
        "raw_answer_length": len(full_text),
        "token_count": token_count,
        "citations_removed": citations_removed,
        "streamed": True,
    }, trace_id=trace_id)

    if citations_removed:
        logger.warning(
            "_stream_rag_tokens: invalid citations stripped from streamed response. "
            "Raw length=%d, clean length=%d.",
            len(full_text),
            len(clean_answer),
        )

    log_event("trace_end", {"intent": "DOCS_QA", "streamed": True}, trace_id=trace_id)


# ---------- Main RAG function ----------

def answer_from_docs(
    user_input: str,
    top_k: int = 5,
    history: list[dict] = None,
    stream: bool = False,
    trace_id: str = None,
) -> str | Iterator[str]:
    """
    Full RAG pipeline: rewrite -> retrieve -> prompt -> generate -> guardrail.

    1. Rewrite the user's query into a standalone question (if history exists)
       so that follow-up questions like "How does it work?" retrieve correctly.
    2. Retrieve the top_k most relevant chunks using the rewritten query.
    3. Build a prompt with the chunks (labeled by id) and conversation history.
    4. Ask the LLM to answer using only those chunks and to cite by id.
    5. Post-process: remove any citation whose id was not in the retrieved set.

    Key design: the rewritten query is used ONLY for retrieval (step 2).
    The original user_input is used in the generation prompt (step 3) so the
    user's exact wording is preserved — the rewrite is invisible to the user.

    Args:
        user_input: The user's question (may contain pronouns or references).
        top_k:      Number of chunks to retrieve.
        history:    Optional conversation history (list of role/content dicts).
        stream:     If True, returns an Iterator[str] that yields tokens as they
                    arrive. The citation guardrail runs after the last token is
                    buffered internally. If False (default), returns a full str.

    Returns:
        The LLM's answer with only valid citations (str), or an Iterator[str]
        that yields tokens one by one when stream=True.
    """
    # 1. Rewrite — resolve follow-up references into a standalone search query.
    with Span("docs_qa_rewrite", trace_id=trace_id) as s:
        retrieval_query = rewrite_query(user_input, history or [], trace_id=trace_id)
        s.payload = {
            "original": user_input,
            "rewritten": retrieval_query,
            "rewrite_applied": retrieval_query != user_input,
        }

    # 2. Retrieve — use the rewritten query for embedding + similarity search.
    with Span("docs_qa_retrieve", trace_id=trace_id) as s:
        chunks = search_docs(retrieval_query, top_k=top_k)
        if not chunks:
            s.payload = {"top_k": top_k, "retrieved": 0}
            return "I don't have any documents to answer from. Try ingesting a file first."
        valid_ids = {c["id"] for c in chunks}
        s.payload = {
            "query": retrieval_query,
            "top_k": top_k,
            "retrieved_ids": list(valid_ids),
        }

    # 3. Prompt — system prompt (RAG rules + chunks) first, then conversation
    #    history, then the ORIGINAL user_input (not the rewritten query).
    prompt = _build_prompt(user_input, chunks)
    messages: list[dict] = [{"role": "system", "content": prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_input})

    # 4a. Streaming path — returns a generator immediately; generation and
    #     guardrail happen lazily as the caller iterates.
    if stream:
        logger.debug("answer_from_docs: returning streaming generator for query=%r.", user_input)
        return _stream_rag_tokens(messages, valid_ids, user_input, trace_id=trace_id)

    # 4b. Non-streaming path (default) — wait for full response, then guardrail.
    with Span("docs_qa_generate", trace_id=trace_id) as s:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
        )
        log_token_usage(response, "docs_qa_generate", trace_id=trace_id)
        raw_answer = response.choices[0].message.content or ""
        answer = _strip_invalid_citations(raw_answer, valid_ids)
        s.payload = {
            "query": user_input,
            "raw_answer_length": len(raw_answer),
            "citations_removed": raw_answer != answer,
            "streamed": False,
        }

    log_event("trace_end", {"intent": "DOCS_QA", "streamed": False}, trace_id=trace_id)
    return answer


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is RAG?"
    print(f"Question: {query}\n")
    result = answer_from_docs(query)
    print("Answer:\n")
    print(result)
