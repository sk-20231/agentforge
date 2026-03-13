# Revision Notes — Phase 1 & 2 Changes

A compact reference for every major change made so far. For each step: what changed, which file and function, why, and the one concept to remember.

---

## Step 1 — Conversation History

### The problem
Every message was independent. The agent had no memory of what was said earlier in the same session. A follow-up like "How does it work?" failed because "it" had no context.

### What changed

**`run.py` — the CLI loop**
- Added `history: list[dict] = []` before the loop
- After every turn, two lines appended to it:
  ```python
  history.append({"role": "user", "content": user_input})
  history.append({"role": "assistant", "content": result})
  ```
- `history` is passed into `run_agent(... history=history)`

**`agentforge/main.py` — `run_agent`**
- Added `history: list[dict] = None` parameter
- Passes `history` down to `answer_with_memory` and `answer_from_docs`

**`agentforge/memory/response.py` — `answer_with_memory`**
- Added `history: list[dict] = None` parameter
- Inserts history into the messages array between the system prompt and the current user message:
  ```python
  if history:
      messages.extend(history)
  messages.append({"role": "user", "content": user_input})
  ```

### Why
The LLM sees prior turns as part of the prompt. It can now refer back to earlier parts of the conversation.

### Concept
**Conversation context / message roles** — The OpenAI API takes a `messages` array of `{"role": ..., "content": ...}` dicts. Role is `system`, `user`, or `assistant`. History is just previous user+assistant pairs inserted into this array. The LLM reads the whole array top-to-bottom before generating a reply.

---

## Step 2 — Token Budget and Truncation

### The problem
A long conversation would keep growing. Eventually the history would be too large and expensive to send every turn. At the extreme it would hit the model's context window limit and error.

### What changed

**`agentforge/conversation.py` — new file with two functions**

`count_tokens(messages)`:
- Estimates token count using the 4 chars ≈ 1 token heuristic
- Adds 4 tokens overhead per message (for role label + API formatting)
- Deliberately over-estimates (safer to trim slightly more)

`trim_history(history, budget)`:
- Works on a **copy** of the list — never mutates the caller's original
- Drops the **oldest user+assistant pair** (2 messages at a time) in a loop until the estimate fits within `budget`
- Always keeps at least the last 2 messages (most recent turn) no matter what

**`agentforge/config.py`**
- Added `HISTORY_TOKEN_BUDGET = 2000` (overridable via env var)
- Conservative default — leaves room for system prompts and RAG chunks

**`agentforge/main.py` — `run_agent`**
- Added one call at the very top of every turn:
  ```python
  safe_history = trim_history(history or [], HISTORY_TOKEN_BUDGET)
  ```
- All pipelines receive `safe_history` instead of raw `history`

### Why
Cost control and reliability. At 2000 tokens for history + ~1000 for RAG chunks + system prompt, total prompt stays well within limits. Trimming oldest turns first keeps the most recent and relevant context.

### Concept
**Sliding window / context window management** — LLMs have a fixed context window. You pay per token and hit hard limits if you exceed it. A sliding window that drops the oldest turns first is the standard solution. `tiktoken` is the precise alternative to the 4-char heuristic if accuracy matters.

---

## Step 3 — Query Rewriting (Conversation-Aware RAG)

### The problem
When a user asks a follow-up like "How does it handle citations?", the word "it" has no meaning to the embedding model. Embedding that sentence gives a poor vector → wrong chunks retrieved → bad RAG answer. Retrieval runs before the LLM sees history, so it can't resolve pronouns on its own.

### What changed

**`agentforge/conversation.py` — new function `rewrite_query`**
- Takes `user_input` and `history`
- If history is empty: returns `user_input` unchanged (first turn is always self-contained)
- If history exists: formats history as readable text and sends a small LLM call with this prompt:
  > "Rewrite the user's latest message as a complete standalone question using the conversation history. Return only the rewritten question."
- Falls back to original `user_input` on any failure — retrieval always runs

**`agentforge/rag/qa.py` — `answer_from_docs`**
- Added one call before `search_docs`:
  ```python
  retrieval_query = rewrite_query(user_input, history or [])
  ```
- `retrieval_query` is used for embedding + search
- The original `user_input` is still used in the generation prompt — the rewrite is invisible to the user

### Why
The rewrite resolves pronouns and references so the search query is self-contained. "How does it handle citations?" → "How does RAG handle citations?" — the second embeds and retrieves correctly.

### Concept
**Query rewriting / contextualization** — one of the highest-impact RAG improvements. Used in virtually every production RAG system. The key insight: retrieval and generation are separate steps, and retrieval needs a self-contained query. Use it when follow-up questions break your search results.

---

## Step 4 — Stream Basic LLM Response

### The problem
The agent waited for the complete LLM response before printing anything. Users stared at a blank screen for 2–5 seconds. Every production chat UI streams token by token.

### What changed

**`agentforge/main.py` — new function `stream_llm_answer`**
- A Python **generator** that uses `stream=True` in the OpenAI call
- Iterates over response chunks; yields each `chunk.choices[0].delta.content` token
- The `if token:` guard skips the final `None` sentinel chunk OpenAI sends

**`agentforge/main.py` — `run_agent`**
- Added `stream: bool = False` parameter
- For ANSWER / RESPOND_WITH_MEMORY: passes `stream` into `answer_with_memory`
- For DOCS_QA: passes `stream` into `answer_from_docs`

**`agentforge/memory/response.py` — `answer_with_memory`**
- Added `stream: bool = False` parameter
- When `stream=True`: returns `_stream_tokens(messages)` — a generator
- When `stream=False`: existing behaviour unchanged (returns full `str`)

**`agentforge/memory/response.py` — new function `_stream_tokens`**
- Internal generator separated from `answer_with_memory` for clean error handling
- `stream=True` on the OpenAI call; yields each token as it arrives

**`run.py` — CLI loop**
- Passes `stream=True` to `run_agent`
- Detects whether result is a string or a generator:
  ```python
  if isinstance(result, str):
      print("Agent:", result)          # REMEMBER, ACT, REACT, IGNORE
  else:
      print("Agent: ", end="", flush=True)
      for token in result:             # ANSWER, DOCS_QA
          print(token, end="", flush=True)
          full_text += token
      print()
  ```
- `flush=True` is critical — without it Python buffers stdout and the streaming effect is lost

### Why
Streaming makes the agent feel responsive. First token appears in ~200ms instead of waiting for the full response. Backward compatible — `stream=False` is the default so all existing callers (tests etc.) are unaffected.

### Concept
**Streaming API + Python generators** — `stream=True` switches the OpenAI response from one object to an iterator of `ChatCompletionChunk` objects, one per token. A generator (`yield`) lets the caller consume one token at a time. Use streaming whenever the user should see partial output before the model finishes.

---

## Step 5 — Stream the RAG Answer

### The problem
RAG answers are the longest responses — often 3–5 seconds of generation. The streaming UX improvement is highest here. But RAG has a complication: the citation guardrail needs the *complete* response to check which `[doc-X]` citations are valid.

### What changed

**`agentforge/rag/qa.py` — `answer_from_docs`**
- Added `stream: bool = False` parameter
- When `stream=True`: returns `_stream_rag_tokens(messages, valid_ids, user_input)` immediately
- When `stream=False`: existing non-streaming path unchanged

**`agentforge/rag/qa.py` — new function `_stream_rag_tokens`**
- Implements the **stream-and-collect** pattern:
  ```python
  full_text = ""
  for chunk in response:
      token = chunk.choices[0].delta.content
      if token:
          full_text += token   # collect for guardrail
          yield token          # give to caller immediately

  # After loop — full response is complete
  clean_answer = _strip_invalid_citations(full_text, valid_ids)
  log_event("docs_qa_answer", {"citations_removed": full_text != clean_answer, "streamed": True})
  ```
- The guardrail runs after the last token — logs whether citations were stripped
- Trade-off: the user has already seen the raw tokens; the guardrail here is **observability**, not prevention

**`agentforge/main.py` — `run_agent` DOCS_QA branch**
- One line updated:
  ```python
  return answer_from_docs(user_input, history=safe_history, stream=stream)
  ```

### Why
RAG generation is the slowest step. Steps 1–3 (rewrite → retrieve → prompt-build) are always synchronous because generation can't start until chunks are retrieved. Only the generation step streams. The user sees a brief pause (retrieval), then tokens flow.

### Concept
**Stream + collect + post-process** — when you need live UX *and* a whole-response operation (guardrail, eval, logging). You can't run the guardrail on a half-finished sentence, so you buffer every token as you yield it, then process the buffer after the loop. Used in LangChain callbacks, LangSmith tracing, and any streaming pipeline with post-generation validation.

### The honest trade-off
With `stream=True`, a hallucinated citation like `[doc_chunk_99]` is visible to the user before the guardrail catches it. With `stream=False`, the guardrail runs first and the user only sees the clean answer. For citation integrity, non-streaming is safer. For UX, streaming wins. Production systems handle this with better prompting (reduce hallucinated citations at the source) rather than buffering.

---

## Quick Reference — What Lives Where

| Concept | File | Function |
|---|---|---|
| History accumulation | `run.py` | CLI loop |
| Token counting | `agentforge/conversation.py` | `count_tokens` |
| History trimming | `agentforge/conversation.py` | `trim_history` |
| Query rewriting | `agentforge/conversation.py` | `rewrite_query` |
| Intent routing | `agentforge/main.py` | `run_agent` |
| Memory-aware answer (non-stream) | `agentforge/memory/response.py` | `answer_with_memory` |
| Memory-aware answer (stream) | `agentforge/memory/response.py` | `_stream_tokens` |
| RAG pipeline | `agentforge/rag/qa.py` | `answer_from_docs` |
| RAG streaming + guardrail | `agentforge/rag/qa.py` | `_stream_rag_tokens` |
| Citation guardrail | `agentforge/rag/qa.py` | `_strip_invalid_citations` |
| Chunk retrieval (top-k) | `agentforge/rag/document_store.py` | `search_docs` |
| CLI stream dispatch | `run.py` | `isinstance(result, str)` block |

---

## The One-Line Summary of Each Step

| Step | One line |
|---|---|
| 1 — History | Append user+assistant turns to a list; pass it into every LLM call |
| 2 — Token budget | Estimate token count; drop oldest pairs until it fits within 2000 tokens |
| 3 — Query rewriting | Ask the LLM to turn follow-up questions into standalone queries before searching |
| 4 — Stream basic | Add `stream=True`; yield tokens one by one with a generator; detect in CLI with `isinstance` |
| 5 — Stream RAG | Same as Step 4 but also buffer every token so the citation guardrail can run after |
