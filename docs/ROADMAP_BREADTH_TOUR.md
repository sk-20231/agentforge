# Roadmap: Breadth-First AI Engineering Tour (Small Steps)

After completing the Document Q&A roadmap (RAG, chunking, embeddings, guardrails), this roadmap fills in the remaining layers of AI engineering: conversation, streaming, structured outputs, evaluation, UI, observability, and portfolio polish.

Same rules as before: pick a step, study it, ask questions, learn, then move to the next.

**Goal:** By the end of this roadmap the project is portfolio-ready on GitHub — clean code, real evaluation metrics, CI/CD gating, and a README that communicates production-level thinking to a hiring manager in 60 seconds.

---

## Progress Tracker

| Step | Topic | Status |
|------|-------|--------|
| 1 | Conversation history | ✅ Done |
| 2 | Token budget and truncation | ✅ Done |
| 3 | Query rewriting (conversation-aware RAG) | ✅ Done |
| 4 | Stream basic LLM response | ✅ Done |
| 5 | Stream the RAG answer | ✅ Done |
| 6 | Structured output for intent classification | ✅ Done |
| 7 | Structured output for ReAct engine | ⬜ Pending |
| 8 | Create evaluation dataset | ⬜ Pending |
| 9 | Measure retrieval quality (Recall@K) | ⬜ Pending |
| 10 | Measure answer faithfulness | ⬜ Pending |
| 11 | Streamlit chat interface | ⬜ Pending |
| 12 | Add streaming to the UI | ⬜ Pending |
| 13 | Structured trace logging | ⬜ Pending |
| 14 | Cost tracking | ⬜ Pending |
| 15 | CI/CD evaluation gating (GitHub Actions) | ⬜ Pending |
| 16 | Portfolio README and architecture docs | ⬜ Pending |

---

## Phase 1: Multi-Turn Conversation

Right now every message is independent — the agent has no idea what you said 10 seconds ago. This phase fixes that.

---

### Step 1: Add conversation history to run_agent

**What:** Add a `conversation_history` list (of `{"role": ..., "content": ...}` dicts) that persists across turns within a session. Each time the user sends a message, append it; each time the agent replies, append the reply. Pass the history to the LLM calls that need it (e.g. `answer_with_memory`, `answer_from_docs`).

**Why:** Without history, follow-up questions fail ("What is RAG?" → "How does it handle citations?" — the agent doesn't know what "it" refers to). Every real chatbot maintains a conversation buffer.

**Concepts:** Conversation context, message roles (system/user/assistant), context window.

**Where you'll look:** `agentforge/main.py` — `run_agent` and the CLI loop. Pass history into the functions that call the LLM.

**Size:** A list, a few appends, and passing it through. Small change, big UX improvement.

---

### Step 2: Token budget and truncation

**What:** Add a helper that counts (or estimates) tokens in the conversation history and truncates older messages when the total exceeds a budget (e.g. 3000 tokens). Keep the system prompt and the most recent N messages.

**Why:** LLMs have a fixed context window (e.g. 128k for GPT-4o-mini, but you pay per token). If you send the entire history of a 200-turn conversation, it's slow and expensive. A token budget keeps things practical.

**Concepts:** Token counting (tiktoken or estimate), sliding window, context window management. Trade-off: more history = more context but higher cost/latency.

**Where you'll look:** New helper in `agentforge/main.py` or a small `agentforge/conversation.py`. Called before every LLM call to trim the history.

**Size:** One function (~20 lines). Optional: install `tiktoken` for accurate counting, or estimate at ~4 chars per token.

---

### Step 3: Conversation-aware RAG (query rewriting)

**What:** Before calling `search_docs`, check if the user's message is a follow-up (e.g. "How does it handle that?"). If conversation history exists, ask the LLM to rewrite the query into a standalone question using the history (e.g. "How does RAG handle citations?"). Then search with the rewritten query.

**Why:** Embedding a follow-up question like "How does it work?" gives bad retrieval because "it" has no meaning without context. Rewriting fixes this — it's one of the highest-impact RAG improvements.

**Concepts:** Query rewriting / contextualization, when to transform input before retrieval. This is used in almost every production RAG system.

**Where you'll look:** `agentforge/rag/qa.py` — before the `search_docs` call in `answer_from_docs`. A small LLM call that takes history + current query → standalone query.

**Size:** One small function + one LLM call. The prompt is simple: "Given this conversation, rewrite the user's latest question as a standalone question."

---

## Phase 2: Streaming Responses

The agent currently waits for the full LLM response before printing anything. Real apps stream token by token.

---

### Step 4: Stream a basic LLM response

**What:** Modify `simple_llm_answer` (or create a new helper) to use `stream=True` in the OpenAI API call, then yield/print tokens as they arrive instead of waiting for the complete response.

**Why:** Streaming makes the agent feel responsive. Users see the first word in ~200ms instead of waiting 2–5 seconds for the full response. Every production LLM app does this.

**Concepts:** Streaming API (`stream=True`, iterating over chunks), SSE (Server-Sent Events) pattern, generator functions in Python.

**Where you'll look:** `agentforge/main.py` — the simplest LLM call (`simple_llm_answer`). Then optionally extend to `answer_with_memory`.

**Size:** A few lines changed. The API call adds `stream=True`, then you iterate over `response` and print each `chunk.choices[0].delta.content`.

---

### Step 5: Stream the RAG answer

**What:** Extend `answer_from_docs` to support streaming: retrieve chunks (non-streaming), build the prompt, then stream the LLM generation. The citation guardrail runs after the full response is collected.

**Why:** RAG answers are often the longest responses. Streaming them is high-value UX. The interesting design challenge: the guardrail needs the full text, so you stream to the user AND collect the full text for post-processing.

**Concepts:** Streaming + post-processing (collect while streaming), buffering, when guardrails conflict with streaming.

**Where you'll look:** `agentforge/rag/qa.py` — `answer_from_docs`. The retrieve + prompt steps stay synchronous; only the generate step streams.

**Size:** Medium. The streaming part is straightforward; the guardrail integration requires a small buffer pattern.

---

## Phase 3: Structured Outputs

Your intent classifier currently parses free-text JSON and hopes it's valid. The OpenAI API can guarantee structure.

---

### Step 6: Use response_format for intent classification

**What:** Replace the current "return JSON" prompt hack in `classify_intent` with OpenAI's native `response_format={"type": "json_object"}` (or `response_format` with a JSON schema if using a newer model). The LLM is guaranteed to return valid JSON.

**Why:** Right now, if the LLM returns `Sure! Here's the JSON: {"intent": ...}`, your `json.loads` fails and you fall back to ANSWER. With `response_format`, the API itself enforces valid JSON — no parsing failures.

**Concepts:** Constrained decoding (the model is forced to only generate tokens that produce valid JSON), structured outputs, why this is more reliable than prompt-based JSON.

**Where you'll look:** `agentforge/main.py` — `classify_intent`, the `client.chat.completions.create` call. Add `response_format={"type": "json_object"}`.

**Size:** One parameter added to one API call. Optionally define a Pydantic model for the schema if using `response_format` with a schema.

**Prompt versioning discipline (do this now):** After Step 6, your prompts are stable enough to treat like code. Move all prompt strings into `agentforge/prompts.py` (already exists), give each a name and a version comment, and track changes in git. A prompt change can break your eval metrics just as badly as a code change — version-controlling them is what separates engineering from tinkering. Production teams do this from day one.

---

### Step 7: Structured output for the ReAct engine

**What:** Apply the same `response_format` approach to the ReAct loop, which also expects JSON (thought, action, reply). This eliminates the `json.JSONDecodeError` fallback.

**Why:** The ReAct loop is the most fragile JSON consumer — it expects a complex schema with nested fields. Structured output makes it robust.

**Concepts:** Same as Step 6, applied to a more complex schema. Shows how constrained decoding scales to nested objects.

**Where you'll look:** `agentforge/reasoning/react_engine.py` — the `client.chat.completions.create` call inside the loop.

**Size:** Small — one parameter change, but you may also want to define the schema explicitly.

---

## Phase 4: Evaluation

You can't improve what you can't measure. This phase adds simple evaluation so you can answer "is my RAG actually good?"

---

### Step 8: Create an evaluation dataset

**What:** Create a JSON file (`tests/eval_dataset.json`) with question/answer/source triples. Each entry has: a question, the expected answer (or key facts), and which chunk ids should be retrieved. These are your "ground truth."

**Why:** Without ground truth, you can't measure retrieval quality or answer faithfulness. Even 5 examples is enough to catch regressions while learning. For a portfolio, aim for 50–200 verified examples — that's the bar production teams and hiring managers recognise as credible. The video by Ashwarashan (ex-Microsoft/Google/IBM) explicitly called this out: a larger, curated golden dataset with CI/CD gating is what separates a demo from a production-grade system.

**Concepts:** Evaluation datasets, ground truth, why you need test cases that are separate from your code. Same idea as unit tests, but for AI output quality.

**Where you'll look:** `tests/eval_dataset.json` (new file). Based on documents you've already ingested.

**Size:** Start with 5–10 to learn the pattern. Grow to 50–200 before publishing to GitHub. Each example takes ~5 minutes to write and verify manually — it's worth the time.

**Portfolio note:** Label each example with the source document and the exact chunk id(s) that should be retrieved. This makes Recall@K (Step 9) directly measurable and shows rigour to anyone reading your repo.

---

### Step 9: Measure retrieval quality (Recall@K)

**What:** Write a script (or test) that runs each eval question through `search_docs`, checks whether the expected chunk ids appear in the top-K results, and computes Recall@K (fraction of expected chunks that were retrieved).

**Why:** If retrieval is bad, the LLM can't give good answers no matter how good the prompt is. Retrieval quality is the #1 bottleneck in RAG. Measuring it tells you where to focus.

**Concepts:** Recall@K, precision@K, MRR (Mean Reciprocal Rank). These are standard information retrieval metrics used everywhere from Google Search to RAG systems.

**Where you'll look:** New file `agentforge/evaluation.py` or a test in `tests/`. Loops over eval dataset, calls `search_docs`, compares to expected ids.

**Size:** One function, ~30 lines. Prints a summary like "Recall@5: 80% (4/5 questions had the right chunk in top 5)."

**RAGAS connection:** [RAGAS](https://docs.ragas.io) is the industry framework that productionises exactly this metric. Building it by hand first means you'll understand precisely what RAGAS is computing when you use it later. After Step 9 works, look at RAGAS's `context_recall` metric — it's the same idea with more statistical rigour.

---

### Step 10: Measure answer faithfulness

**What:** For each eval question, run `answer_from_docs`, then use an LLM call to judge: "Does this answer only contain information from the provided chunks, or did it add/invent information?" Score as faithful or not.

**Why:** Even with good retrieval, the LLM might hallucinate facts not in the chunks. Faithfulness evaluation catches this. This is the core idea behind RAGAS and similar frameworks.

**Concepts:** LLM-as-judge (using one LLM to evaluate another's output), faithfulness scoring, automated evaluation. This pattern is used in RAGAS, DeepEval, and most AI evaluation frameworks.

**Where you'll look:** Same `agentforge/evaluation.py`. A function that takes (question, answer, chunks) and asks an LLM "is the answer faithful to the chunks?"

**Size:** One function + one LLM call per question. Prints "Faithfulness: 90% (9/10 answers were faithful)."

**RAGAS connection:** RAGAS's `faithfulness` metric is the productionised version of exactly this — it uses an LLM-as-judge with a more refined prompt and aggregates scores across a dataset. After Step 10 works manually, look at how RAGAS computes it. You'll find the logic almost identical to what you built.

---

## Phase 5: Simple Web UI

Make the agent usable by someone who doesn't have a terminal open.

---

### Step 11: Streamlit chat interface

**What:** Create a simple Streamlit app (`app.py`) with a chat interface. The user types a message, the agent responds, conversation history is displayed. Add a sidebar for user ID and a button to ingest a file.

**Why:** A UI makes the agent tangible — you can show it to someone, get feedback, and see how all the pieces come together. Streamlit is the fastest way to build an AI chat UI in Python.

**Concepts:** Streamlit session state, `st.chat_message`, `st.chat_input`, file upload. How to connect a backend (your agent) to a frontend.

**Where you'll look:** New file `app.py` at the project root. Imports `run_agent` from `agentforge.main` and `ingest_file` from `agentforge.rag.document_store`.

**Size:** ~50–80 lines. Streamlit handles most of the UI; you wire it to your existing functions.

---

### Step 12: Add streaming to the UI

**What:** Instead of `st.write(result)` after the full response, stream tokens into the chat bubble as they arrive (using `st.write_stream` or a placeholder that updates).

**Why:** Combines Phase 2 (streaming) with Phase 5 (UI). The user sees the answer build up word by word, which is the standard UX for AI chat apps.

**Concepts:** Streaming + UI integration, Streamlit's `st.write_stream`, generator-to-UI pattern.

**Where you'll look:** `app.py` — replace the blocking `run_agent` call with a streaming variant for ANSWER and DOCS_QA intents.

**Size:** Small change in the UI code, reuses the streaming work from Phase 2.

---

## Phase 6: Observability

When things go wrong (and they will), you need to see what happened inside the agent's chain of calls.

---

### Step 13: Structured trace logging

**What:** Upgrade `log_event` to produce structured traces: each `run_agent` call gets a unique trace ID, and every sub-step (intent classification, retrieval, LLM call, tool call) is logged with that trace ID, timestamps, and duration. Track latency at P50 and P95 percentiles — not just averages.

**Why:** Right now your logs are flat events. With traces, you can see the full chain: "User asked X → classified as DOCS_QA (120ms) → retrieved 5 chunks (350ms) → generated answer (1200ms) → guardrail removed 1 citation." This is essential for debugging and optimization. **Averages hide worst-case performance** — P95 tells you what your slowest 5% of users experience, which is what gets escalated in production.

**Concepts:** Distributed tracing (simplified), trace IDs, spans, latency percentiles (P50/P95), latency budget decomposition. Same concepts used in LangSmith, Langfuse, Phoenix (Arize), and OpenTelemetry for LLMs.

**Where you'll look:** `agentforge/logger.py` — add trace ID generation and duration tracking. Update callers to pass trace context.

**Tools to know:** After building this manually, [Langfuse](https://langfuse.com) is the recommended next step — open-source, self-hostable, shows the exact messages array and token counts for every LLM call. Free to start and far easier to self-host than LangSmith.

**Size:** Medium. Upgrading the logger + updating the main call sites.

**Portfolio note:** Being able to say "our median response is 800ms and our P95 is 2.1 seconds, here's the per-span breakdown" in a portfolio README or interview is a strong signal of production-level thinking.

---

### Step 14: Cost tracking

**What:** After each LLM call, log the token usage (`response.usage.prompt_tokens`, `completion_tokens`) and estimate cost based on model pricing. Add a summary at the end of each session or available via the UI.

**Why:** LLM API calls cost money. Knowing which operations are expensive (e.g. RAG prompts with 5 long chunks vs. intent classification with a short prompt) helps you optimize. In production, cost tracking is mandatory.

**Concepts:** Token usage, API cost estimation, prompt efficiency. Understanding the relationship between prompt length, response length, and cost.

**Where you'll look:** Every place that calls `client.chat.completions.create` — extract `response.usage` and log it. Aggregate in the logger or a separate tracker.

**Size:** Small per call site, but touches many files. Could add a wrapper function to centralize.

---

## Phase 7: Portfolio Polish

The code works. Now make it presentable. A hiring manager will spend 60–90 seconds on your GitHub repo before deciding whether to read further. This phase is what makes those 90 seconds count.

---

### Step 15: CI/CD evaluation gating (GitHub Actions)

**What:** Add a GitHub Actions workflow (`.github/workflows/eval.yml`) that runs your evaluation suite (`agentforge/evaluation.py`) automatically on every push to main. If Recall@K drops below a threshold (e.g. 70%) or faithfulness drops below a threshold (e.g. 80%), the workflow fails and the commit is flagged.

**Why:** This is exactly how production AI teams operate. Every model update, prompt change, or retrieval tweak can silently degrade quality. Automated gating catches regressions before they ship. More importantly: **this is rare in portfolio projects** — most candidates only show working demos, not quality discipline. A failing CI badge that you then fix tells a better story than one that never ran.

**Concepts:** Continuous integration for AI systems, regression gating, quality thresholds. Same principle as unit test gates in software engineering, applied to AI output quality metrics.

**Where you'll look:** New file `.github/workflows/eval.yml`. Runs `python -m agentforge.evaluation` and exits non-zero if metrics are below threshold. Uses `OPENAI_API_KEY` as a GitHub Actions secret.

**Size:** ~30 lines of YAML + a small exit-code check in `evaluation.py`.

**Portfolio signal:** A green CI badge on your README ("Eval passing: Recall@5=85%, Faithfulness=90%") tells a hiring manager immediately that you think about AI quality like an engineer, not like a hobbyist.

---

### Step 16: Portfolio README and architecture documentation

**What:** Write a `README.md` at the project root that tells the story of what you built, why each decision was made, and what someone can learn from the code. Include an architecture diagram, a concepts map (which AI concept maps to which file), how to run it locally, and the eval metrics from Step 15.

**Why:** Your code might be excellent but if someone can't understand it in 90 seconds, they move on. A README is your pitch to a hiring manager, a collaborator, or your future self. It's also where you demonstrate the meta-skill of communicating technical decisions clearly — which is what senior engineers and tech leads do constantly.

**What the README should cover:**
1. **What this project is** — one paragraph, plain English
2. **Architecture diagram** — the layered diagram from this roadmap, updated with all phases
3. **AI concepts map** — table linking each concept (RAG, embeddings, query rewriting, LLM-as-judge, streaming, tracing) to the file where it lives
4. **How to run** — `pip install -r requirements.txt`, set `OPENAI_API_KEY`, run CLI or Streamlit
5. **Eval results** — your actual Recall@K and faithfulness numbers from Phase 4
6. **What I learned** — honest 3–5 bullet points about the hardest problems you solved and why they mattered
7. **What's next** — Advanced RAG, fine-tuning, local models — shows you know the landscape beyond what you built

**Where you'll look:** New `README.md` at the project root. Reference `docs/ROADMAP_BREADTH_TOUR.md` for the architecture and concept explanations you've already written.

**Size:** 200–400 words of prose + the architecture diagram + the concepts table. Takes 2–3 hours to write well. Worth every minute.

**Portfolio signal:** The video's key quote: *"The kind of work that makes someone look at your GitHub and say, 'Okay, this person actually understands how production AI systems work.'"* The README is where you say that — not just show it.

---

## How to use this roadmap

1. Read the phases and steps in order.
2. Pick **one step** (e.g. "let's do Step 1").
3. We implement only that step; I explain what, why, where to look, and the AI concepts.
4. You study the code, run it, and ask questions until it's clear.
5. When ready, pick the next step and repeat.

You don't need to do every step in every phase. If a phase doesn't interest you, skip to the next one. The steps within each phase do build on each other, though.

---

## What's NOT in this roadmap (for later)

These are important but better tackled after the breadth tour:

- **Advanced RAG** (hybrid search BM25 + vector, cross-encoder re-ranking, HyDE, chunk overlap) — optimize retrieval after you can *measure* it (Phase 4). Chunk overlap (100-token overlap between chunks) prevents important sentences being split at boundaries and is one of the easiest quality wins.
- **Multi-agent orchestration** — after you've built several single-agent capabilities
- **Fine-tuning** (LoRA, QLoRA, DPO preference tuning) — after you understand when RAG + prompting isn't enough and have eval metrics to prove the gap
- **Vector database** (Chroma, FAISS, pgvector) — after your JSON corpus becomes a bottleneck
- **Deployment** (Docker, cloud hosting) — after you have the UI and want to share it
- **Multi-modal** (images, audio, voice) — after you're comfortable with text-based AI
- **Local model inference** (Ollama, Llama 3, quantization GGUF Q4/Q5) — privacy-first and edge deployments; a strong separate portfolio project once the core agent is solid

---

## The big picture

After this roadmap you'll have touched every layer:

```
┌──────────────────────────────────────────────────────┐
│  Portfolio / Career (Phase 7)                        │
│    CI/CD eval gating, README, architecture docs      │
├──────────────────────────────────────────────────────┤
│  UX / Interface (Phase 5)                            │
│    Streamlit chat, streaming UI                      │
├──────────────────────────────────────────────────────┤
│  Agent / Orchestration                               │
│    Intent routing, ReAct (already done)              │
├──────────────────────────────────────────────────────┤
│  Capabilities / Pipelines                            │
│    RAG, memory, tools (already done)                 │
│    + conversation + query rewriting (Phase 1)        │
│    + structured outputs (Phase 3)                    │
├──────────────────────────────────────────────────────┤
│  Foundation / Primitives                             │
│    Embeddings, LLM calls (already done)              │
│    + streaming (Phase 2)                             │
├──────────────────────────────────────────────────────┤
│  Ops / Quality (Phases 4 + 6)                        │
│    Evaluation (Recall@K, faithfulness, RAGAS)        │
│    Tracing (P50/P95 latency, trace IDs, Langfuse)    │
│    Cost tracking                                     │
└──────────────────────────────────────────────────────┘
```

You'll be able to build, evaluate, debug, and present any LLM-based application — and show it on GitHub in a way that communicates production-level thinking to a hiring manager.
