# How We Learn in This Project

> Read this first if you are an AI agent, a collaborator, or a future version of me picking up this project. It explains the learning philosophy so every session stays consistent.

---

## What This Project Is For

The primary goal is **not** to ship a product. The primary goal is to **learn AI engineering** — how LLMs work, how to compose them into agents, how production systems are built, and how to think about quality, cost, and reliability.

The code is the medium. The understanding is the goal.

---

## The One Rule That Governs Everything

**Never make a change without explaining it.**

Specifically, for every code or config change, four things must be said:

| # | What to say | Example |
|---|---|---|
| 1 | **What** is changing | "Adding `stream_llm_answer` to `main.py`" |
| 2 | **Why** it was added | "So the CLI prints tokens as they arrive instead of waiting for the full response" |
| 3 | **Where** to look | "The generator is in `main.py`; the CLI dispatch is in `run.py`, the `isinstance(result, str)` branch" |
| 4 | **How to learn** from it | "This is the Python generator pattern — `yield` pauses the function and gives the caller one value at a time" |

A change that "just works" but isn't explained is useless here.

---

## How to Teach Each AI Concept

When a change involves a core AI pattern, go one level deeper. For every concept used, cover:

### 1. Name and map it
Say the concept name and connect it to the code explicitly.

> "This is **RAG — retrieval-augmented generation**: we retrieve chunks first, then pass them to the LLM as context. The retrieval is in `document_store.py`, the generation is in `qa.py`."

### 2. When to use it
Give the problem signal — the moment when you'd reach for this pattern.

> "Use RAG when the answer must be grounded in a specific corpus (documents, policies, tickets) and you want to avoid the LLM making things up."

### 3. Where else it applies
Connect it to other places in the project and in the world outside it.

> "In this project: the same pattern is used for the memory pipeline. Outside: every enterprise document search, compliance assistant, and support bot uses RAG."

### 4. How to identify the need
Give the trigger — the symptom that tells you this is the right tool.

> "Signal: 'I need the model to answer from my documents, not from its training data.' → RAG."

You don't need all four for every single line. Focus on the **main AI constructs** in each change.

---

## The Step-by-Step Rhythm

Every topic follows the same rhythm:

```
1. Explain first  →  explain what, why, where, and the concept
2. Ask / confirm  →  give the learner a chance to ask questions before code appears
3. Implement      →  write the code with inline notes that reinforce the concept
4. Point to learn →  after the change, say exactly where to read the logic and what to look for
```

Do **not** drop code first and explain later. The explanation is what makes the code stick.

---

## What Not to Do

- **Do not batch many changes with one summary.** Each logical change gets its own explanation.
- **Do not assume "it working" is enough.** If it works but isn't understood, it failed.
- **Do not use jargon without defining it.** "Chunking", "embedding", "guardrail" — define each the first time.
- **Do not skip the "where else" step.** Seeing one concept reused in multiple places is what builds intuition.
- **Do not over-explain trivial things.** A one-liner import doesn't need a paragraph. Focus depth on the AI concepts.

---

## The AI Concept Reference

A quick map of the concepts this project covers and when to use each one.

| Concept | Use when... | Problem signal |
|---|---|---|
| **Embeddings + semantic search** | You need "find things similar to this" | "I need to search by meaning, not keywords" |
| **RAG** | Answer must be grounded in a corpus | "The model must not invent — it must cite sources" |
| **Intent classification** | Same input can mean different actions | "I need to route the user to different pipelines" |
| **Tools / function calling** | Model needs deterministic or live data | "I need exact computation or a real API call" |
| **Semantic memory** | Long-term, queryable user context | "I need the agent to remember facts across sessions" |
| **Conversation history** | Multi-turn coherence | "The model needs to know what was said before" |
| **Token budget / truncation** | Context window management | "I'm sending too much history — it's slow and expensive" |
| **Query rewriting** | Follow-up questions break retrieval | "Pronouns like 'it' and 'that' ruin my search results" |
| **Streaming** | User shouldn't stare at a blank screen | "The response takes 3 seconds — show progress" |
| **Structured output** | LLM must return valid, parseable data | "My `json.loads` keeps failing on the LLM response" |
| **Guardrails** | Output must satisfy hard constraints | "The model is citing sources it didn't retrieve" |
| **Evaluation (Recall@K, faithfulness)** | You need to measure quality | "I changed the prompt — did it get better or worse?" |
| **LLM-as-judge** | Automated quality scoring | "I need to check 100 answers for hallucination" |
| **Tracing (P50/P95)** | Debugging and latency optimization | "Something is slow — I don't know which step" |
| **Cost tracking** | Operational awareness | "I don't know how much each request costs" |

---

## The Roadmap

The learning is structured as a breadth-first tour across all layers of AI engineering. The tracker lives in `docs/ROADMAP_BREADTH_TOUR.md`. Always check that file to see where we are and what is next.

The phases, in order:

| Phase | Topic |
|---|---|
| 1 | Multi-turn conversation (history, token budget, query rewriting) |
| 2 | Streaming responses |
| 3 | Structured outputs |
| 4 | Evaluation (retrieval quality, answer faithfulness) |
| 5 | Web UI (Streamlit, streaming UI) |
| 6 | Observability (tracing, cost tracking) |
| 7 | Portfolio polish (CI/CD, README) |

---

## Code Layout

```
agentforge/
  main.py              ← intent routing, run_agent, CLI entry point
  config.py            ← model, URL, token budget settings
  conversation.py      ← token counting, history trimming, query rewriting
  prompts.py           ← all prompt strings (versioned like code)
  logger.py            ← structured event logging
  tools.py             ← calculator and other deterministic tools
  memory/
    semantic.py        ← store and retrieve facts by embedding similarity
    response.py        ← answer_with_memory — LLM call with memory context
  rag/
    document_store.py  ← ingest, chunk, embed, and search documents
    qa.py              ← answer_from_docs — RAG pipeline with citation guardrail
  reasoning/
    react_engine.py    ← ReAct loop — multi-step reasoning with tools
docs/
  ROADMAP_BREADTH_TOUR.md  ← step-by-step learning roadmap with status tracker
  HOW_WE_LEARN.md          ← this file
tests/
  test_*.py            ← unit tests for each module
run.py                 ← thin CLI entry point (python run.py to start)
```

---

## For an AI Agent Picking This Up

If you are an AI agent starting a new session on this project:

1. **Read `docs/ROADMAP_BREADTH_TOUR.md`** — find the first step marked `⬜ Pending`. That is where we are.
2. **Read `docs/HOW_WE_LEARN.md`** (this file) — understand the teaching rules before making any change.
3. **Explain before you implement** — always cover what, why, where, and how to learn before writing code.
4. **One step at a time** — do not jump ahead or combine steps unless explicitly asked.
5. **Mark the step done** in the roadmap tracker after implementing it.

The learner's understanding comes first. Working code is a by-product of that, not the goal.
