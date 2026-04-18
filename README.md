# AgentForge

> A production-style AI agent built from primitives — no LangChain, no LlamaIndex, no framework magic. Intent routing, RAG with citation guardrails, ReAct reasoning, evaluation gating in CI, cost tracking, and trace observability — all implemented directly against the OpenAI API.

![CI](https://github.com/sk-20231/agentforge/actions/workflows/ci.yml/badge.svg)
![Eval Gate](https://github.com/sk-20231/agentforge/actions/workflows/eval.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![Tests](https://img.shields.io/badge/tests-139_passing-blue)

---

## Why this exists

High-level frameworks (LangChain, LlamaIndex) get you running fast but hide the parts that matter when you need to **debug, evaluate, or scale** an AI system:

- What does retrieval actually return?
- Why did the model hallucinate that citation?
- Which prompt burned the token budget?
- How do you *know* the new embedding model didn't regress quality?

AgentForge is the same capabilities, built from the ground up — so every decision (chunk size, similarity metric, routing logic, citation format, eval thresholds) is visible and tunable. It's what I'd reach for if I needed to understand *what's really happening* behind an agent.

---

## What it does

An interactive agent (CLI or Streamlit UI) that can:

- **Classify intent** and route each message to the right pipeline (remember, act, reason, search docs, etc.)
- **Remember things** using semantic memory (embeddings + cosine similarity)
- **Use tools** — Wikipedia lookup (more tools coming: weather, news, markets)
- **Multi-step reasoning** — ReAct (Reason + Act) loop for complex tasks
- **Answer from documents (RAG)** — chunk, embed, retrieve, generate with a citation guardrail that strips hallucinated references
- **Personalize** answers with stored user facts
- **Rewrite follow-ups** into standalone queries so "How does it work?" retrieves correctly in a 10-turn conversation
- **Stream** tokens to the CLI and Streamlit UI as they arrive
- **Self-evaluate** — automated Recall@K retrieval tests + LLM-as-judge faithfulness scoring, gated in CI
- **Observability** — every run has a trace ID, per-span latency, P50/P95 percentiles, token counts, and USD cost

---

## Architecture

```mermaid
flowchart TD
    A[User Input] --> B[run_agent]
    B --> C[trim_history<br/>token budget]
    C --> D[classify_intent<br/>LLM call - JSON mode]
    D --> E{Intent?}
    E -->|REMEMBER| F[store_memory<br/>embed + persist]
    E -->|ACT| G[run_llm_with_tools<br/>wikipedia, weather, news, market]
    E -->|REACT| H[react_loop<br/>think → act → observe]
    E -->|DOCS_QA| I[rewrite_query →<br/>search_docs →<br/>answer_from_docs]
    E -->|ANSWER / RESPOND_WITH_MEMORY| J[answer_with_memory<br/>inject facts]
    E -->|IGNORE| K[static reply]
    F --> L[Response + trace_id]
    G --> L
    H --> L
    I --> L
    J --> L
    K --> L
    L --> M[log event<br/>tokens + cost + latency]
```

Every LLM call goes through a `Span` context manager that records start/end timestamps, duration, and token usage. Traces are structured JSONL — grep-friendly, easy to aggregate.

---

## AI concepts covered

| Concept | Where in the code | Why it matters |
|---|---|---|
| **Intent classification** | [`main.py` — `classify_intent`](agentforge/main.py) | One input can mean many actions — specialize per intent |
| **Semantic memory** | [`memory/semantic.py`](agentforge/memory/semantic.py) | Long-term, queryable user context via embeddings |
| **Embeddings + cosine similarity** | [`memory/semantic.py`](agentforge/memory/semantic.py) | Core primitive for "find things similar to this" |
| **Tool calling** | [`tools.py`](agentforge/tools.py) | Deterministic operations + live external data |
| **ReAct reasoning** | [`reasoning/react_engine.py`](agentforge/reasoning/react_engine.py) | Multi-step tasks need think-act-observe loops |
| **RAG (retrieval-augmented generation)** | [`rag/`](agentforge/rag/) | Ground answers in a real corpus, not model memory |
| **Chunking with overlap** | [`rag/document_store.py — chunk_text`](agentforge/rag/document_store.py) | Retrieval quality depends on chunk boundaries |
| **Citation guardrails** | [`rag/qa.py — _strip_invalid_citations`](agentforge/rag/qa.py) | Deterministic safety layer on top of probabilistic output |
| **Conversation buffer + token budget** | [`conversation.py — trim_history`](agentforge/conversation.py) | Keep context cost predictable as history grows |
| **Query rewriting** | [`conversation.py — rewrite_query`](agentforge/conversation.py) | Follow-up pronouns break RAG; rewrite first |
| **Streaming** | multiple modules | Better UX — tokens arrive as they're generated |
| **Structured outputs** | [`prompts.py — OUTPUT_SCHEMA`](agentforge/prompts.py) | `response_format={"type": "json_object"}` for reliable parsing |
| **Evaluation — Recall@K** | [`evaluation.py`](agentforge/evaluation.py) | Measure retrieval quality objectively |
| **Evaluation — Faithfulness (LLM-as-judge)** | [`evaluation.py`](agentforge/evaluation.py) | Is the answer actually supported by the retrieved chunks? |
| **CI/CD eval gating** | [`.github/workflows/eval.yml`](.github/workflows/eval.yml) | Fail the build if Recall@K < 70% or faithfulness < 80% |
| **Trace logging** | [`logger.py — Span`](agentforge/logger.py) | Every run has a trace_id, per-span latency, JSONL events |
| **Cost tracking** | [`logger.py — log_token_usage`](agentforge/logger.py) | Per-operation USD cost, visible in the Streamlit sidebar |
| **Embedding model provenance** | [`rag/document_store.py`](agentforge/rag/document_store.py) | Corpus stores which model generated the embeddings — fail fast on mismatch |

---

## Technical decisions worth explaining

### 1. Intent-first routing, not one mega-prompt

A single prompt that does everything ("remember if it's a fact, calculate if it's math, search docs if it mentions files...") splits the model's attention across all responsibilities. Quality degrades on every task.

Instead, the first LLM call is a **cheap, fast classifier** with one job: return the intent as JSON. Then each intent has its own specialized pipeline with its own prompt, its own tools, and its own evaluation target.

**Trade-off:** adds one LLM call per turn. In practice, that call is tiny (a dozen output tokens) and the quality win on each pipeline more than pays for it. It also makes the system **debuggable per-intent** — you can measure and tune each pipeline independently.

See [`main.py — classify_intent`](agentforge/main.py) and [`prompts.py`](agentforge/prompts.py).

### 2. Embedding model guard — fail fast on model swap

Embeddings from different models live in **different vector spaces**. If you swap `text-embedding-3-small` (1536 dim) for `text-embedding-3-large` (3072 dim) and reuse your old corpus, cosine similarity returns nonsense. Retrieval quality silently collapses — your tests might still pass, your users might still get *an* answer, but the wrong chunks are being retrieved.

AgentForge's corpus persists the embedding model name alongside the chunks. On load, [`load_corpus()`](agentforge/rag/document_store.py) compares the stored model against the configured model. On mismatch, it raises `RuntimeError` with a recovery guide baked into the error message:

```
Corpus was built with text-embedding-3-small but config says text-embedding-3-large.
Re-ingest your documents:
  1. Delete corpus.json (or set AGENT_CORPUS_FILE to a new path)
  2. Run: python -m agentforge.rag.document_store path/to/your/docs
  Current documents in corpus: EmployeeHandbook.txt, RefundPolicy.txt
```

**Why this pattern:** silent regressions are the worst kind of bug. Fail loudly at the boundary — with enough info for the reader to recover without grepping the code.

### 3. Citation guardrails — verify, don't trust

RAG systems ask the LLM to cite its sources. The LLM **sometimes invents citations** — `[Chunk_47]` when your corpus only goes up to `Chunk_12`. Telling it not to in the prompt reduces the rate, but doesn't eliminate it.

AgentForge runs a **post-generation guardrail**: regex-extract every citation marker, intersect with the actual chunk IDs returned by retrieval, and strip any that don't match. The answer is rewritten with the invalid citations removed. See [`qa.py — _strip_invalid_citations`](agentforge/rag/qa.py).

**Why this pattern:** the generative layer is probabilistic — always. Build deterministic checks on top. Belt and suspenders. This is the same philosophy as SQL parameterization, CSRF tokens, or any other "never trust the upstream layer" security pattern — applied to LLM output.

---

## Project structure

```
agentforge/
├── agentforge/                   # Main package
│   ├── config.py                 # Environment-based configuration
│   ├── conversation.py           # Token budget trimming + query rewriting
│   ├── logger.py                 # Structured event logging (JSONL) + cost + spans
│   ├── main.py                   # Intent classification + agent orchestrator + CLI
│   ├── prompts.py                # System prompts and structured output schema
│   ├── tools/                    # Plugin-style tool package (one module per tool)
│   │   ├── __init__.py           # Registry + orchestration (execute_tool, run_llm_with_tools)
│   │   └── wikipedia.py          # Wikipedia lookup tool
│   ├── evaluation.py             # Recall@K + faithfulness scoring
│   ├── memory/                   # Semantic memory sub-package
│   │   ├── semantic.py           # Embeddings, similarity, memory store/retrieve
│   │   └── response.py           # Memory-aware answer generation
│   ├── rag/                      # RAG sub-package
│   │   ├── document_store.py     # Chunking, corpus load/save, search, ingestion
│   │   └── qa.py                 # RAG answer pipeline + citation guardrail
│   └── reasoning/
│       └── react_engine.py       # ReAct loop (think → act → observe → repeat)
├── tests/                        # 139 unit tests (mocked, no API calls)
│   └── fixtures/
│       └── test_corpus.json      # Pre-built corpus so CI evals run API-key-free
├── .github/workflows/            # CI/CD pipelines
│   ├── ci.yml                    # Unit tests on every push/PR (free, fast)
│   └── eval.yml                  # Recall@K + faithfulness gate on push to main
├── docs/                         # Roadmaps and design docs
├── run.py                        # CLI entry point
├── app.py                        # Streamlit UI entry point
├── requirements.txt              # Production dependencies
├── requirements-dev.txt          # Test dependencies
├── .env.example                  # Environment variable template
└── .gitignore
```

---

## Setup

```bash
git clone git@github.com:sk-20231/agentforge.git
cd agentforge
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
pip install -r requirements-dev.txt   # for tests

cp .env.example .env                  # then add OPENAI_API_KEY
```

## Usage

**Run the CLI agent:**
```bash
python run.py
```

**Run the Streamlit UI:**
```bash
streamlit run app.py
```

**Ingest a document for RAG:**
```bash
python -m agentforge.rag.document_store path/to/file.txt
```

**Run tests:**
```bash
pytest tests/ -v
```

**Run the evaluation suite:**
```bash
python -m agentforge.evaluation --eval --faithfulness
```

---

## Configuration

All settings read from environment variables (with defaults) in [`config.py`](agentforge/config.py):

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | (required) | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | Chat model |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `OPENAI_BASE_URL` | (OpenAI default) | Custom API base URL |
| `AGENT_MEMORY_DIR` | `memory` | Directory for user memory files |
| `AGENT_LOG_FILE` | `agent_logs.jsonl` | Event log file |
| `AGENT_CORPUS_FILE` | `corpus.json` | RAG document corpus file |
| `HISTORY_TOKEN_BUDGET` | `2000` | Max tokens kept in history before trimming |

---

## Intent routing

Every message is classified into one of these intents:

| Intent | What happens | Example |
|---|---|---|
| `REMEMBER` | Stores a personal fact in semantic memory | "I prefer dark roast coffee" |
| `ACT` | Calls a tool (Wikipedia, others coming) | "Tell me about the Eiffel Tower" |
| `REACT` | Multi-step reasoning with tools | "Plan a weekend trip to Chicago" |
| `DOCS_QA` | RAG: retrieves from docs, generates cited answer | "What do the docs say about guardrails?" |
| `ANSWER` | General knowledge answer (with memory context) | "What is the capital of France?" |
| `RESPOND_WITH_MEMORY` | Answer using stored personal info | "What do you know about me?" |
| `IGNORE` | Greeting / small talk | "Hi there" |

---

## Testing & CI

**Two separate GitHub Actions workflows**, because unit tests and eval runs have very different cost profiles:

- **[`ci.yml`](.github/workflows/ci.yml)** — runs 139 mocked unit tests on every push and PR. Free. No API key required. Takes ~15 seconds.
- **[`eval.yml`](.github/workflows/eval.yml)** — runs the full eval suite (Recall@K + faithfulness) on push to `main`. Uses a pre-built test corpus fixture so Recall@K costs nothing; only faithfulness scoring calls the API. Gates merges: Recall@K < 70% or faithfulness < 80% → build fails.

