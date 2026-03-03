# AgentForge

A learning-focused AI agent built from scratch with Python and the OpenAI API. The goal is to understand core AI/agent concepts by implementing them step by step — not by using high-level frameworks, but by writing the logic yourself.

## What it does

An interactive CLI agent that can:

- **Classify intent** — Routes each user message to the right pipeline (remember, act, answer, reason, search docs, etc.)
- **Remember things** — Stores long-term personal facts using semantic memory (embeddings + cosine similarity)
- **Use tools** — Calls deterministic tools like a calculator when needed
- **Multi-step reasoning** — Uses a ReAct (Reason + Act) loop for complex tasks
- **Answer from documents (RAG)** — Ingests files, chunks and embeds them, retrieves relevant chunks, and generates cited answers with a guardrail that strips hallucinated citations
- **Memory-aware responses** — Uses stored personal info to personalize answers
- **Multi-turn conversation** — Maintains conversation history across turns; trims to a token budget so cost stays predictable
- **Conversation-aware RAG** — Rewrites follow-up questions into standalone queries before retrieval so "How does it work?" retrieves correctly even in a 10-turn conversation

## AI concepts covered

| Concept | Where in the code | When to use it |
|---|---|---|
| **Intent classification** | `agentforge/main.py` — `classify_intent` | When one input can mean different actions |
| **Semantic memory** | `agentforge/memory/semantic.py` | When you need long-term, queryable user context |
| **Embeddings + similarity** | `agentforge/memory/semantic.py` — `get_embedding`, `cosine_similarity` | When you need "find things similar to this" |
| **Tool calling** | `agentforge/tools.py` | When the model needs deterministic or live data |
| **ReAct reasoning** | `agentforge/reasoning/react_engine.py` | When a task needs multiple think-act-observe steps |
| **RAG (retrieval-augmented generation)** | `agentforge/rag/` | When answers must be grounded in a corpus |
| **Chunking** | `agentforge/rag/document_store.py` — `chunk_text` | First step of any RAG pipeline |
| **Citation guardrails** | `agentforge/rag/qa.py` — `_strip_invalid_citations` | When output must satisfy hard constraints |
| **Conversation buffer** | `agentforge/main.py` — CLI loop + `run_agent` | When the model needs prior turns to resolve follow-ups |
| **Token budget / context window management** | `agentforge/conversation.py` — `trim_history` | When history grows too long and cost/latency must be controlled |
| **Query rewriting (contextualization)** | `agentforge/conversation.py` — `rewrite_query` | When follow-up questions contain pronouns that break RAG retrieval |

## Project structure

```
agentforge/
├── agentforge/                   # Main package
│   ├── config.py                 # Environment-based configuration
│   ├── conversation.py           # Token budget trimming + query rewriting
│   ├── logger.py                 # Event logging (JSONL)
│   ├── main.py                   # Intent classification + agent orchestrator + CLI
│   ├── prompts.py                # System prompts and structured output schema
│   ├── tools.py                  # Tool registry, calculator, LLM tool calling
│   ├── memory/                   # Semantic memory sub-package
│   │   ├── semantic.py           # Embeddings, similarity, memory store/retrieve
│   │   └── response.py           # Memory-aware answer generation
│   ├── rag/                      # RAG sub-package
│   │   ├── document_store.py     # Chunking, corpus load/save, search, file ingestion
│   │   └── qa.py                 # RAG answer pipeline + citation guardrail
│   └── reasoning/                # Reasoning sub-package
│       └── react_engine.py       # ReAct loop (think → act → observe → repeat)
├── tests/                        # Unit tests (mocked, no API calls)
├── docs/                         # Roadmaps and design docs
├── run.py                        # Thin CLI entry point
├── requirements.txt              # Production dependencies
├── requirements-dev.txt          # Test dependencies
├── .env.example                  # Environment variable template
└── .gitignore
```

## Setup

### 1. Clone and create a virtual environment

```bash
git clone <your-repo-url>
cd agentforge
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt   # for tests
```

### 3. Set your OpenAI API key

Copy `.env.example` to `.env` and add your key:

```bash
cp .env.example .env
```

Then edit `.env`:

```
OPENAI_API_KEY=sk-...
```

## Usage

### Run the agent (interactive CLI)

```bash
python run.py
```

You'll be prompted for a user ID, then you can chat:

```
Enter user id: alice
alice> I like coffee and hiking
Agent: Got it! I'll remember that.
alice> What's 42 * 17?
Agent: 42 * 17 = 714
alice> What do you know about me?
Agent: You like coffee and hiking.
```

### Ingest a document for RAG

```bash
python -m agentforge.rag.document_store path/to/file.txt
```

Then ask questions about it:

```
alice> According to the docs, what is chunking?
Agent: Chunking is the process of splitting long text into smaller pieces... [ROADMAP_DOCUMENT_QA_chunk_3]
```

### Run tests

```bash
pytest tests/ -v
```

## Configuration

All settings are in `agentforge/config.py` and read from environment variables (with defaults):

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | (required) | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | Chat model |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `OPENAI_BASE_URL` | (OpenAI default) | Custom API base URL |
| `AGENT_MEMORY_DIR` | `memory` | Directory for user memory files |
| `AGENT_LOG_FILE` | `agent_logs.jsonl` | Event log file |
| `AGENT_CORPUS_FILE` | `corpus.json` | RAG document corpus file |
| `HISTORY_TOKEN_BUDGET` | `2000` | Max tokens kept in conversation history before trimming |

## Intent routing

The agent classifies every message into one of these intents:

| Intent | What happens | Example |
|---|---|---|
| `REMEMBER` | Stores a personal fact in semantic memory | "I prefer dark roast coffee" |
| `ACT` | Calls a tool (e.g. calculator) | "What's 15% of 230?" |
| `REACT` | Multi-step reasoning with tools | "Plan a weekend trip to Chicago" |
| `DOCS_QA` | RAG: retrieves from docs, generates cited answer | "What do the docs say about guardrails?" |
| `ANSWER` | General knowledge answer (with memory context) | "What is the capital of France?" |
| `RESPOND_WITH_MEMORY` | Answer using stored personal info | "What do you know about me?" |
| `IGNORE` | Greeting / small talk | "Hi there" |

## Roadmaps

- [Document Q&A with Citations](docs/ROADMAP_DOCUMENT_QA.md) — step-by-step RAG implementation guide
- [Breadth-First AI Engineering Tour](docs/ROADMAP_BREADTH_TOUR.md) — conversation, streaming, structured outputs, evaluation, UI, observability, and portfolio polish (in progress)
