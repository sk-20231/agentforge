# Roadmap: Document Q&A with Citations (Small Steps)

Use this roadmap one step at a time: pick a step, study it, ask questions, learn, then move to the next. Nothing is implemented until you choose that step and we do it together.

---

## Step 1: Add config for the corpus path

**What:** Add one setting to `config.py`: `AGENT_CORPUS_FILE` (e.g. default `corpus.json`), read from the environment.

**Why:** The document store needs a single place to read/write the corpus. Config keeps it out of code so you can change it per environment.

**Concepts:** Configuration, separation of concerns.

**Where you’ll look:** `config.py` (with other paths like `AGENT_MEMORY_DIR`, `AGENT_LOG_FILE`).

**Size:** One line of config + one line in the docstring/comments if we add it.

---

## Step 2: Chunking function (no embeddings yet)

**What:** Add a function that splits long text into smaller pieces (e.g. by paragraph, then by max length with overlap). No file I/O, no embeddings—pure text in, list of strings out.

**Why:** RAG works on chunks, not whole docs. Chunking is the first part of “chunk → embed → store → retrieve.”

**Concepts:** RAG pipeline (chunking step), why chunk size and overlap matter.

**Where you’ll look:** New file `agent_document_store.py`, function e.g. `chunk_text(text, max_chars=500, overlap=50)`.

**Size:** One function, ~20–30 lines. Can add a tiny test or a `if __name__ == "__main__"` to try it on a sample string.

---

## Step 3: Corpus load and save (structure only)

**What:** Define the in-memory shape of the corpus (list of items with `id`, `text`, `embedding`, `source`) and add `load_corpus()` and `save_corpus()` that read/write a single JSON file using the path from config.

**Why:** We need a persistent store for chunks and their embeddings. JSON keeps it simple and inspectable.

**Concepts:** Persistence, data shape for RAG (id + text + embedding + source).

**Where you’ll look:** Same file, `load_corpus()`, `save_corpus()`, and the path from `config.AGENT_CORPUS_FILE`.

**Size:** A few functions; no search or embedding yet. You can manually create a minimal JSON to test load/save.

---

## Step 4: Add documents to the corpus (chunk + embed + save)

**What:** Add `add_document(doc_id, text, source)` that: chunks the text, gets an embedding for each chunk (reuse `get_embedding` from semantic memory), and appends `{id, text, embedding, source}` to the corpus and saves.

**Why:** This is the “ingest” step of RAG: turn raw text into searchable chunks with vectors.

**Concepts:** Embeddings (turning text into vectors), reusing existing embedding code, id design (e.g. `doc_id_chunk_0`).

**Where you’ll look:** `agent_document_store.py` — `add_document`, and `agent_semantic_memory.get_embedding`.

**Size:** One function that ties together chunking, embedding, and save. Optional: small script or `__main__` to add one file and inspect `corpus.json`.

---

## Step 5: Search the corpus (retrieve by similarity)

**What:** Add `search_docs(query, top_k=5)` that: embeds the query, compares it to every chunk’s embedding (cosine similarity), and returns the top_k chunks (id, text, source—no need to return the big embedding vector).

**Why:** This is the “retrieve” step of RAG: find which chunks are most relevant to the user’s question.

**Concepts:** Semantic search, cosine similarity, reusing `get_embedding` and your existing similarity helper.

**Where you’ll look:** `agent_document_store.py` — `search_docs`; similarity logic in `agent_semantic_memory` (e.g. `cosine_similarity`).

**Size:** One function. After this step you can test: add a doc, then call `search_docs("some question")` and see which chunks come back.

---

## Step 6: Ingest a file from disk (CLI helper)

**What:** Add `ingest_file(file_path, doc_id)` that reads a text/markdown file and calls `add_document`. Optionally add `if __name__ == "__main__"` so you can run `python -m agent_document_store path/to/file.txt`.

**Why:** Lets you add documents without writing code each time; you can try different files and see how the corpus grows.

**Concepts:** File I/O, encoding (e.g. utf-8, errors="replace"), minimal CLI.

**Where you’ll look:** `agent_document_store.py` — `ingest_file` and bottom-of-file CLI block.

**Size:** Small; builds on Step 4.

---

## Step 7: Answer from docs (RAG + citation guardrail)

**What:** New module (e.g. `agent_docs_qa.py`) with `answer_from_docs(user_input, top_k=5)` that: calls `search_docs`, builds a prompt that includes the retrieved chunks (each labeled with its id), asks the LLM to answer using only those chunks and to cite by id, then post-processes the reply to remove any citation whose id was not in the retrieved set (guardrail).

**Why:** This is the “generate” step of RAG plus safety: answer only from retrieved context and don’t allow invented citations.

**Concepts:** RAG end-to-end (retrieve → prompt → generate), prompt design for citations, guardrails (output validation).

**Where you’ll look:** `agent_docs_qa.py` — full flow; `agent_document_store.search_docs`.

**Size:** One small module, one main function. No wiring to the agent yet; you could call `answer_from_docs("your question")` from a script to test.

---

## Step 8: Add DOCS_QA intent and wire it in the agent

**What:** (1) Add `DOCS_QA` to the list of valid intents in `agent_main.py`. (2) In the intent classifier prompt, add one line describing when to use DOCS_QA (e.g. “user wants an answer from uploaded documents”). (3) In `run_agent`, add a branch: if intent is `DOCS_QA`, call `answer_from_docs(user_input)` and return the result.

**Why:** So the agent can route “answer from my documents” questions to the RAG path instead of generic ANSWER or memory.

**Concepts:** Intent classification, routing, where the agent decides which pipeline runs.

**Where you’ll look:** `agent_main.py` — `VALID_INTENTS`, the classifier prompt string, and the `run_agent` intent switch; import of `answer_from_docs`.

**Size:** A few edits in one file. After this, the full flow works from the CLI: add docs, ask a question that gets classified as DOCS_QA, get a cited answer.

---

## Optional later steps (not part of this roadmap)

- Add a test for `search_docs` (e.g. with a tiny fixed corpus and mocked embedding).
- Add a test for `answer_from_docs` with mocked search and LLM.
- Add `corpus.json` to `.gitignore` if you don’t want to commit the corpus.

---

## How to use this roadmap

1. Read the steps in order.
2. Choose **one** step (e.g. “let’s do Step 2”).
3. We implement only that step; I explain what, why, where to look, and the AI concepts (per your project rule).
4. You study the code, run it if applicable, and ask questions until it’s clear.
5. When you’re ready, pick the next step and repeat.

No step depends on doing everything at once; each step is a small, learnable unit.
