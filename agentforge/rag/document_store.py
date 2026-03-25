# agent/rag/document_store.py
"""
Document corpus for RAG: chunking, load/save, search, ingest.
Steps 2–6: chunk_text, corpus load/save, add_document, search_docs, ingest_file.
"""
import json
import os
import re
import sys

from agentforge.config import AGENT_CORPUS_FILE, OPENAI_EMBEDDING_MODEL
from agentforge.memory.semantic import get_embedding, cosine_similarity


# ---------- Corpus shape ----------
# In-memory corpus: list of items, each a dict with:
#   id: str       — unique chunk id (e.g. "doc123_chunk_0")
#   text: str     — chunk text
#   embedding: list[float] — vector from embedding model (from get_embedding)
#   source: str   — origin (e.g. filename or document id)


# ---------- Chunking ----------

def chunk_text(text: str, max_chars: int = 500, overlap: int = 50) -> list[str]:
    """
    Split long text into smaller pieces for RAG: by paragraph first, then by
    max length with overlap so context isn't lost at boundaries.

    Args:
        text: Raw input text (e.g. from a document).
        max_chars: Maximum characters per chunk. Default 500.
        overlap: Number of characters to overlap between consecutive chunks
                 when a paragraph is longer than max_chars. Default 50.

    Returns:
        List of non-empty chunk strings.
    """
    if not text or not text.strip():
        return []

    # Split into paragraphs (one or more newlines)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []

    for para in paragraphs:
        if len(para) <= max_chars:
            chunks.append(para)
        else:
            # Split long paragraph with overlap
            start = 0
            while start < len(para):
                end = start + max_chars
                chunk = para[start:end].strip()
                if chunk:
                    chunks.append(chunk)
                # Next window: step forward by (max_chars - overlap) so we overlap
                start = end - overlap
                if overlap >= max_chars:
                    start = end  # avoid infinite loop

    return chunks


# ---------- Corpus load / save ----------

def load_corpus() -> list[dict]:
    """
    Load the corpus from the JSON file (path from config).
    Returns a list of items with id, text, embedding, source.
    Returns [] if the file does not exist or is empty/invalid.

    Embedding model validation (fail-fast):
    If the corpus was saved with a different embedding model than the one
    currently configured, raises RuntimeError immediately. Mixing embeddings
    from different models produces meaningless similarity scores — silent
    corruption that's worse than a loud crash.

    Backward compatibility:
    Old corpus files (bare JSON array, no model metadata) are loaded as-is
    without validation. Re-save them via ingest to upgrade to the new format.
    """
    path = AGENT_CORPUS_FILE
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return []
    except (json.JSONDecodeError, TypeError):
        return []

    # New format: {"embedding_model": "...", "chunks": [...]}
    if isinstance(data, dict):
        stored_model = data.get("embedding_model")
        if stored_model and stored_model != OPENAI_EMBEDDING_MODEL:
            raise RuntimeError(
                f"\n"
                f"  ⚠  EMBEDDING MODEL MISMATCH — your corpus is SAFE, nothing has been deleted.\n"
                f"\n"
                f"  Why this is happening:\n"
                f"    Your corpus was built with '{stored_model}'\n"
                f"    but OPENAI_EMBEDDING_MODEL is now set to '{OPENAI_EMBEDDING_MODEL}'.\n"
                f"    Vectors from different models live in incompatible spaces — mixing\n"
                f"    them produces meaningless similarity scores and broken retrieval.\n"
                f"\n"
                f"  If this was unintentional:\n"
                f"    Revert OPENAI_EMBEDDING_MODEL back to '{stored_model}' in config.py\n"
                f"    (or in your .env file) and restart.\n"
                f"\n"
                f"  If you intentionally switched models:\n"
                f"    1. Run this first to see what's in your corpus:\n"
                f"         python -m agentforge.rag.document_store --list-sources\n"
                f"    2. Make sure you still have those files on disk.\n"
                f"    3. Delete corpus.json  (this is permanent — step 1 first!)\n"
                f"    4. Re-ingest each file:\n"
                f"         python -m agentforge.rag.document_store path/to/file.txt\n"
                f"    5. Verify the new corpus:\n"
                f"         python -m agentforge.rag.document_store --list-sources\n"
            )
        return data.get("chunks", [])

    # Old format: bare list — no model validation possible, load as-is
    if isinstance(data, list):
        return data

    return []


def save_corpus(corpus: list[dict]) -> None:
    """
    Write the corpus (list of {id, text, embedding, source}) to the JSON file.
    Stores the current embedding model name alongside the chunks so that
    load_corpus() can detect model mismatches on future loads.
    """
    path = AGENT_CORPUS_FILE
    data = {
        "embedding_model": OPENAI_EMBEDDING_MODEL,
        "chunks": corpus,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ---------- Ingest (chunk + embed + save) ----------

def add_document(doc_id: str, text: str, source: str) -> int:
    """
    Chunk the text, embed each chunk (via get_embedding), append items to the
    corpus, and save. This is the ingest step of RAG: raw text -> searchable chunks.

    Args:
        doc_id: Unique identifier for this document (used as prefix for chunk ids).
        text: Raw document text.
        source: Origin label (e.g. filename or document name) for attribution.

    Returns:
        Number of chunks added. Zero if text is empty or chunks to nothing.
    """
    chunks = chunk_text(text)
    if not chunks:
        return 0

    corpus = load_corpus()
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        corpus.append({
            "id": f"{doc_id}_chunk_{i}",
            "text": chunk,
            "embedding": embedding,
            "source": source,
        })
    save_corpus(corpus)
    return len(chunks)


# ---------- Search (retrieve by similarity) ----------

def search_docs(query: str, top_k: int = 5) -> list[dict]:
    """
    Embed the query, compare to every chunk in the corpus by cosine similarity,
    and return the top_k most relevant chunks. This is the retrieve step of RAG.

    Args:
        query: User's question or search string.
        top_k: Number of results to return.

    Returns:
        List of dicts with id, text, source, score (no embedding vector).
        Sorted by score descending. Empty list if corpus is empty or query is blank.
    """
    if not query or not query.strip():
        return []

    corpus = load_corpus()
    if not corpus:
        return []

    query_embedding = get_embedding(query.strip())

    scored = []
    for item in corpus:
        emb = item.get("embedding")
        if not emb:
            continue
        score = cosine_similarity(query_embedding, emb)
        scored.append({
            "id": item["id"],
            "text": item["text"],
            "source": item.get("source", ""),
            "score": float(score),
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


# ---------- File ingestion (CLI helper) ----------

def ingest_file(file_path: str, doc_id: str = None) -> int:
    """
    Read a text or markdown file from disk and add it to the corpus via
    add_document (chunk -> embed -> save).

    Args:
        file_path: Path to a .txt or .md file.
        doc_id: Optional document id. If not provided, derived from the
                filename without extension (e.g. "notes" from "notes.md").

    Returns:
        Number of chunks added. Zero if file is empty.
    """
    if doc_id is None:
        doc_id = os.path.splitext(os.path.basename(file_path))[0]

    source = os.path.basename(file_path)

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    n = add_document(doc_id, text, source)
    print(f"Ingested '{source}': {n} chunk(s) added (doc_id='{doc_id}').")
    return n


def list_sources() -> list[str]:
    """
    Return the unique source filenames currently in the corpus, sorted.

    Use this BEFORE deleting corpus.json when switching embedding models —
    it tells you exactly which files you'll need to re-ingest.

    Returns [] if the corpus is empty or does not exist.
    Note: raises RuntimeError if corpus exists but has a model mismatch
    (same guard as load_corpus). In that case, read corpus.json directly
    to inspect the 'chunks[*].source' field manually.
    """
    corpus = load_corpus()
    sources = sorted({item.get("source", "") for item in corpus if item.get("source")})
    return sources


if __name__ == "__main__":
    # CLI usage:
    #   python -m agentforge.rag.document_store path/to/file.txt [optional_doc_id]
    #   python -m agentforge.rag.document_store --list-sources
    if len(sys.argv) >= 2 and sys.argv[1] == "--list-sources":
        sources = list_sources()
        corpus = load_corpus()
        print(f"\nCorpus: {len(corpus)} chunk(s) from {len(sources)} source file(s)")
        print(f"Embedding model: {OPENAI_EMBEDDING_MODEL}\n")
        if sources:
            print("Ingested sources (you will need these files to re-ingest):")
            for s in sources:
                count = sum(1 for c in corpus if c.get("source") == s)
                print(f"  {s}  ({count} chunk(s))")
        else:
            print("  (corpus is empty)")
        print()

    elif len(sys.argv) >= 2:
        path_arg = sys.argv[1]
        id_arg = sys.argv[2] if len(sys.argv) >= 3 else None
        ingest_file(path_arg, id_arg)

    else:
        # No arguments: show usage and current corpus stats
        corpus = load_corpus()
        print(f"\nCurrent corpus: {len(corpus)} chunk(s)")
        print(f"Embedding model: {OPENAI_EMBEDDING_MODEL}")
        print("\nUsage:")
        print("  python -m agentforge.rag.document_store path/to/file.txt [doc_id]")
        print("  python -m agentforge.rag.document_store --list-sources")
