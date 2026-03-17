# agent/rag/document_store.py
"""
Document corpus for RAG: chunking, load/save, search, ingest.
Steps 2–6: chunk_text, corpus load/save, add_document, search_docs, ingest_file.
"""
import json
import os
import re
import sys

from agentforge.config import AGENT_CORPUS_FILE
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
    """
    path = AGENT_CORPUS_FILE
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except FileNotFoundError:
        return []
    except (json.JSONDecodeError, TypeError):
        return []


def save_corpus(corpus: list[dict]) -> None:
    """
    Write the corpus (list of {id, text, embedding, source}) to the JSON file.
    Overwrites the file. Use the path from config (AGENT_CORPUS_FILE).
    """
    path = AGENT_CORPUS_FILE
    with open(path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2)


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


if __name__ == "__main__":
    # CLI mode: ingest a file from disk
    #   python -m agentforge.rag.document_store path/to/file.txt [optional_doc_id]
    if len(sys.argv) >= 2:
        path_arg = sys.argv[1]
        id_arg = sys.argv[2] if len(sys.argv) >= 3 else None
        ingest_file(path_arg, id_arg)
    else:
        # No arguments: show a chunking demo (read-only, never touches corpus.json)
        sample = """
        First paragraph. It is short.

        Second paragraph is much longer. It has many words so that we can see how
        chunking works when a single paragraph exceeds the max_chars limit. We want
        to split it into overlapping pieces so that no important context is lost
        at the boundary between one chunk and the next. Overlap helps the model
        see a bit of the previous chunk when answering from the next one.

        Third paragraph. Also short.
        """
        result = chunk_text(sample.strip(), max_chars=120, overlap=30)
        print(f"Got {len(result)} chunks:")
        for i, c in enumerate(result):
            print(f"  [{i}] ({len(c)} chars): {c[:60]}...")

        corpus = load_corpus()
        print(f"\nCurrent corpus: {len(corpus)} chunk(s)")

        print("\nTo ingest a file, run:")
        print("  python -m agentforge.rag.document_store path/to/file.txt [doc_id]")
