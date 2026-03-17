"""
Evaluation utilities for measuring RAG quality.

Step 8: Dataset loader and validator.
Step 9: recall_at_k (retrieval quality).
Step 10 will add: faithfulness scoring (answer quality).
"""
import json
import logging
import os

logger = logging.getLogger(__name__)

DEFAULT_EVAL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "tests", "eval_dataset.json"
)

REQUIRED_FIELDS = {"id", "question", "expected_facts", "expected_chunk_ids"}


def load_eval_dataset(path: str = None) -> list[dict]:
    """Load and validate the evaluation dataset from a JSON file.

    Each entry must have: id, question, expected_facts, expected_chunk_ids.
    Raises ValueError if any entry is missing required fields.
    """
    path = path or DEFAULT_EVAL_PATH
    path = os.path.abspath(path)

    logger.info("Loading eval dataset from %s", path)

    with open(path, encoding="utf-8") as f:
        dataset = json.load(f)

    if not isinstance(dataset, list) or len(dataset) == 0:
        raise ValueError(f"Eval dataset must be a non-empty JSON array, got {type(dataset).__name__}")

    seen_ids = set()
    for i, entry in enumerate(dataset):
        missing = REQUIRED_FIELDS - set(entry.keys())
        if missing:
            raise ValueError(f"Entry {i} missing fields: {missing}")

        if entry["id"] in seen_ids:
            raise ValueError(f"Duplicate eval id: {entry['id']}")
        seen_ids.add(entry["id"])

        if not isinstance(entry["expected_facts"], list) or len(entry["expected_facts"]) == 0:
            raise ValueError(f"Entry {entry['id']}: expected_facts must be a non-empty list")

        if not isinstance(entry["expected_chunk_ids"], list) or len(entry["expected_chunk_ids"]) == 0:
            raise ValueError(f"Entry {entry['id']}: expected_chunk_ids must be a non-empty list")

    logger.info("Loaded %d eval examples (IDs: %s)", len(dataset), [e["id"] for e in dataset])
    return dataset


def validate_against_corpus(dataset: list[dict], corpus_path: str = None) -> dict:
    """Check that every expected_chunk_id in the dataset exists in the corpus.

    Returns a dict with 'valid' (bool) and 'missing' (list of {eval_id, chunk_id}).
    """
    from agentforge.config import AGENT_CORPUS_FILE

    corpus_path = corpus_path or AGENT_CORPUS_FILE
    with open(corpus_path, encoding="utf-8") as f:
        corpus = json.load(f)

    corpus_ids = {c["id"] for c in corpus}
    missing = []

    for entry in dataset:
        for cid in entry["expected_chunk_ids"]:
            if cid not in corpus_ids:
                missing.append({"eval_id": entry["id"], "chunk_id": cid})

    if missing:
        logger.warning("Eval dataset references %d chunk IDs not found in corpus", len(missing))
    else:
        logger.info("All expected_chunk_ids validated against corpus (%d chunks)", len(corpus_ids))

    return {"valid": len(missing) == 0, "missing": missing}


# ---------------------------------------------------------------------------
# Step 9 — Retrieval quality (Recall@K)
# ---------------------------------------------------------------------------

def recall_at_k(expected_ids: list[str], retrieved_ids: list[str]) -> float:
    """Fraction of expected chunk IDs that appear anywhere in the retrieved list.

    Recall@K answers: "Of the chunks that *should* have been retrieved,
    how many actually were?"  A score of 1.0 means every expected chunk
    was found in the top-K results.
    """
    if not expected_ids:
        return 1.0
    expected = set(expected_ids)
    retrieved = set(retrieved_ids)
    return len(expected & retrieved) / len(expected)


def run_retrieval_eval(
    dataset: list[dict],
    top_k: int = 5,
    verbose: bool = True,
) -> dict:
    """Run every eval question through search_docs and compute Recall@K.

    Returns a dict with:
      - recall: overall Recall@K (float)
      - total_expected: total number of expected chunks across all questions
      - total_hits: how many were actually retrieved
      - per_question: list of per-question results (id, question, recall, missed)
      - by_difficulty: recall broken down by difficulty label
    """
    from agentforge.rag.document_store import search_docs

    per_question = []
    total_hits = 0
    total_expected = 0
    difficulty_buckets: dict[str, dict] = {}

    for entry in dataset:
        question = entry["question"]
        expected = entry["expected_chunk_ids"]
        difficulty = entry.get("difficulty", "unknown")

        results = search_docs(question, top_k=top_k)
        retrieved_ids = [r["id"] for r in results]

        q_recall = recall_at_k(expected, retrieved_ids)
        found = set(expected) & set(retrieved_ids)
        missed = sorted(set(expected) - set(retrieved_ids))
        hits = len(found)

        total_hits += hits
        total_expected += len(expected)

        detail = {
            "id": entry["id"],
            "question": question,
            "difficulty": difficulty,
            "recall": q_recall,
            "expected": expected,
            "retrieved": retrieved_ids,
            "scores": {r["id"]: round(r["score"], 3) for r in results},
            "missed": missed,
        }
        per_question.append(detail)

        if difficulty not in difficulty_buckets:
            difficulty_buckets[difficulty] = {"hits": 0, "expected": 0}
        difficulty_buckets[difficulty]["hits"] += hits
        difficulty_buckets[difficulty]["expected"] += len(expected)

        if verbose:
            status = "PASS" if not missed else "MISS"
            print(f"[{status}] {entry['id']} ({difficulty})")
            print(f"  Q: {question}")
            print(f"  Expected : {expected}")
            print(f"  Retrieved: {retrieved_ids}")
            scores_str = [f"{r['id']}={r['score']:.3f}" for r in results]
            print(f"  Scores   : {scores_str}")
            if missed:
                print(f"  MISSED   : {missed}")
            print()

    overall_recall = total_hits / total_expected if total_expected else 0.0
    by_difficulty = {
        d: b["hits"] / b["expected"] if b["expected"] else 0.0
        for d, b in difficulty_buckets.items()
    }

    logger.info(
        "Retrieval eval complete: Recall@%d = %.0f%% (%d/%d)",
        top_k, overall_recall * 100, total_hits, total_expected,
    )

    return {
        "recall": overall_recall,
        "top_k": top_k,
        "total_expected": total_expected,
        "total_hits": total_hits,
        "per_question": per_question,
        "by_difficulty": by_difficulty,
    }


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    dataset = load_eval_dataset()
    print(f"\nLoaded {len(dataset)} eval examples:\n")
    for entry in dataset:
        diff = entry.get("difficulty", "?")
        n_chunks = len(entry["expected_chunk_ids"])
        n_facts = len(entry["expected_facts"])
        print(f"  {entry['id']} [{diff}]  {n_chunks} chunks, {n_facts} facts")
        print(f"    Q: {entry['question']}")

    corpus_result = validate_against_corpus(dataset)
    print()
    if corpus_result["valid"]:
        print("All expected_chunk_ids exist in corpus.")
    else:
        print("MISSING chunk IDs:")
        for m in corpus_result["missing"]:
            print(f"  {m['eval_id']}: {m['chunk_id']}")
        print("\nFix missing chunk IDs before running retrieval eval.")
        sys.exit(1)

    # --- Retrieval eval (requires OPENAI_API_KEY for embeddings) ---
    run_eval = "--eval" in sys.argv
    if not run_eval:
        print("\nTo run retrieval eval, add --eval flag:")
        print("  python -m agentforge.evaluation --eval")
        sys.exit(0)

    top_k = 5
    for arg in sys.argv:
        if arg.startswith("--top-k="):
            top_k = int(arg.split("=")[1])

    print(f"\n{'='*60}")
    print(f"  Running Retrieval Eval (Recall@{top_k})")
    print(f"{'='*60}\n")

    result = run_retrieval_eval(dataset, top_k=top_k, verbose=True)

    print(f"{'='*60}")
    print(f"  Recall@{top_k}: {result['recall']:.0%} ({result['total_hits']}/{result['total_expected']})")
    print()
    print("  By difficulty:")
    for diff, score in sorted(result["by_difficulty"].items()):
        print(f"    {diff:8s}: {score:.0%}")
    print(f"{'='*60}")

    missed_questions = [q for q in result["per_question"] if q["missed"]]
    if missed_questions:
        print(f"\n  {len(missed_questions)} question(s) with missed chunks:")
        for q in missed_questions:
            print(f"    {q['id']}: missed {q['missed']}")

    RECALL_THRESHOLD = 0.7
    if result["recall"] < RECALL_THRESHOLD:
        print(f"\nFAIL: Recall@{top_k} ({result['recall']:.0%}) is below threshold ({RECALL_THRESHOLD:.0%})")
        sys.exit(1)
    else:
        print(f"\nPASS: Recall@{top_k} ({result['recall']:.0%}) meets threshold ({RECALL_THRESHOLD:.0%})")
