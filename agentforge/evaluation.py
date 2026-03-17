"""
Evaluation utilities for measuring RAG quality.

Step 8: Dataset loader and validator.
Step 9: recall_at_k (retrieval quality).
Step 10: faithfulness scoring (answer quality via LLM-as-judge).
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


# ---------------------------------------------------------------------------
# Step 10 — Answer faithfulness (LLM-as-judge)
# ---------------------------------------------------------------------------

FAITHFULNESS_JUDGE_PROMPT = """\
You are an impartial judge evaluating whether an AI assistant's answer is \
faithful to the provided source chunks.

"Faithful" means the answer ONLY contains information that is supported by \
the chunks. It does NOT add facts, numbers, names, or claims that cannot be \
found in the chunks.

You will receive:
- The user's QUESTION
- The CHUNKS that were given to the assistant
- The assistant's ANSWER

Respond in JSON with exactly two fields:
{
  "faithful": true or false,
  "reason": "one-sentence explanation of your verdict"
}

Rules:
- If every claim in the answer can be traced to at least one chunk, it is faithful.
- Minor rephrasing or summarising is fine — that is still faithful.
- If the answer says "I don't have enough information", that is faithful.
- If the answer adds ANY fact not present in any chunk, it is NOT faithful.
"""


def score_faithfulness(
    question: str,
    answer: str,
    chunks: list[dict],
) -> dict:
    """Use an LLM to judge whether the answer is faithful to the chunks.

    Returns {"faithful": bool, "reason": str}.
    Falls back to {"faithful": False, "reason": "..."} on any error.
    """
    from openai import OpenAI
    from agentforge.config import OPENAI_MODEL, OPENAI_BASE_URL

    _client = OpenAI(base_url=OPENAI_BASE_URL) if OPENAI_BASE_URL else OpenAI()

    chunks_text = "\n\n---\n\n".join(
        f"[{c['id']}] {c['text']}" for c in chunks
    )

    user_message = (
        f"QUESTION:\n{question}\n\n"
        f"CHUNKS:\n{chunks_text}\n\n"
        f"ANSWER:\n{answer}"
    )

    try:
        response = _client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": FAITHFULNESS_JUDGE_PROMPT},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        return {
            "faithful": bool(data.get("faithful", False)),
            "reason": data.get("reason", ""),
        }
    except Exception as exc:
        logger.error("score_faithfulness failed: %s", exc)
        return {"faithful": False, "reason": f"Judge error: {exc}"}


def run_faithfulness_eval(
    dataset: list[dict],
    top_k: int = 5,
    verbose: bool = True,
) -> dict:
    """Run the full RAG pipeline for each eval question, then judge faithfulness.

    For each question:
      1. search_docs → get retrieved chunks
      2. answer_from_docs → get the generated answer (non-streaming)
      3. score_faithfulness → LLM-as-judge verdict

    Returns a dict with:
      - faithfulness: overall score (float, 0.0–1.0)
      - total: number of questions evaluated
      - faithful_count: how many were faithful
      - per_question: list of per-question results
      - by_difficulty: faithfulness broken down by difficulty label
    """
    from agentforge.rag.document_store import search_docs
    from agentforge.rag.qa import answer_from_docs

    per_question = []
    faithful_count = 0
    difficulty_buckets: dict[str, dict] = {}

    for i, entry in enumerate(dataset):
        question = entry["question"]
        difficulty = entry.get("difficulty", "unknown")

        if verbose:
            print(f"[{i+1}/{len(dataset)}] {entry['id']} ({difficulty})")
            print(f"  Q: {question}")

        chunks = search_docs(question, top_k=top_k)
        answer = answer_from_docs(question, top_k=top_k, stream=False)

        verdict = score_faithfulness(question, answer, chunks)
        is_faithful = verdict["faithful"]

        if is_faithful:
            faithful_count += 1

        detail = {
            "id": entry["id"],
            "question": question,
            "difficulty": difficulty,
            "answer": answer,
            "faithful": is_faithful,
            "reason": verdict["reason"],
            "retrieved_ids": [c["id"] for c in chunks],
        }
        per_question.append(detail)

        if difficulty not in difficulty_buckets:
            difficulty_buckets[difficulty] = {"faithful": 0, "total": 0}
        difficulty_buckets[difficulty]["total"] += 1
        if is_faithful:
            difficulty_buckets[difficulty]["faithful"] += 1

        if verbose:
            status = "FAITHFUL" if is_faithful else "UNFAITHFUL"
            print(f"  A: {answer[:120]}{'...' if len(answer) > 120 else ''}")
            print(f"  [{status}] {verdict['reason']}")
            print()

    total = len(dataset)
    overall = faithful_count / total if total else 0.0
    by_difficulty = {
        d: b["faithful"] / b["total"] if b["total"] else 0.0
        for d, b in difficulty_buckets.items()
    }

    logger.info(
        "Faithfulness eval complete: %.0f%% (%d/%d)",
        overall * 100, faithful_count, total,
    )

    return {
        "faithfulness": overall,
        "total": total,
        "faithful_count": faithful_count,
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

    # --- Parse flags ---
    run_recall = "--eval" in sys.argv
    run_faith = "--faithfulness" in sys.argv

    if not run_recall and not run_faith:
        print("\nTo run evaluations, add flags:")
        print("  python -m agentforge.evaluation --eval              # Recall@K only")
        print("  python -m agentforge.evaluation --faithfulness      # Faithfulness only")
        print("  python -m agentforge.evaluation --eval --faithfulness  # Both")
        print("  Add --top-k=N to change K (default 5)")
        sys.exit(0)

    top_k = 5
    for arg in sys.argv:
        if arg.startswith("--top-k="):
            top_k = int(arg.split("=")[1])

    RECALL_THRESHOLD = 0.7
    FAITH_THRESHOLD = 0.8
    failed = False

    # --- Retrieval eval ---
    if run_recall:
        print(f"\n{'='*60}")
        print(f"  Running Retrieval Eval (Recall@{top_k})")
        print(f"{'='*60}\n")

        recall_result = run_retrieval_eval(dataset, top_k=top_k, verbose=True)

        print(f"{'='*60}")
        print(f"  Recall@{top_k}: {recall_result['recall']:.0%} ({recall_result['total_hits']}/{recall_result['total_expected']})")
        print()
        print("  By difficulty:")
        for diff, score in sorted(recall_result["by_difficulty"].items()):
            print(f"    {diff:8s}: {score:.0%}")
        print(f"{'='*60}")

        missed_questions = [q for q in recall_result["per_question"] if q["missed"]]
        if missed_questions:
            print(f"\n  {len(missed_questions)} question(s) with missed chunks:")
            for q in missed_questions:
                print(f"    {q['id']}: missed {q['missed']}")

        if recall_result["recall"] < RECALL_THRESHOLD:
            print(f"\nFAIL: Recall@{top_k} ({recall_result['recall']:.0%}) below threshold ({RECALL_THRESHOLD:.0%})")
            failed = True
        else:
            print(f"\nPASS: Recall@{top_k} ({recall_result['recall']:.0%}) meets threshold ({RECALL_THRESHOLD:.0%})")

    # --- Faithfulness eval ---
    if run_faith:
        print(f"\n{'='*60}")
        print(f"  Running Faithfulness Eval (LLM-as-judge)")
        print(f"{'='*60}\n")

        faith_result = run_faithfulness_eval(dataset, top_k=top_k, verbose=True)

        print(f"{'='*60}")
        print(f"  Faithfulness: {faith_result['faithfulness']:.0%} ({faith_result['faithful_count']}/{faith_result['total']})")
        print()
        print("  By difficulty:")
        for diff, score in sorted(faith_result["by_difficulty"].items()):
            print(f"    {diff:8s}: {score:.0%}")
        print(f"{'='*60}")

        unfaithful = [q for q in faith_result["per_question"] if not q["faithful"]]
        if unfaithful:
            print(f"\n  {len(unfaithful)} unfaithful answer(s):")
            for q in unfaithful:
                print(f"    {q['id']}: {q['reason']}")

        if faith_result["faithfulness"] < FAITH_THRESHOLD:
            print(f"\nFAIL: Faithfulness ({faith_result['faithfulness']:.0%}) below threshold ({FAITH_THRESHOLD:.0%})")
            failed = True
        else:
            print(f"\nPASS: Faithfulness ({faith_result['faithfulness']:.0%}) meets threshold ({FAITH_THRESHOLD:.0%})")

    sys.exit(1 if failed else 0)
