"""Judge-agreement eval (issue #95): how often does the faithfulness judge agree
with a reference label?

WHY THIS EXISTS:
    The faithfulness score comes from an LLM judge (``evaluation.score_faithfulness``).
    A score from a judge that was never compared with human labels is a number
    nobody can defend. This module measures the judge itself.

WHAT IT USES:
    A gold set (``tests/judge_gold_dataset.json``). Each case FREEZES the
    question, the chunks and the answer, so a run measures only the judge — not
    retrieval and not the answering model. Each case carries a reference label
    and ``labelled_by``, which says who gave it: "human" or "ai". Only labels
    by a human support a claim of human agreement; the report prints who
    labelled, so a number from AI labels is never read as one from people.
    About half the cases are real pipeline answers; the other half were written
    to be hard (a changed number, an added fact, outside knowledge, a heavy
    paraphrase, a refusal), because real answers are mostly faithful, and
    agreement on a set where one label dominates says very little.

    Every case sits in a fixed half, ``dev`` or ``test``, chosen before the
    first run. Change the judge prompt using ``dev`` only; report from ``test``.
    A prompt tuned on the cases it is scored on only looks better.

WHAT IT REPORTS:
    - Cohen's kappa: agreement with the agreement expected by chance removed.
      Raw agreement alone is misleading: two labellers who each say "faithful"
      90% of the time agree 82% of the time by chance.
    - A 95% bootstrap interval for kappa. With about 60 cases it is wide; the
      report shows it, so nobody reads more into the number than is there.
    - The confusion matrix and both error rates: unfaithful answers the judge
      passed, and faithful answers the judge flagged.
    - Judge failures (API error, unusable reply) as their own outcome. They
      stay in the denominator and always count as a disagreement.
    - Consistency: the share of cases where every run gave the same verdict.
    - The settings behind the numbers: model, temperature, prompt hash.

TWO COMMANDS (both dispatched from ``agentforge.evaluation``):
    python -m agentforge.evaluation --label-gold
        Label cases one at a time. The judge's verdict is never shown.
    python -m agentforge.evaluation --judge-agreement [--runs=3] [--split=all]
        Run the judge on the labelled cases (live; a few cents) and gate on kappa.
"""
import hashlib
import json
import logging
import os
import random
import sys
from collections import Counter
from typing import Callable, Optional

logger = logging.getLogger(__name__)

DEFAULT_GOLD_PATH = os.path.join(
    os.path.dirname(__file__), "..", "tests", "judge_gold_dataset.json"
)

FAITHFUL = "faithful"
UNFAITHFUL = "unfaithful"
JUDGE_ERROR = "error"          # the judge failed; never a reference label
LABELS = (FAITHFUL, UNFAITHFUL)
# Who gave a reference label. "ai" = an AI assistant read the case (for a
# written case, the label is also the error that was planted on purpose).
HUMAN = "human"
AI = "ai"
LABELLERS = (HUMAN, AI)
SPLITS = ("dev", "test")
SOURCES = ("pipeline", "authored")
REQUIRED_FIELDS = {"id", "question", "chunks", "answer", "split", "source", "category"}

# Fixed before the first measured run (roadmap Step 32a). Kappa 0.6 is the
# usual lower edge of "substantial" agreement.
KAPPA_TARGET = 0.6
# 0 makes a verdict as repeatable as the API allows, so two runs can be compared.
JUDGE_TEMPERATURE = 0
DEFAULT_RUNS = 3

# Shown to the person labelling. It restates the judge's own rules, so the
# human and the judge answer the SAME question; otherwise kappa would measure
# two definitions, not the judge.
LABELLING_RULES = """\
You decide one thing per case: is the ANSWER faithful to the CHUNKS?

  faithful    Every claim in the answer is supported by at least one chunk.
              Rephrasing and summarising are fine. An answer that says it does
              not have enough information is faithful. An answer that leaves
              something out is still faithful: this is not about completeness.
  unfaithful  The answer states at least one fact, number or name that is not
              in any chunk, even if that fact is true in the real world.

Judge only against the chunks shown, not against what you know."""


# ---------------------------------------------------------------------------
# Gold set: load, validate, save
# ---------------------------------------------------------------------------

def load_gold_dataset(path: str = None) -> list[dict]:
    """Load and validate the gold set. Raises ValueError on a malformed case.

    ``reference_label`` may be missing or null (not labelled yet); when present it
    must be "faithful" or "unfaithful", and ``labelled_by`` must say who gave
    it ("human" or "ai").
    """
    path = os.path.abspath(path or DEFAULT_GOLD_PATH)
    logger.info("Loading judge gold set from %s", path)
    with open(path, encoding="utf-8") as f:
        cases = json.load(f)

    if not isinstance(cases, list) or not cases:
        raise ValueError("Judge gold set must be a non-empty JSON array")

    seen = set()
    for i, case in enumerate(cases):
        missing = REQUIRED_FIELDS - set(case)
        if missing:
            raise ValueError(f"Case {i} missing fields: {sorted(missing)}")
        cid = case["id"]
        if cid in seen:
            raise ValueError(f"Duplicate gold id: {cid}")
        seen.add(cid)
        if case["split"] not in SPLITS:
            raise ValueError(f"{cid}: split must be one of {SPLITS}")
        if case["source"] not in SOURCES:
            raise ValueError(f"{cid}: source must be one of {SOURCES}")
        if case.get("reference_label") not in (None, *LABELS):
            raise ValueError(f"{cid}: reference_label must be null or one of {LABELS}")
        if case.get("reference_label") and case.get("labelled_by") not in LABELLERS:
            raise ValueError(f"{cid}: a labelled case needs labelled_by, one of {LABELLERS}")
        chunks = case["chunks"]
        if (not isinstance(chunks, list) or not chunks
                or any(not isinstance(c, dict) or "id" not in c or "text" not in c
                       for c in chunks)):
            raise ValueError(f"{cid}: chunks must be a non-empty list of {{id, text}}")
        for field in ("question", "answer"):
            if not isinstance(case[field], str) or not case[field].strip():
                raise ValueError(f"{cid}: {field} must be a non-empty string")

    labelled = sum(1 for c in cases if c.get("reference_label"))
    logger.info("Loaded %d gold cases (%d labelled)", len(cases), labelled)
    return cases


def save_gold_dataset(cases: list[dict], path: str = None) -> None:
    """Write the gold set. Writes a temp file first and then replaces the real
    one, so stopping in the middle of a labelling session cannot leave a
    half-written file."""
    path = os.path.abspath(path or DEFAULT_GOLD_PATH)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)
    logger.info("Saved %d gold cases to %s", len(cases), path)


# ---------------------------------------------------------------------------
# Pure metrics — no API, no files
# ---------------------------------------------------------------------------

def confusion_matrix(human: list[str], judge: list[str]) -> dict:
    """Count (human label, judge label) pairs.

    Rows are the human label, columns the judge's. The judge has one more
    column than the human: "error", for a run where the judge failed.

    Returns {"faithful": {"faithful": n, "unfaithful": n, "error": n},
             "unfaithful": {...}}.
    """
    if len(human) != len(judge):
        raise ValueError("human and judge must have the same length")
    matrix = {h: {j: 0 for j in (*LABELS, JUDGE_ERROR)} for h in LABELS}
    for h, j in zip(human, judge):
        if h not in LABELS:
            raise ValueError(f"human label must be one of {LABELS}, got {h!r}")
        if j not in (*LABELS, JUDGE_ERROR):
            raise ValueError(f"judge label must be faithful/unfaithful/error, got {j!r}")
        matrix[h][j] += 1
    return matrix


def cohens_kappa(human: list[str], judge: list[str]) -> Optional[float]:
    """Cohen's kappa: (observed agreement - chance agreement) / (1 - chance).

    Chance agreement is what two labellers with these label rates would reach
    without looking at the cases. A judge failure ("error") is its own judge
    category; a human never uses it, so it adds nothing to chance agreement and
    always counts as a disagreement — failures stay in the denominator.

    Returns None when kappa is undefined: no cases, or both sides gave one and
    the same label to every case (chance agreement is then 100%, and there is
    nothing left to measure). None is deliberate: 1.0 there would be a lie.
    """
    if len(human) != len(judge):
        raise ValueError("human and judge must have the same length")
    n = len(human)
    if n == 0:
        return None
    observed = sum(1 for h, j in zip(human, judge) if h == j) / n
    human_counts, judge_counts = Counter(human), Counter(judge)
    chance = sum(
        (human_counts[c] / n) * (judge_counts[c] / n)
        for c in set(human_counts) | set(judge_counts)
    )
    if chance >= 1.0:
        return None
    return (observed - chance) / (1.0 - chance)


def kappa_interval(
    human: list[str], judge: list[str], resamples: int = 2000, seed: int = 0,
) -> Optional[tuple]:
    """95% bootstrap interval for kappa: resample the cases with replacement,
    recompute kappa each time, and take the 2.5th and 97.5th percentiles.

    Seeded, so the same labels always give the same interval. Returns None when
    kappa itself is undefined or too few resamples have a defined kappa.
    """
    n = len(human)
    if cohens_kappa(human, judge) is None:
        return None
    rng = random.Random(seed)
    values = []
    for _ in range(resamples):
        picks = [rng.randrange(n) for _ in range(n)]
        k = cohens_kappa([human[i] for i in picks], [judge[i] for i in picks])
        if k is not None:
            values.append(k)
    if len(values) < resamples // 2:
        return None
    values.sort()
    low = values[int(0.025 * (len(values) - 1))]
    high = values[int(0.975 * (len(values) - 1))]
    return (low, high)


def agreement_report(human: list[str], judge: list[str]) -> dict:
    """Everything about one run's agreement, from the two label lists.

    Both error rates are reported, never one alone:
      - missed_unfaithful_rate: of the answers a human called unfaithful, the
        share the judge passed as faithful (the dangerous direction: a bad
        answer scores as good).
      - wrongly_flagged_rate: of the answers a human called faithful, the share
        the judge called unfaithful.
    """
    matrix = confusion_matrix(human, judge)
    n = len(human)
    agree = matrix[FAITHFUL][FAITHFUL] + matrix[UNFAITHFUL][UNFAITHFUL]
    reference_unfaithful = sum(matrix[UNFAITHFUL].values())
    reference_faithful = sum(matrix[FAITHFUL].values())
    interval = kappa_interval(human, judge)
    return {
        "n": n,
        "agreement": agree / n if n else None,
        "kappa": cohens_kappa(human, judge),
        "kappa_interval_95": list(interval) if interval else None,
        "matrix": matrix,
        "reference_faithful": reference_faithful,
        "reference_unfaithful": reference_unfaithful,
        "missed_unfaithful_rate": (
            matrix[UNFAITHFUL][FAITHFUL] / reference_unfaithful if reference_unfaithful else None),
        "wrongly_flagged_rate": (
            matrix[FAITHFUL][UNFAITHFUL] / reference_faithful if reference_faithful else None),
        "judge_errors": matrix[FAITHFUL][JUDGE_ERROR] + matrix[UNFAITHFUL][JUDGE_ERROR],
    }


def prompt_fingerprint(prompt: str) -> str:
    """Short hash of the judge prompt, stored with every report: a number is
    only comparable with another number from the same prompt."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Live run
# ---------------------------------------------------------------------------

def _judge_label(verdict: dict) -> str:
    """Map one judge result to faithful / unfaithful / error."""
    if not isinstance(verdict, dict) or verdict.get("error"):
        return JUDGE_ERROR
    return FAITHFUL if verdict.get("faithful") is True else UNFAITHFUL


def run_judge_agreement(
    cases: list[dict],
    runs: int = DEFAULT_RUNS,
    model: str = None,
    temperature: float = JUDGE_TEMPERATURE,
    judge: Callable = None,
    verbose: bool = True,
) -> dict:
    """Run the judge ``runs`` times on every LABELLED case and compare with
    the reference label.

    ``judge`` defaults to ``evaluation.score_faithfulness``; tests pass a fake.
    Unlabelled cases are skipped and counted. One failing judge call becomes an
    "error" verdict for that case and run — it never stops the loop.

    Returns a dict with per-run reports, the mean kappa (None when any run's
    kappa is undefined), consistency across runs, agreement by source and by
    category, every case's verdicts, the disagreements, and the settings.
    """
    from agentforge.config import OPENAI_MODEL
    from agentforge import evaluation

    if runs < 1:
        raise ValueError("runs must be at least 1")
    judge = judge or evaluation.score_faithfulness
    labelled = [c for c in cases if c.get("reference_label") in LABELS]
    skipped = len(cases) - len(labelled)
    logger.info("Judge agreement: %d labelled cases, %d unlabelled skipped, %d run(s)",
                len(labelled), skipped, runs)

    per_case = [
        {"id": c["id"], "split": c["split"], "source": c["source"],
         "category": c["category"], "reference_label": c["reference_label"],
         "judge_labels": [], "judge_reasons": []}
        for c in labelled
    ]
    for run in range(runs):
        for case, row in zip(labelled, per_case):
            try:
                verdict = judge(case["question"], case["answer"], case["chunks"],
                                model=model, temperature=temperature)
            except Exception as exc:  # the judge catches its own errors; this is a second net
                logger.error("Judge call raised for %s: %s", case["id"], type(exc).__name__)
                verdict = {"faithful": False, "reason": "", "error": type(exc).__name__}
            label = _judge_label(verdict)
            row["judge_labels"].append(label)
            row["judge_reasons"].append(str((verdict or {}).get("reason", "")))
            if verbose:
                mark = "ok  " if label == row["reference_label"] else "DIFF"
                print(f"  [{mark}] run {run + 1} {case['id']}: "
                      f"reference={row['reference_label']}, judge={label}")

    human = [row["reference_label"] for row in per_case]
    per_run = [
        agreement_report(human, [row["judge_labels"][run] for row in per_case])
        for run in range(runs)
    ]
    kappas = [r["kappa"] for r in per_run]
    kappa_mean = sum(kappas) / len(kappas) if kappas and None not in kappas else None

    for row in per_case:
        row["consistent"] = len(set(row["judge_labels"])) == 1
    consistent = sum(1 for row in per_case if row["consistent"])

    def _grouped(key: str) -> dict:
        # Agreement only: kappa on a group of five cases would be noise.
        groups: dict = {}
        for row in per_case:
            g = groups.setdefault(row[key], {"cases": 0, "agree": 0, "verdicts": 0})
            g["cases"] += 1
            g["verdicts"] += len(row["judge_labels"])
            g["agree"] += sum(1 for j in row["judge_labels"] if j == row["reference_label"])
        return groups

    result = {
        "settings": {
            "model": model or OPENAI_MODEL,
            "temperature": temperature,
            "prompt_sha256_12": prompt_fingerprint(evaluation.FAITHFULNESS_JUDGE_PROMPT),
            "runs": runs,
            "kappa_target": KAPPA_TARGET,
        },
        "labelled": len(labelled),
        "labelled_by": dict(Counter(c.get("labelled_by") or "unknown" for c in labelled)),
        "skipped_unlabelled": skipped,
        "per_run": per_run,
        "kappa_mean": kappa_mean,
        "consistency": consistent / len(per_case) if per_case else None,
        "by_source": _grouped("source"),
        "by_category": _grouped("category"),
        "per_case": per_case,
        "disagreements": [
            row for row in per_case
            if any(j != row["reference_label"] for j in row["judge_labels"])
        ],
        "gate_passed": kappa_mean is not None and kappa_mean >= KAPPA_TARGET,
    }
    logger.info(
        "Judge agreement complete: kappa mean %s, %d of %d cases consistent, gate %s",
        "n/a" if kappa_mean is None else f"{kappa_mean:.2f}",
        consistent, len(per_case), "passed" if result["gate_passed"] else "failed",
    )
    return result


# ---------------------------------------------------------------------------
# Labelling
# ---------------------------------------------------------------------------

def label_gold_interactive(
    path: str = None,
    input_fn: Callable = input,
    print_fn: Callable = print,
    relabel: bool = False,
    seed: int = 0,
) -> dict:
    """Label gold cases one at a time and save after every answer.

    The case's source, category and the judge's verdict are NOT shown: the
    label must come from the question, the chunks and the answer alone. Cases
    come in a fixed shuffled order, so written and real cases are mixed.

    Keys: f = faithful, u = unfaithful, s = skip for now, q = save and stop.
    ``relabel=True`` goes through every case again, labelled or not.

    Returns {"labelled": n, "skipped": n, "remaining": n}.
    """
    cases = load_gold_dataset(path)
    todo = [c for c in cases if relabel or c.get("reference_label") not in LABELS]
    random.Random(seed).shuffle(todo)
    logger.info("Labelling session: %d case(s) to label of %d", len(todo), len(cases))

    done = skipped = 0
    if not todo:
        print_fn("Every case already has a label. Use --relabel to go through them again.")
        return {"labelled": 0, "skipped": 0, "remaining": 0}

    print_fn(LABELLING_RULES)
    print_fn("\nKeys: f = faithful, u = unfaithful, s = skip, q = save and stop.")
    print_fn("Labels and notes are saved in the gold set file, which is part of the repo.")

    for position, case in enumerate(todo, start=1):
        print_fn(f"\n{'=' * 70}\nCase {position} of {len(todo)}  ({case['id']})\n{'=' * 70}")
        print_fn(f"QUESTION:\n{case['question']}\n")
        print_fn("CHUNKS:")
        for chunk in case["chunks"]:
            print_fn(f"[{chunk['id']}] {chunk['text']}\n")
        print_fn(f"ANSWER:\n{case['answer']}\n")

        choice = ""
        while choice not in ("f", "u", "s", "q"):
            choice = input_fn("faithful / unfaithful / skip / quit [f/u/s/q]: ").strip().lower()[:1]
        if choice == "q":
            break
        if choice == "s":
            skipped += 1
            continue
        case["reference_label"] = FAITHFUL if choice == "f" else UNFAITHFUL
        case["labelled_by"] = HUMAN      # this command is only ever run by a person
        note = input_fn("Note (optional, Enter for none): ").strip()
        if note:
            case["note"] = note
        save_gold_dataset(cases, path)
        done += 1

    remaining = sum(1 for c in cases if c.get("reference_label") not in LABELS)
    logger.info("Labelling session ended: %d labelled, %d skipped, %d without a label",
                done, skipped, remaining)
    print_fn(f"\nLabelled {done}, skipped {skipped}. Cases without a label: {remaining}.")
    return {"labelled": done, "skipped": skipped, "remaining": remaining}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _flag(argv: list, name: str, default=None):
    for arg in argv:
        if arg.startswith(f"--{name}="):
            return arg.split("=", 1)[1]
    return default


def _pct(value) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def _print_report(result: dict) -> None:
    s = result["settings"]
    print(f"\n{'=' * 60}")
    print(f"  Judge: {s['model']}, temperature {s['temperature']}, "
          f"prompt {s['prompt_sha256_12']}")
    print(f"  Cases: {result['labelled']} labelled "
          f"({result['skipped_unlabelled']} unlabelled skipped), {s['runs']} run(s)")
    by = result.get("labelled_by", {})
    print("  Reference labels by: "
          + ", ".join(f"{who} {n}" for who, n in sorted(by.items())))
    if set(by) != {HUMAN}:
        print("  NOTE: not every label is from a human. This is agreement with "
              "reference labels,\n        not agreement with people.")
    for i, r in enumerate(result["per_run"], start=1):
        kappa = "n/a" if r["kappa"] is None else f"{r['kappa']:.2f}"
        ci = r["kappa_interval_95"]
        ci_text = f" (95% interval {ci[0]:.2f} to {ci[1]:.2f})" if ci else ""
        print(f"\n  Run {i}: kappa {kappa}{ci_text}, raw agreement {_pct(r['agreement'])}")
        m = r["matrix"]
        print(f"    {'':22s}{'judge faithful':>16s}{'judge unfaithful':>18s}{'judge error':>13s}")
        for h in LABELS:
            print(f"    {'reference ' + h:22s}{m[h][FAITHFUL]:>16d}"
                  f"{m[h][UNFAITHFUL]:>18d}{m[h][JUDGE_ERROR]:>13d}")
        print(f"    Unfaithful answers the judge passed: {_pct(r['missed_unfaithful_rate'])} "
              f"of {r['reference_unfaithful']}")
        print(f"    Faithful answers the judge flagged : {_pct(r['wrongly_flagged_rate'])} "
              f"of {r['reference_faithful']}")
        print(f"    Judge failures: {r['judge_errors']}")

    print(f"\n  Same verdict in every run: {_pct(result['consistency'])} of cases")
    for title, key in (("By source", "by_source"), ("By category", "by_category")):
        print(f"\n  {title} (verdicts that match the reference label):")
        for name, g in sorted(result[key].items()):
            print(f"    {name:24s}: {g['agree']}/{g['verdicts']}  ({g['cases']} cases)")
    print(f"{'=' * 60}")

    # Read the misses — never report only the headline.
    if result["disagreements"]:
        print(f"\n  {len(result['disagreements'])} case(s) where the judge differed "
              f"from the reference label in at least one run:")
        for row in result["disagreements"]:
            print(f"    {row['id']} [{row['source']}/{row['category']}] "
                  f"reference={row['reference_label']}, judge={row['judge_labels']}")
            print(f"      judge said: {row['judge_reasons'][0][:200]}")


def main(argv: Optional[list] = None) -> int:
    """CLI for both commands. Returns the exit code.

    --label-gold [--relabel] [--gold=PATH]
    --judge-agreement [--runs=N] [--split=dev|test|all] [--model=ID]
                      [--out=PATH] [--gold=PATH]
        0 = kappa at or above the target, 1 = below it or undefined,
        2 = nothing to measure (no labelled case in the chosen split).
    """
    argv = sys.argv if argv is None else argv
    gold_path = _flag(argv, "gold")

    # A Windows console (cp1252) cannot print every character a chunk or a
    # judge reason may hold; show "?" for those instead of crashing mid-session.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    if "--label-gold" in argv:
        label_gold_interactive(gold_path, relabel="--relabel" in argv)
        return 0

    runs = int(_flag(argv, "runs", DEFAULT_RUNS))
    split = _flag(argv, "split", "all")
    model = _flag(argv, "model")
    out_path = _flag(argv, "out")
    if split not in (*SPLITS, "all"):
        print(f"--split must be one of {(*SPLITS, 'all')}")
        return 2

    cases = load_gold_dataset(gold_path)
    if split != "all":
        cases = [c for c in cases if c["split"] == split]
    labelled = [c for c in cases if c.get("reference_label") in LABELS]
    if not labelled:
        print(f"No labelled case in split '{split}'. Label some first:\n"
              f"  python -m agentforge.evaluation --label-gold")
        return 2

    print(f"\n{'=' * 60}\n  Judge-agreement eval (live; {len(labelled)} cases x "
          f"{runs} run(s), split: {split})\n{'=' * 60}\n")
    result = run_judge_agreement(cases, runs=runs, model=model)
    result["settings"]["split"] = split

    if out_path:
        # Every verdict of this run, so later numbers and charts come from the same run.
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nSaved every verdict of this run to {out_path}")

    _print_report(result)

    kappa = result["kappa_mean"]
    if kappa is None:
        print(f"\nFAIL: kappa is undefined (every label on one side is the same). "
              f"The gold set needs both faithful and unfaithful cases.")
        return 1
    if result["gate_passed"]:
        print(f"\nPASS: mean kappa {kappa:.2f} meets the target ({KAPPA_TARGET}).")
        return 0
    print(f"\nFAIL: mean kappa {kappa:.2f} is below the target ({KAPPA_TARGET}).")
    return 1
