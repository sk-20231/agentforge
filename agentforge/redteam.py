"""Red-team evaluation of the content guardrail (Step 17e gap E follow-up).

WHY this exists:
    We added an injection-classifier guardrail (``agentforge.guardrail``) but adding
    a security control is not the same as knowing it works. This harness MEASURES
    that layer against a held-out public prompt-injection dataset and reports two
    numbers that only mean something together:
      - detection rate  — share of real injections it CATCHES, and
      - false-positive rate — share of safe text it WRONGLY blocks.
    Blocking everything gets 100%% detection and is useless; you need both.

    This is "test my own defenses," not a verdict on any vendor. Frame results that
    way wherever they're shared.

METHOD (state these caveats with any published number):
    - Dataset: ``deepset/prompt-injections`` (Apache-2.0): column ``text`` + ``label``
      (1 = injection, 0 = legit), ~662 rows.
    - HELD-OUT: this dataset is NOT in the default model's listed training data
      (ProtectAI deberta-v3-base-prompt-injection-v2), so it approximates unseen
      data. We cannot guarantee zero example overlap; we disclose that.
    - We measure the CLASSIFIER LAYER ONLY — not the full agent or the full
      deterministic+HITL defence stack. The full-stack eval is a follow-up.

EGRESS:
    The classifier runs locally; the only network is a one-time download of the
    public dataset (same class as the model download). No agent/LLM API calls.

Run:  python -m agentforge.redteam            (uses AGENT_GUARDRAIL_THRESHOLD)
      python -m agentforge.redteam --limit 100 --out redteam_report
"""
import argparse
import json
import logging
from dataclasses import asdict, dataclass
from typing import List, Optional, Sequence, Tuple

from agentforge import guardrail
from agentforge.config import AGENT_GUARDRAIL_THRESHOLD

logger = logging.getLogger(__name__)

_DATASET_REPO = "deepset/prompt-injections"


# --------------------------------------------------------------------------- #
# Data loading — held-out public dataset via huggingface_hub + pandas.        #
# --------------------------------------------------------------------------- #

def load_examples(repo: str = _DATASET_REPO,
                  limit: Optional[int] = None) -> List[Tuple[str, int]]:
    """Return ``[(text, label)]`` from the public dataset (label 1 = injection).

    Loads the parquet files directly with ``huggingface_hub`` + ``pandas`` (no
    ``datasets`` dependency). One-time download of public data; cached afterward.
    Raises with a clear message if the data libs aren't installed.
    """
    try:
        import pandas as pd
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError as exc:
        raise RuntimeError(
            "Red-team eval needs the data libs. Install with: "
            'pip install -e ".[redteam]"'
        ) from exc

    parquet_files = [
        f for f in list_repo_files(repo, repo_type="dataset") if f.endswith(".parquet")
    ]
    if not parquet_files:
        raise RuntimeError(f"No parquet files found in dataset repo '{repo}'.")

    frames = []
    for fname in sorted(parquet_files):
        local = hf_hub_download(repo, fname, repo_type="dataset")
        frames.append(pd.read_parquet(local))
    df = pd.concat(frames, ignore_index=True)

    if "text" not in df.columns or "label" not in df.columns:
        raise RuntimeError(f"Unexpected dataset columns: {list(df.columns)}")

    examples = [(str(t), int(l)) for t, l in zip(df["text"], df["label"]) if str(t).strip()]
    if limit is not None:
        examples = examples[:limit]
    return examples


# --------------------------------------------------------------------------- #
# Scoring + metrics.                                                          #
# --------------------------------------------------------------------------- #

def score_examples(examples: Sequence[Tuple[str, int]]) -> Tuple[List[float], List[int]]:
    """Run the guardrail on each text; return ``(malicious_scores, labels)``.

    Uses ``guardrail.scan_external_text`` whose ``.score`` is the malicious
    probability regardless of verdict — so we can sweep thresholds afterward
    without re-running the model. Aborts if the model is unavailable (an eval with
    no classifier is meaningless).
    """
    scores: List[float] = []
    labels: List[int] = []
    for i, (text, label) in enumerate(examples):
        result = guardrail.scan_external_text(text)
        if result.verdict == guardrail.Verdict.UNAVAILABLE:
            raise RuntimeError(
                "Guardrail model unavailable — cannot run the eval. Install the "
                'classifier with: pip install -e ".[guardrail]" and download the model. '
                f"(reason: {result.reason})"
            )
        scores.append(result.score)
        labels.append(int(label))
        if (i + 1) % 100 == 0:
            logger.info("scored %d/%d", i + 1, len(examples))
    return scores, labels


@dataclass
class Metrics:
    """Binary-classification metrics at one threshold (positive class = injection)."""
    threshold: float
    n: int
    tp: int
    fp: int
    tn: int
    fn: int
    detection_rate: float   # recall on injections = TP / (TP + FN)
    false_positive_rate: float  # FP / (FP + TN)
    precision: float
    f1: float
    accuracy: float


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def compute_metrics(labels: Sequence[int], scores: Sequence[float],
                    threshold: float) -> Metrics:
    """Confusion-matrix + rates at ``threshold`` (prediction = score >= threshold)."""
    tp = fp = tn = fn = 0
    for label, score in zip(labels, scores):
        predicted_injection = score >= threshold
        if label == 1 and predicted_injection:
            tp += 1
        elif label == 1 and not predicted_injection:
            fn += 1
        elif label == 0 and predicted_injection:
            fp += 1
        else:
            tn += 1
    detection = _safe_div(tp, tp + fn)
    fpr = _safe_div(fp, fp + tn)
    precision = _safe_div(tp, tp + fp)
    f1 = _safe_div(2 * precision * detection, precision + detection)
    accuracy = _safe_div(tp + tn, tp + fp + tn + fn)
    return Metrics(
        threshold=round(threshold, 4), n=len(labels), tp=tp, fp=fp, tn=tn, fn=fn,
        detection_rate=round(detection, 4), false_positive_rate=round(fpr, 4),
        precision=round(precision, 4), f1=round(f1, 4), accuracy=round(accuracy, 4),
    )


def threshold_sweep(labels: Sequence[int], scores: Sequence[float],
                    thresholds: Optional[Sequence[float]] = None) -> List[Metrics]:
    """Metrics across a range of thresholds (the detection vs false-positive tradeoff)."""
    if thresholds is None:
        thresholds = [i / 20 for i in range(1, 20)]  # 0.05 .. 0.95
    return [compute_metrics(labels, scores, t) for t in thresholds]


# --------------------------------------------------------------------------- #
# Chart (optional) + CLI.                                                      #
# --------------------------------------------------------------------------- #

def make_chart(sweep: Sequence[Metrics], path: str) -> bool:
    """Save a detection-vs-false-positive sweep chart. Returns False if matplotlib
    isn't installed (numbers still print; the PNG is the only thing skipped)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping chart. (pip install matplotlib)")
        return False
    xs = [m.threshold for m in sweep]
    plt.figure(figsize=(7, 4.5))
    plt.plot(xs, [m.detection_rate for m in sweep], marker="o", label="Detection rate (injections caught)")
    plt.plot(xs, [m.false_positive_rate for m in sweep], marker="o", label="False-positive rate (safe text blocked)")
    plt.xlabel("Decision threshold")
    plt.ylabel("Rate")
    plt.title("Injection guardrail: detection vs false positives")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return True


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Red-team eval of the content guardrail.")
    parser.add_argument("--limit", type=int, default=None, help="cap number of examples")
    parser.add_argument("--threshold", type=float, default=AGENT_GUARDRAIL_THRESHOLD,
                        help="primary decision threshold (default = AGENT_GUARDRAIL_THRESHOLD)")
    parser.add_argument("--out", type=str, default="redteam_report",
                        help="output basename (.json + .png)")
    args = parser.parse_args(argv)

    logger.info("Loading held-out dataset '%s'...", _DATASET_REPO)
    examples = load_examples(limit=args.limit)
    n_inj = sum(1 for _, l in examples if l == 1)
    logger.info("Loaded %d examples (%d injection, %d legit). Scoring...",
                len(examples), n_inj, len(examples) - n_inj)

    scores, labels = score_examples(examples)
    primary = compute_metrics(labels, scores, args.threshold)
    sweep = threshold_sweep(labels, scores)

    report = {
        "dataset": _DATASET_REPO,
        "n": primary.n,
        "n_injection": n_inj,
        "n_legit": primary.n - n_inj,
        "primary": asdict(primary),
        "sweep": [asdict(m) for m in sweep],
        "note": ("classifier layer only; held-out dataset not in the model's listed "
                 "training data (overlap not guaranteed zero)"),
    }
    with open(f"{args.out}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    charted = make_chart(sweep, f"{args.out}.png")

    p = primary
    logger.info("\n===== Red-team result (threshold %.2f) =====", p.threshold)
    logger.info("Examples: %d  (%d injection / %d legit)", p.n, n_inj, p.n - n_inj)
    logger.info("Detection rate (injections caught): %.1f%%", p.detection_rate * 100)
    logger.info("False-positive rate (safe blocked):  %.1f%%", p.false_positive_rate * 100)
    logger.info("Precision: %.1f%%   F1: %.3f   Accuracy: %.1f%%",
                p.precision * 100, p.f1, p.accuracy * 100)
    logger.info("Confusion: TP=%d FP=%d TN=%d FN=%d", p.tp, p.fp, p.tn, p.fn)
    logger.info("Report: %s.json%s", args.out, f"  Chart: {args.out}.png" if charted else "")


if __name__ == "__main__":
    main()
