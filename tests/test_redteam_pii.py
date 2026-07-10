"""Tests for the output-guardrail PII eval harness (agentforge.redteam_pii, issue #24).

Hermetic — the harness is pinned to the regex engine so the confusion-matrix math and
the per-type / over-redaction accounting are deterministic without loading any model.
The point of these tests is the MEASUREMENT logic, not the detector quality.
"""
import pytest

from agentforge import output_guardrail
from agentforge import redteam_pii
from agentforge.redteam_pii import PII_EVAL_EXAMPLES, PiiExample, evaluate


@pytest.fixture(autouse=True)
def _force_regex_engine(monkeypatch):
    """Grade against the deterministic regex engine regardless of [pii] being installed."""
    monkeypatch.setattr(output_guardrail, "AGENT_OUTPUT_GUARDRAIL_ENGINE", "regex")


def _recall_by_type(report):
    return {t.entity_type: t.recall for t in report.type_recall}


class TestConfusionMatrixMath:
    def test_counts_on_a_tiny_known_set(self):
        examples = [
            PiiExample("Reach me at a@b.com please.", frozenset({"EMAIL"})),  # TP
            PiiExample("Please contact John Smith.", frozenset({"PERSON"})),  # FN (regex blind)
            PiiExample("The sky is blue today.", frozenset()),               # TN
            PiiExample("Order 1234567890123456 shipped.", frozenset()),      # FP (card over-match)
        ]
        m = evaluate(examples, aggressive=False).metrics
        assert (m.tp, m.fn, m.tn, m.fp) == (1, 1, 1, 1)
        assert m.n == 4 and m.n_pii == 2 and m.n_clean == 2
        assert m.detection_rate == 0.5      # 1 / (1+1)
        assert m.false_positive_rate == 0.5  # 1 / (1+1)

    def test_clean_only_set_has_no_false_negatives(self):
        examples = [PiiExample("Just a normal sentence.", frozenset())]
        m = evaluate(examples).metrics
        assert (m.tp, m.fn, m.tn, m.fp) == (0, 0, 1, 0)
        assert m.detection_rate == 0.0  # no positives -> safe-divide to 0


class TestRegexBaselineOnAuthoredSet:
    """The authored dataset must expose the regex engine's known shape."""

    def test_structured_pii_fully_caught_freeform_fully_missed(self):
        report = evaluate(PII_EVAL_EXAMPLES, aggressive=False)
        recall = _recall_by_type(report)
        # Regex nails structured PII...
        for t in ("EMAIL", "SSN", "SECRET", "CREDIT_CARD"):
            assert recall[t] == 1.0, f"{t} should be fully caught by regex"
        # ...and is structurally blind to names/addresses (this is the #24 gap).
        assert recall["PERSON"] == 0.0
        assert recall["ADDRESS"] == 0.0
        assert report.metrics.fn > 0

    def test_card_regex_over_redacts_long_digit_runs(self):
        # The known no-Luhn bug must show up as over-redactions on clean digit runs.
        report = evaluate(PII_EVAL_EXAMPLES, aggressive=False)
        over_texts = " ".join(text for text, _t in report.over_redactions)
        assert "1234567890123456" in over_texts


def test_active_engine_reports_regex_when_forced(monkeypatch):
    monkeypatch.setattr(redteam_pii, "AGENT_OUTPUT_GUARDRAIL_ENGINE", "regex")
    assert redteam_pii._active_engine() == "regex"
