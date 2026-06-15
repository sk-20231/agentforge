"""Tests for the red-team eval harness (agentforge.redteam).

Hermetic: no model, no dataset download. We verify the metrics MATH (the part that
must be exactly right for the published numbers to be trustworthy) on hand-built
labels/scores, plus the threshold-sweep shape and the score-collection guard.
"""
from unittest.mock import patch

import pytest

from agentforge import guardrail, redteam
from agentforge.guardrail import GuardrailResult, Verdict


class TestComputeMetrics:
    # labels  = [1,   1,   0,   0]   (1 = injection)
    # scores  = [0.9, 0.4, 0.6, 0.1]
    # @0.5 -> preds = [1, 0, 1, 0]  => TP=1 FN=1 FP=1 TN=1
    LABELS = [1, 1, 0, 0]
    SCORES = [0.9, 0.4, 0.6, 0.1]

    def test_confusion_counts(self):
        m = redteam.compute_metrics(self.LABELS, self.SCORES, 0.5)
        assert (m.tp, m.fn, m.fp, m.tn) == (1, 1, 1, 1)
        assert m.n == 4

    def test_rates(self):
        m = redteam.compute_metrics(self.LABELS, self.SCORES, 0.5)
        assert m.detection_rate == 0.5        # TP / (TP+FN) = 1/2
        assert m.false_positive_rate == 0.5   # FP / (FP+TN) = 1/2
        assert m.precision == 0.5             # TP / (TP+FP) = 1/2
        assert m.f1 == 0.5
        assert m.accuracy == 0.5              # (TP+TN)/n = 2/4

    def test_threshold_at_zero_flags_everything(self):
        # Every score >= 0 -> all predicted injection: catch all, but FPR = 100%.
        m = redteam.compute_metrics(self.LABELS, self.SCORES, 0.0)
        assert m.detection_rate == 1.0
        assert m.false_positive_rate == 1.0
        assert (m.tp, m.fn, m.fp, m.tn) == (2, 0, 2, 0)

    def test_high_threshold_flags_nothing(self):
        m = redteam.compute_metrics(self.LABELS, self.SCORES, 0.99)
        assert m.detection_rate == 0.0
        assert m.false_positive_rate == 0.0
        assert (m.tp, m.fp) == (0, 0)

    def test_no_divide_by_zero_when_one_class_absent(self):
        # All legit, none injection -> recall denom is 0, must not raise.
        m = redteam.compute_metrics([0, 0], [0.9, 0.1], 0.5)
        assert m.detection_rate == 0.0   # safe_div guard
        assert m.false_positive_rate == 0.5


class TestThresholdSweep:
    def test_sweep_returns_one_row_per_threshold(self):
        sweep = redteam.threshold_sweep([1, 0], [0.8, 0.2], thresholds=[0.1, 0.5, 0.9])
        assert [m.threshold for m in sweep] == [0.1, 0.5, 0.9]

    def test_detection_is_monotonic_non_increasing_as_threshold_rises(self):
        labels = [1, 1, 1, 0, 0, 0]
        scores = [0.2, 0.5, 0.8, 0.3, 0.6, 0.9]
        sweep = redteam.threshold_sweep(labels, scores)
        dets = [m.detection_rate for m in sweep]
        assert all(dets[i] >= dets[i + 1] for i in range(len(dets) - 1))


class TestScoreExamples:
    def test_uses_guardrail_score_and_collects_labels(self):
        def fake_scan(text, **kw):
            return GuardrailResult(Verdict.BLOCK if "attack" in text else Verdict.ALLOW,
                                   score=0.9 if "attack" in text else 0.1)
        examples = [("attack now", 1), ("hello", 0)]
        with patch("agentforge.guardrail.scan_external_text", side_effect=fake_scan):
            scores, labels = redteam.score_examples(examples)
        assert scores == [0.9, 0.1]
        assert labels == [1, 0]

    def test_aborts_if_model_unavailable(self):
        with patch("agentforge.guardrail.scan_external_text",
                   return_value=GuardrailResult(Verdict.UNAVAILABLE, reason="not installed")):
            with pytest.raises(RuntimeError, match="unavailable"):
                redteam.score_examples([("x", 1)])
