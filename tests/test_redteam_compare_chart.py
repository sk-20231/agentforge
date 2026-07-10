"""Tests for the --compare ON-vs-OFF chart (agentforge.redteam_fullstack.make_compare_chart).

Hermetic: builds a synthetic ``--compare`` report dict and renders to a temp file with
matplotlib's Agg backend — no live agent run, no API calls. Skips if matplotlib is absent
(the [redteam] extra), mirroring how the harness itself degrades.
"""
import pytest

from agentforge import redteam_fullstack as rf

pytest.importorskip("matplotlib")


def _combined(on_asr, off_asr, off_layers):
    """A minimal --compare artifact: {classifier_on, classifier_off, asr_gap}.

    The subordinate lower panel is the OFF-run attribution, so the by-layer data
    lives on ``classifier_off`` (that's what make_compare_chart plots)."""
    return {
        "classifier_on": {"n": 28, "asr_overall": on_asr, "blocked_by": {}},
        "classifier_off": {"n": 28, "asr_overall": off_asr, "blocked_by": off_layers},
        "asr_gap": round(off_asr - on_asr, 4),
    }


def test_compare_chart_with_layer_breakdown(tmp_path):
    out = tmp_path / "cmp.png"
    combined = _combined(
        on_asr=0.0, off_asr=0.107,
        off_layers={rf.L_SPOTLIGHT: 16, rf.L_MODEL_REFUSED: 9, rf.L_SUCCEEDED: 3},
    )
    assert rf.make_compare_chart(combined, str(out)) is True
    assert out.exists() and out.stat().st_size > 0


def test_compare_chart_without_layers_still_renders(tmp_path):
    """No OFF by-layer data -> the two-bar primary alone must still render."""
    out = tmp_path / "cmp_nolayers.png"
    combined = _combined(on_asr=0.0, off_asr=0.10, off_layers={})
    assert rf.make_compare_chart(combined, str(out)) is True
    assert out.exists() and out.stat().st_size > 0
