"""Pricing resolution for DATED model ids (issue #39).

``log_token_usage`` priced each call by exact lookup in ``MODEL_COSTS``, falling
back to gpt-4o-mini rates on a miss. The OpenAI API returns dated ids
("gpt-4o-2024-08-06"), which never match the unversioned keys — so every call was
priced at mini rates, measured ~9x under on a frontier turn. Harmless while both
tiers were mini; wrong the moment AGENT_MODEL_FRONTIER names a distinct model,
which is the entire point of Step 28.
"""
import json
from unittest.mock import MagicMock, patch

import pytest

import agentforge.logger as logger_mod
from agentforge.logger import (
    MODEL_COSTS,
    _costs_for_model,
    _FALLBACK_COST_MODEL,
    compute_cost_summary,
    log_token_usage,
)


@pytest.fixture(autouse=True)
def _reset_warn_dedup():
    """The warned-ids set is module state; keep tests independent of each other."""
    logger_mod._UNPRICED_MODELS_WARNED.clear()
    yield
    logger_mod._UNPRICED_MODELS_WARNED.clear()


@pytest.fixture(autouse=True)
def _temp_log(tmp_path, monkeypatch):
    """Write to a temp log rather than the developer's real one.

    Self-contained on purpose: this must hold regardless of whether the suite-wide
    redirect fixture (issue #43) is present, and a test-level patch overrides it
    anyway once it is.
    """
    monkeypatch.setattr(logger_mod, "AGENT_LOG_FILE", str(tmp_path / "agent_logs.jsonl"))


def _response(model, prompt_tokens, completion_tokens):
    response = MagicMock()
    response.model = model
    response.usage.prompt_tokens = prompt_tokens
    response.usage.completion_tokens = completion_tokens
    return response


class TestCostsForModel:
    def test_exact_id_resolves_to_itself(self):
        assert _costs_for_model("gpt-4o") is MODEL_COSTS["gpt-4o"]

    def test_dated_frontier_id_resolves_to_frontier_rates(self):
        """The bug: this used to fall through to mini rates."""
        assert _costs_for_model("gpt-4o-2024-08-06") is MODEL_COSTS["gpt-4o"]

    def test_dated_mini_id_prefers_the_longer_prefix(self):
        """"gpt-4o-mini-2024-07-18" prefixes BOTH "gpt-4o" and "gpt-4o-mini"."""
        assert _costs_for_model("gpt-4o-mini-2024-07-18") is MODEL_COSTS["gpt-4o-mini"]

    def test_dated_gpt4_id_does_not_match_gpt4o(self):
        assert _costs_for_model("gpt-4-0613") is MODEL_COSTS["gpt-4"]

    def test_dated_gpt35_id_resolves(self):
        assert _costs_for_model("gpt-3.5-turbo-0125") is MODEL_COSTS["gpt-3.5-turbo"]

    @pytest.mark.parametrize("model", ["", None, "claude-opus-5", "text-embedding-3-small"])
    def test_unknown_ids_fall_back(self, model):
        assert _costs_for_model(model, warn_on_fallback=False) is MODEL_COSTS[_FALLBACK_COST_MODEL]


class TestFallbackIsLoud:
    def test_unknown_model_emits_a_warning_event(self):
        with patch.object(logger_mod, "log_event") as mock_log:
            _costs_for_model("some-unknown-model")

        mock_log.assert_called_once()
        event_type, payload = mock_log.call_args[0][0], mock_log.call_args[0][1]
        assert event_type == "cost_model_unpriced"
        assert payload["model"] == "some-unknown-model"
        assert payload["priced_as"] == _FALLBACK_COST_MODEL

    def test_warning_is_emitted_once_per_model_id(self):
        with patch.object(logger_mod, "log_event") as mock_log:
            for _ in range(5):
                _costs_for_model("some-unknown-model")

        assert mock_log.call_count == 1

    def test_distinct_unknown_ids_each_warn(self):
        with patch.object(logger_mod, "log_event") as mock_log:
            _costs_for_model("unknown-a")
            _costs_for_model("unknown-b")

        assert mock_log.call_count == 2

    def test_known_model_does_not_warn(self):
        with patch.object(logger_mod, "log_event") as mock_log:
            _costs_for_model("gpt-4o-2024-08-06")

        mock_log.assert_not_called()

    def test_read_path_does_not_warn(self):
        with patch.object(logger_mod, "log_event") as mock_log:
            _costs_for_model("some-unknown-model", warn_on_fallback=False)

        mock_log.assert_not_called()


class TestLoggedCostIsCorrect:
    def test_dated_frontier_call_is_priced_at_frontier_rates(self, tmp_path):
        """The measurement from the issue: react_step_1 on gpt-4o, 861 + 192 tokens."""
        log_token_usage(_response("gpt-4o-2024-08-06", 861, 192), "react_step_1")

        record = json.loads((tmp_path / "agent_logs.jsonl").read_text(encoding="utf-8").strip())
        expected = 861 * MODEL_COSTS["gpt-4o"]["prompt"] + 192 * MODEL_COSTS["gpt-4o"]["completion"]

        assert record["payload"]["cost_usd"] == pytest.approx(round(expected, 6))

    def test_pins_the_size_of_the_correction(self, tmp_path):
        """Pin the error the fix removes, so a silent regression is obvious.

        Per FRONTIER CALL the correction is 16.7x (the gpt-4o : gpt-4o-mini rate
        ratio, identical on both prompt and completion). The ~9x in issue #39 is a
        different figure — the aggregate over a mixed run of one mini turn and one
        4o turn — so the two numbers are consistent, not contradictory.
        """
        log_token_usage(_response("gpt-4o-2024-08-06", 861, 192), "react_step_1")

        record = json.loads((tmp_path / "agent_logs.jsonl").read_text(encoding="utf-8").strip())
        mini = MODEL_COSTS["gpt-4o-mini"]
        old_cost = 861 * mini["prompt"] + 192 * mini["completion"]

        assert record["payload"]["cost_usd"] / old_cost == pytest.approx(16.7, rel=0.05)

    def test_dated_mini_call_is_still_priced_at_mini_rates(self, tmp_path):
        """Backward compatibility: the default no-op tier setup must not change price."""
        log_token_usage(_response("gpt-4o-mini-2024-07-18", 1000, 500), "intent_classification")

        record = json.loads((tmp_path / "agent_logs.jsonl").read_text(encoding="utf-8").strip())
        mini = MODEL_COSTS["gpt-4o-mini"]
        expected = 1000 * mini["prompt"] + 500 * mini["completion"]

        assert record["payload"]["cost_usd"] == pytest.approx(round(expected, 6))

    def test_cost_summary_reflects_the_corrected_price(self, tmp_path):
        log_token_usage(_response("gpt-4o-2024-08-06", 861, 192), "react_step_1")

        summary = compute_cost_summary(str(tmp_path / "agent_logs.jsonl"))
        expected = 861 * MODEL_COSTS["gpt-4o"]["prompt"] + 192 * MODEL_COSTS["gpt-4o"]["completion"]

        assert summary["total"]["cost_usd"] == pytest.approx(round(expected, 6), rel=1e-3)
