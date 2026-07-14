"""Unit tests for model routing (Step 28) — agentforge/router.py.

All hermetic: the router is pure Python over config constants, so no API calls.
We patch the router module's tier/policy globals to exercise each branch
independently of the ambient environment.
"""
from unittest.mock import patch

import pytest

from agentforge import router
from agentforge.router import (
    choose_model,
    estimate_difficulty,
    HARD,
    ROUTINE,
    TIER_SMALL,
    TIER_FRONTIER,
)


class TestEstimateDifficulty:

    def test_react_is_hard(self):
        assert estimate_difficulty("REACT") == HARD

    @pytest.mark.parametrize("intent", ["ACT", "ANSWER", "DOCS_QA", "REMEMBER",
                                        "IGNORE", "RESPOND_WITH_MEMORY"])
    def test_other_intents_are_routine(self, intent):
        assert estimate_difficulty(intent) == ROUTINE

    def test_case_insensitive(self):
        assert estimate_difficulty("react") == HARD

    def test_none_is_routine(self):
        # None/unknown counts as routine here; the low-confidence path in
        # choose_model is what forces uncertain turns up, not this function.
        assert estimate_difficulty(None) == ROUTINE
        assert estimate_difficulty("SOMETHING_NEW") == ROUTINE

    def test_hard_set_is_config_driven(self):
        with patch.object(router, "ROUTING_HARD_INTENTS", frozenset({"REACT", "DOCS_QA"})):
            assert estimate_difficulty("DOCS_QA") == HARD
            assert estimate_difficulty("ACT") == ROUTINE


class TestChooseModel:
    """Distinct tier models so the branch under test is unambiguous."""

    def _distinct_tiers(self):
        return patch.multiple(
            router,
            AGENT_MODEL_ROUTING_ENABLED=True,
            MODEL_TIER_SMALL="small-model",
            MODEL_TIER_FRONTIER="frontier-model",
        )

    def test_react_routes_to_frontier(self):
        with self._distinct_tiers():
            d = choose_model("REACT")
        assert d.model == "frontier-model"
        assert d.tier == TIER_FRONTIER
        assert d.difficulty == HARD
        assert d.reason == "hard_intent:REACT"

    def test_routine_routes_to_small(self):
        with self._distinct_tiers():
            d = choose_model("ANSWER")
        assert d.model == "small-model"
        assert d.tier == TIER_SMALL
        assert d.difficulty == ROUTINE
        assert d.reason == "routine_intent:ANSWER"

    def test_low_confidence_forces_frontier_even_for_routine(self):
        # The safety net: an uncertain classification escalates rather than risk a
        # wrong answer on the cheap model — regardless of the (guessed) intent.
        with self._distinct_tiers():
            d = choose_model("ANSWER", low_confidence=True)
        assert d.model == "frontier-model"
        assert d.tier == TIER_FRONTIER
        assert d.reason == "low_confidence_fallback"

    def test_routing_disabled_always_small(self):
        with patch.multiple(router,
                            AGENT_MODEL_ROUTING_ENABLED=False,
                            MODEL_TIER_SMALL="small-model",
                            MODEL_TIER_FRONTIER="frontier-model"):
            # Even REACT and low-confidence stay on small when routing is off —
            # exactly the pre-Step-28 single-model behaviour.
            assert choose_model("REACT").model == "small-model"
            assert choose_model("ANSWER", low_confidence=True).model == "small-model"
            assert choose_model("REACT").reason == "routing_disabled"

    def test_default_config_is_noop_model_but_labels_tier(self):
        # Backward-compat guarantee: when both tiers are the SAME model, a hard turn
        # still gets the "frontier" TIER label (mechanism observable) while the model
        # actually called is unchanged from the single-model default.
        with patch.multiple(router,
                            AGENT_MODEL_ROUTING_ENABLED=True,
                            MODEL_TIER_SMALL="gpt-4o-mini",
                            MODEL_TIER_FRONTIER="gpt-4o-mini"):
            d = choose_model("REACT")
        assert d.model == "gpt-4o-mini"
        assert d.tier == TIER_FRONTIER


class TestClassifyIntentConfidenceFlag:
    """The router's low_confidence input comes from classify_intent (main.py)."""

    def test_default_intent_is_low_confidence(self):
        from agentforge.main import _default_intent
        assert _default_intent("anything")["low_confidence"] is True
