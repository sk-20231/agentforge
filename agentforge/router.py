"""Model routing (Step 28) — difficulty-based dispatch across a small/frontier tier.

The AI concept: **model routing / cascades.** In production, sending every request
to one frontier model is wasteful — most turns are easy. A router estimates each
request's difficulty and sends the easy majority to a small/cheap model, escalating
only the hard minority to a frontier model. Reported bills drop 40–85% with no
visible quality loss (validate on your OWN traffic — those numbers are benchmark-
specific).

What makes it cheap HERE: we do not add a second "difficulty classifier" call. The
intent classifier (agentforge.main.classify_intent) already runs first on every
turn and produces a label — that label IS the difficulty signal. So the estimator
is effectively free.

The safety rule (uncertainty-aware routing): routing must never *silently* lower
quality. When we are not confident about the intent (classification failed or
returned something invalid), we route UP to the frontier tier — the safe default —
rather than down. A cheaper bill is never worth a wrong answer we could have avoided.

Config (agentforge.config):
    AGENT_MODEL_ROUTING_ENABLED  master switch (default on)
    MODEL_TIER_SMALL             cheap model (default = OPENAI_MODEL)
    MODEL_TIER_FRONTIER          frontier model (defaults to SMALL → no-op until set)
    ROUTING_HARD_INTENTS         intents that escalate (default {"REACT"})
"""
from __future__ import annotations

from dataclasses import dataclass

from agentforge.config import (
    AGENT_MODEL_ROUTING_ENABLED,
    MODEL_TIER_SMALL,
    MODEL_TIER_FRONTIER,
    ROUTING_HARD_INTENTS,
)

# Difficulty labels (kept as plain strings so they log/serialize cleanly).
HARD = "hard"
ROUTINE = "routine"

# Tier labels — used by the cost tracker to split spend per tier.
TIER_SMALL = "small"
TIER_FRONTIER = "frontier"


@dataclass(frozen=True)
class RouteDecision:
    """The outcome of one routing decision.

    model      — the concrete model id to call (what the pipeline actually uses).
    tier       — "small" | "frontier" (the human-readable bucket, for cost splits).
    difficulty — "routine" | "hard" (the estimate the decision was based on).
    reason     — one-line explanation, for the audit log and for teaching.
    """
    model: str
    tier: str
    difficulty: str
    reason: str


def estimate_difficulty(intent: str) -> str:
    """Map an intent label to a difficulty estimate.

    REACT (multi-step reasoning/planning) is the one intent we treat as hard by
    default; every other intent — single tool-calls, RAG answers, memory lookups,
    plain answers — is routine. The hard set is config-driven (ROUTING_HARD_INTENTS)
    so it can widen without a code change. ``None``/unknown intents count as routine
    here; the low-confidence path in choose_model() is what forces those UP.
    """
    if intent and intent.upper() in ROUTING_HARD_INTENTS:
        return HARD
    return ROUTINE


def choose_model(intent: str, low_confidence: bool = False) -> RouteDecision:
    """Pick the model + tier for a turn, given its classified intent.

    Order of precedence:
      1. Routing disabled  → always the small tier (preserves the pre-routing
         single-model behaviour exactly).
      2. Low confidence    → frontier tier (the safety net: when we are unsure of
         the intent, spend more rather than risk a wrong answer).
      3. Difficulty        → hard → frontier, routine → small.

    Note: with the default config MODEL_TIER_FRONTIER == MODEL_TIER_SMALL, so the
    *model* is identical for every branch until a real frontier model is configured
    — the decision (tier/difficulty/reason) is still computed and logged, which is
    what makes the mechanism observable even in its no-op default state.
    """
    if not AGENT_MODEL_ROUTING_ENABLED:
        return RouteDecision(MODEL_TIER_SMALL, TIER_SMALL, ROUTINE, "routing_disabled")

    if low_confidence:
        return RouteDecision(
            MODEL_TIER_FRONTIER, TIER_FRONTIER, HARD, "low_confidence_fallback"
        )

    difficulty = estimate_difficulty(intent)
    if difficulty == HARD:
        return RouteDecision(
            MODEL_TIER_FRONTIER, TIER_FRONTIER, HARD, f"hard_intent:{intent}"
        )
    return RouteDecision(
        MODEL_TIER_SMALL, TIER_SMALL, ROUTINE, f"routine_intent:{intent}"
    )
