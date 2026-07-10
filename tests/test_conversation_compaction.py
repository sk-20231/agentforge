"""Tests for compact_history (context engineering, Step 18a).

Compaction summarises the OLDEST conversation turns into a single running
summary instead of deleting them (contrast trim_history). These tests mock the
OpenAI client so no API calls are made, and shrink COMPACTION_SUMMARY_MAX_TOKENS
so the keep/drop split is exercised with small fixtures.
"""
from unittest.mock import patch, MagicMock

import pytest

from agentforge import conversation
from agentforge.conversation import compact_history, _is_summary, _SUMMARY_PREFIX


def _make_mock_llm_response(text: str):
    msg = MagicMock()
    msg.content = text
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


def _history(n_pairs: int):
    """n_pairs user+assistant turns, each ~15 estimated tokens."""
    msgs = []
    for i in range(n_pairs):
        msgs.append({"role": "user", "content": f"user message number {i} with some padding text"})
        msgs.append({"role": "assistant", "content": f"assistant reply number {i} with some padding text"})
    return msgs


@pytest.fixture(autouse=True)
def _small_summary_reserve(monkeypatch):
    # Shrink the reserve so keep_budget > 0 and the split keeps a few recent turns
    # verbatim instead of collapsing to the last pair.
    monkeypatch.setattr(conversation, "COMPACTION_SUMMARY_MAX_TOKENS", 20)


def test_empty_history_returns_empty():
    assert compact_history([], budget=100) == []


@patch("agentforge.conversation._client")
def test_under_budget_returns_unchanged_and_makes_no_llm_call(mock_client):
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    result = compact_history(history, budget=10_000)
    assert result == history
    mock_client.chat.completions.create.assert_not_called()


@patch("agentforge.conversation._client")
def test_over_budget_folds_oldest_into_a_summary_message(mock_client):
    mock_client.chat.completions.create.return_value = _make_mock_llm_response(
        "User is Sayali; building a Python RAG app."
    )
    history = _history(8)  # ~240 tokens, well over budget
    result = compact_history(history, budget=100)

    # First message is the tagged running summary carrying the model's text.
    assert _is_summary(result[0])
    assert result[0]["content"].startswith(_SUMMARY_PREFIX)
    assert "Sayali" in result[0]["content"]
    # The most recent turn is kept verbatim.
    assert result[-1] == history[-1]
    # Result fits the budget, and exactly one summary call was made.
    assert conversation.count_tokens(result) <= 100
    mock_client.chat.completions.create.assert_called_once()


@patch("agentforge.conversation._client")
def test_prior_summary_is_merged_not_stacked(mock_client):
    mock_client.chat.completions.create.return_value = _make_mock_llm_response(
        "Merged running summary."
    )
    history = [
        {"role": "system", "content": f"{_SUMMARY_PREFIX}\nEarlier: user is Sayali."},
    ] + _history(8)

    result = compact_history(history, budget=100)

    # Only ONE summary message survives (never a stack of them).
    assert sum(1 for m in result if _is_summary(m)) == 1
    # The prior summary was fed into the summariser payload for merging.
    payload = mock_client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    assert "Previous summary:" in payload
    assert "Sayali" in payload


@patch("agentforge.conversation._client")
def test_llm_failure_degrades_to_delete_oldest_trim(mock_client):
    mock_client.chat.completions.create.side_effect = RuntimeError("boom")
    history = _history(8)

    result = compact_history(history, budget=100)  # must not raise

    # No summary added on failure (no prior summary existed); recent turns kept,
    # oldest dropped — i.e. it degraded to trim_history behaviour.
    assert not any(_is_summary(m) for m in result)
    assert result[-1] == history[-1]
    assert len(result) < len(history)
