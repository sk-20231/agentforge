"""
Tests for intent classification (classify_intent).
Mocks the OpenAI client so no API calls are made.
"""
import json
import pytest
from unittest.mock import patch, MagicMock

from agentforge.main import classify_intent


def _make_mock_response(json_content: dict):
    """Build a mock OpenAI response with the given JSON as message content."""
    msg = MagicMock()
    msg.content = json.dumps(json_content)
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


@patch("agentforge.main._client")
def test_classify_intent_remember(mock_client):
    mock_client.chat.completions.create.return_value = _make_mock_response({
        "intent": "REMEMBER",
        "memory_candidate": "I like dogs",
        "reason": "User stated a preference",
    })
    result = classify_intent("I like dogs")
    assert result["intent"] == "REMEMBER"
    assert result["memory_candidate"] == "I like dogs"
    assert "reason" in result


@patch("agentforge.main._client")
def test_classify_intent_act(mock_client):
    mock_client.chat.completions.create.return_value = _make_mock_response({
        "intent": "ACT",
        "memory_candidate": "",
        "reason": "Needs calculation",
    })
    result = classify_intent("What is 3 + 5?")
    assert result["intent"] == "ACT"
    assert result["memory_candidate"] == ""


@patch("agentforge.main._client")
def test_classify_intent_ignore(mock_client):
    mock_client.chat.completions.create.return_value = _make_mock_response({
        "intent": "IGNORE",
        "memory_candidate": "",
        "reason": "Greeting",
    })
    result = classify_intent("Hi there")
    assert result["intent"] == "IGNORE"


@patch("agentforge.main._client")
def test_classify_intent_answer(mock_client):
    mock_client.chat.completions.create.return_value = _make_mock_response({
        "intent": "ANSWER",
        "memory_candidate": "",
        "reason": "Informational question",
    })
    result = classify_intent("What is the capital of France?")
    assert result["intent"] == "ANSWER"
