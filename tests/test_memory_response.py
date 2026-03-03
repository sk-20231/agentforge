"""
Tests for answer_with_memory (agent.memory.response).
Mocks the OpenAI client and uses temp memory so no API calls or real disk.
"""
import pytest
from unittest.mock import patch, MagicMock

from agentforge.memory.response import answer_with_memory
from agentforge.memory.semantic import save_memory, load_memory


def _make_mock_llm_response(text: str):
    msg = MagicMock()
    msg.content = text
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


@patch("agentforge.memory.response.client")
def test_answer_with_memory_no_stored_info_returns_llm_reply(mock_client, temp_memory_dir):
    mock_client.chat.completions.create.return_value = _make_mock_llm_response(
        "I don't have any information about you yet."
    )
    result = answer_with_memory("user_no_memory", "What do you know about me?")
    assert "don't" in result or "information" in result or "yet" in result
    mock_client.chat.completions.create.assert_called_once()


@patch("agentforge.memory.response.client")
def test_answer_with_memory_injects_stored_facts(mock_client, temp_memory_dir):
    # Pre-populate memory (no embedding needed for this test - we mock the LLM)
    user_id = "user_with_facts"
    memory = [
        {"text": "I like coffee", "embedding": [0.1] * 10},
        {"text": "I work in engineering", "embedding": [0.2] * 10},
    ]
    save_memory(user_id, memory)

    mock_client.chat.completions.create.return_value = _make_mock_llm_response(
        "You like coffee and work in engineering."
    )
    result = answer_with_memory(user_id, "What do you know about me?")
    assert result == "You like coffee and work in engineering."

    # Check that the call included the stored memory in the messages
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    messages = call_kwargs.get("messages", [])
    system_contents = [m["content"] for m in messages if m.get("role") == "system"]
    combined = " ".join(system_contents)
    assert "coffee" in combined
    assert "engineering" in combined


# ---------- Conversation history tests ----------


@patch("agentforge.memory.response.client")
def test_history_is_included_in_messages(mock_client, temp_memory_dir):
    """When history is provided, prior turns should appear in the messages list."""
    history = [
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a programming language."},
    ]
    mock_client.chat.completions.create.return_value = _make_mock_llm_response(
        "It was created by Guido van Rossum."
    )

    result = answer_with_memory("u1", "Who created it?", history=history)
    assert "Guido" in result

    call_kwargs = mock_client.chat.completions.create.call_args[1]
    messages = call_kwargs["messages"]
    roles = [m["role"] for m in messages]
    contents = [m["content"] for m in messages]

    assert "user" in roles
    assert "assistant" in roles
    assert any("What is Python?" in c for c in contents)
    assert any("programming language" in c for c in contents)
    assert messages[-1]["content"] == "Who created it?"


@patch("agentforge.memory.response.client")
def test_no_history_is_backward_compatible(mock_client, temp_memory_dir):
    """When history is None (default), messages should contain only system + user."""
    mock_client.chat.completions.create.return_value = _make_mock_llm_response("Hi!")
    answer_with_memory("u1", "Hello")

    call_kwargs = mock_client.chat.completions.create.call_args[1]
    messages = call_kwargs["messages"]
    roles = [m["role"] for m in messages]
    assert roles == ["system", "system", "user"]
