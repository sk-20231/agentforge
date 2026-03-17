"""
Unit tests for the ReAct reasoning engine.
Focuses on message construction and the loop's handling of LLM responses.
All LLM and memory calls are mocked — no API keys needed.
"""
import json
from unittest.mock import patch, MagicMock

import pytest

from agentforge.prompts import OUTPUT_SCHEMA


class TestReactMessageConstruction:
    """Verify that react_loop builds messages correctly before calling the LLM."""

    @patch("agentforge.reasoning.react_engine.log_event")
    @patch("agentforge.reasoning.react_engine.get_relevant_memories", return_value="no memories")
    @patch("agentforge.reasoning.react_engine.client")
    def test_output_schema_included_in_messages(self, mock_client, mock_mem, mock_log):
        """OUTPUT_SCHEMA must be in messages so response_format=json_object is accepted."""
        final_response = json.dumps({
            "thought": "simple plan",
            "action": {"type": "final"},
            "reply": "Here is your plan.",
        })
        mock_msg = MagicMock()
        mock_msg.content = final_response
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=mock_msg)]
        )

        from agentforge.reasoning.react_engine import react_loop
        react_loop("test_user", "Plan a weekend", max_steps=1)

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        system_contents = [m["content"] for m in messages if m["role"] == "system"]
        assert OUTPUT_SCHEMA in system_contents, (
            "OUTPUT_SCHEMA must be in system messages for response_format=json_object"
        )

    @patch("agentforge.reasoning.react_engine.log_event")
    @patch("agentforge.reasoning.react_engine.get_relevant_memories", return_value="no memories")
    @patch("agentforge.reasoning.react_engine.client")
    def test_json_object_response_format_used(self, mock_client, mock_mem, mock_log):
        """response_format must request json_object for constrained decoding."""
        final_response = json.dumps({
            "thought": "done",
            "action": {"type": "final"},
            "reply": "Done.",
        })
        mock_msg = MagicMock()
        mock_msg.content = final_response
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=mock_msg)]
        )

        from agentforge.reasoning.react_engine import react_loop
        react_loop("test_user", "hello", max_steps=1)

        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["response_format"] == {"type": "json_object"}


class TestReactFinalAction:
    """Verify that the loop returns correctly on a 'final' action."""

    @patch("agentforge.reasoning.react_engine.log_event")
    @patch("agentforge.reasoning.react_engine.get_relevant_memories", return_value="")
    @patch("agentforge.reasoning.react_engine.client")
    def test_final_action_returns_reply(self, mock_client, mock_mem, mock_log):
        final_response = json.dumps({
            "thought": "I can answer directly.",
            "action": {"type": "final"},
            "reply": "Here is your weekend plan!",
        })
        mock_msg = MagicMock()
        mock_msg.content = final_response
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=mock_msg)]
        )

        from agentforge.reasoning.react_engine import react_loop
        result = react_loop("u1", "Plan a weekend", max_steps=3)
        assert result == "Here is your weekend plan!"

    @patch("agentforge.reasoning.react_engine.store_memory")
    @patch("agentforge.reasoning.react_engine.log_event")
    @patch("agentforge.reasoning.react_engine.get_relevant_memories", return_value="")
    @patch("agentforge.reasoning.react_engine.client")
    def test_final_action_stores_memory_when_flagged(self, mock_client, mock_mem, mock_log, mock_store):
        final_response = json.dumps({
            "thought": "User likes hiking.",
            "action": {"type": "final"},
            "reply": "Got it!",
            "store_memory": True,
            "memory_text": "User enjoys hiking on weekends.",
        })
        mock_msg = MagicMock()
        mock_msg.content = final_response
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=mock_msg)]
        )

        from agentforge.reasoning.react_engine import react_loop
        react_loop("u1", "I love hiking", max_steps=1)
        mock_store.assert_called_once_with("u1", "User enjoys hiking on weekends.")


class TestReactErrorHandling:
    """Verify graceful error handling when the LLM call fails."""

    @patch("agentforge.reasoning.react_engine.log_event")
    @patch("agentforge.reasoning.react_engine.get_relevant_memories", return_value="")
    @patch("agentforge.reasoning.react_engine.client")
    def test_llm_exception_returns_friendly_error(self, mock_client, mock_mem, mock_log):
        mock_client.chat.completions.create.side_effect = Exception("API error")

        from agentforge.reasoning.react_engine import react_loop
        result = react_loop("u1", "Plan something", max_steps=1)
        assert "error" in result.lower()

    @patch("agentforge.reasoning.react_engine.log_event")
    @patch("agentforge.reasoning.react_engine.get_relevant_memories", return_value="")
    @patch("agentforge.reasoning.react_engine.client")
    def test_invalid_json_returns_error(self, mock_client, mock_mem, mock_log):
        mock_msg = MagicMock()
        mock_msg.content = "not valid json at all"
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=mock_msg)]
        )

        from agentforge.reasoning.react_engine import react_loop
        result = react_loop("u1", "Do something", max_steps=1)
        assert "error" in result.lower() or "Invalid" in result

    @patch("agentforge.reasoning.react_engine.log_event")
    @patch("agentforge.reasoning.react_engine.get_relevant_memories", return_value="")
    @patch("agentforge.reasoning.react_engine.client")
    def test_max_steps_exceeded(self, mock_client, mock_mem, mock_log):
        """If the model never returns action.type='final', loop should stop at max_steps."""
        tool_response = json.dumps({
            "thought": "need more info",
            "action": {"type": "tool", "tool_name": "calculator", "tool_input": {"expression": "1+1"}},
            "reply": "",
        })
        mock_msg = MagicMock()
        mock_msg.content = tool_response
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=mock_msg)]
        )

        with patch("agentforge.reasoning.react_engine.execute_tool", return_value="2"):
            from agentforge.reasoning.react_engine import react_loop
            result = react_loop("u1", "keep going", max_steps=2)
        assert "too many" in result.lower()
