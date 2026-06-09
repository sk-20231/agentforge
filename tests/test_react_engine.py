"""
Unit tests for the ReAct reasoning engine.
Focuses on message construction and the loop's handling of LLM responses.
All LLM and memory calls are mocked — no API keys needed.

Step 17c.2: ReAct dispatches tools over MCP. These tests fake the MCP gateway
(``agentforge.reasoning.react_engine.mcp_gateway``) so no real servers spawn; the
real-subprocess contract tests live in tests/test_*_mcp_server.py.
"""
import contextlib
import json
from unittest.mock import patch, MagicMock

import pytest

from agentforge.prompts import OUTPUT_SCHEMA


# --------------------------- fake MCP gateway ---------------------------

class _FakeGateway:
    """Stand-in for agentforge.mcp_client.MCPGateway with no subprocesses."""

    def __init__(self, catalog=None, call_result="<untrusted_data>obs</untrusted_data>"):
        self.catalog = catalog or []
        self._call_result = call_result
        self.calls = []  # records (tool_name, tool_input) for assertions

    @property
    def has_tools(self):
        return bool(self.catalog)

    async def call(self, tool_name, arguments):
        self.calls.append((tool_name, arguments))
        return self._call_result


def _fake_gateway_cm(gw):
    """Return a drop-in replacement for the mcp_gateway async context manager."""
    @contextlib.asynccontextmanager
    async def _cm(trace_id=None):
        yield gw
    return _cm


def _patch_gateway(gw):
    return patch("agentforge.reasoning.react_engine.mcp_gateway", _fake_gateway_cm(gw))


class TestReactMessageConstruction:
    """Verify that react_loop builds messages correctly before calling the LLM."""

    @patch("agentforge.reasoning.react_engine.log_event")
    @patch("agentforge.reasoning.react_engine.get_relevant_memories", return_value="no memories")
    @patch("agentforge.reasoning.react_engine._client")
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
        with _patch_gateway(_FakeGateway()):
            react_loop("test_user", "Plan a weekend", max_steps=1)

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        system_contents = [m["content"] for m in messages if m["role"] == "system"]
        assert OUTPUT_SCHEMA in system_contents, (
            "OUTPUT_SCHEMA must be in system messages for response_format=json_object"
        )

    @patch("agentforge.reasoning.react_engine.log_event")
    @patch("agentforge.reasoning.react_engine.get_relevant_memories", return_value="no memories")
    @patch("agentforge.reasoning.react_engine._client")
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
        with _patch_gateway(_FakeGateway()):
            react_loop("test_user", "hello", max_steps=1)

        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["response_format"] == {"type": "json_object"}

    @patch("agentforge.reasoning.react_engine.log_event")
    @patch("agentforge.reasoning.react_engine.get_relevant_memories", return_value="no memories")
    @patch("agentforge.reasoning.react_engine._client")
    def test_discovered_tool_catalog_is_rendered_in_prompt(self, mock_client, mock_mem, mock_log):
        """The live tool catalog (not a hardcoded name) must be injected so the
        model only references tools that are actually served. Regression guard for
        the stale 'calculator' the prompt used to advertise (Step 17c.2)."""
        final_response = json.dumps({
            "thought": "done", "action": {"type": "final"}, "reply": "ok",
        })
        mock_msg = MagicMock()
        mock_msg.content = final_response
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=mock_msg)]
        )

        gw = _FakeGateway(catalog=[
            {"name": "search_wikipedia", "description": "Look up a topic",
             "input_schema": {"properties": {"topic": {"type": "string"}}}},
        ])
        from agentforge.reasoning.react_engine import react_loop
        with _patch_gateway(gw):
            react_loop("u1", "who is Ada Lovelace", max_steps=1)

        messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
        joined = "\n".join(m["content"] for m in messages if m["role"] == "system")
        assert "search_wikipedia" in joined
        assert "calculator" not in joined


class TestReactFinalAction:
    """Verify that the loop returns correctly on a 'final' action."""

    @patch("agentforge.reasoning.react_engine.log_event")
    @patch("agentforge.reasoning.react_engine.get_relevant_memories", return_value="")
    @patch("agentforge.reasoning.react_engine._client")
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
        with _patch_gateway(_FakeGateway()):
            result = react_loop("u1", "Plan a weekend", max_steps=3)
        assert result == "Here is your weekend plan!"

    @patch("agentforge.reasoning.react_engine.store_memory")
    @patch("agentforge.reasoning.react_engine.log_event")
    @patch("agentforge.reasoning.react_engine.get_relevant_memories", return_value="")
    @patch("agentforge.reasoning.react_engine._client")
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
        with _patch_gateway(_FakeGateway()):
            react_loop("u1", "I love hiking", max_steps=1)
        mock_store.assert_called_once_with("u1", "User enjoys hiking on weekends.")


class TestReactToolDispatch:
    """Verify tool steps are dispatched over MCP (Step 17c.2)."""

    @patch("agentforge.reasoning.react_engine.log_event")
    @patch("agentforge.reasoning.react_engine.get_relevant_memories", return_value="")
    @patch("agentforge.reasoning.react_engine._client")
    def test_tool_action_dispatches_through_gateway(self, mock_client, mock_mem, mock_log):
        """A 'tool' action must call gw.call (MCP), not any local registry, then
        the observation feeds the next step which finalizes."""
        tool_step = json.dumps({
            "thought": "look it up",
            "action": {"type": "tool", "tool_name": "search_wikipedia", "tool_input": {"topic": "Ada Lovelace"}},
            "reply": "",
        })
        final_step = json.dumps({
            "thought": "now I can answer",
            "action": {"type": "final"},
            "reply": "Ada Lovelace was a mathematician.",
        })
        msg1, msg2 = MagicMock(), MagicMock()
        msg1.content, msg2.content = tool_step, final_step
        mock_client.chat.completions.create.side_effect = [
            MagicMock(choices=[MagicMock(message=msg1)]),
            MagicMock(choices=[MagicMock(message=msg2)]),
        ]

        gw = _FakeGateway(
            catalog=[{"name": "search_wikipedia", "description": "Look up a topic",
                      "input_schema": {"properties": {"topic": {"type": "string"}}}}],
            call_result="<untrusted_data>Ada Lovelace: a mathematician</untrusted_data>",
        )
        from agentforge.reasoning.react_engine import react_loop
        with _patch_gateway(gw):
            result = react_loop("u1", "who is Ada Lovelace", max_steps=3)

        assert gw.calls == [("search_wikipedia", {"topic": "Ada Lovelace"})]
        assert result == "Ada Lovelace was a mathematician."

    @patch("agentforge.reasoning.react_engine.log_event")
    @patch("agentforge.reasoning.react_engine.get_relevant_memories", return_value="")
    @patch("agentforge.reasoning.react_engine._client")
    def test_observation_fed_back_as_user_not_orphan_tool_role(self, mock_client, mock_mem, mock_log):
        """Regression for issue #1: after a tool step the observation must go back
        as a USER message, never a role:"tool" message.

        A "tool" message is only legal in OpenAI's native function-calling protocol
        (immediately after an assistant message carrying tool_calls). This loop uses
        the JSON-prompt protocol, so an orphan "tool" message makes the *next*
        request fail with a 400. The mocked client can't surface that 400 (the stub
        accepts any messages), which is exactly why the original bug slipped through
        — so this test asserts the message *structure* sent to the API directly."""
        tool_step = json.dumps({
            "thought": "look it up",
            "action": {"type": "tool", "tool_name": "search_wikipedia", "tool_input": {"topic": "Ada Lovelace"}},
            "reply": "",
        })
        final_step = json.dumps({
            "thought": "now I can answer", "action": {"type": "final"},
            "reply": "Ada Lovelace was a mathematician.",
        })
        msg1, msg2 = MagicMock(), MagicMock()
        msg1.content, msg2.content = tool_step, final_step
        mock_client.chat.completions.create.side_effect = [
            MagicMock(choices=[MagicMock(message=msg1)]),
            MagicMock(choices=[MagicMock(message=msg2)]),
        ]

        gw = _FakeGateway(
            catalog=[{"name": "search_wikipedia", "description": "Look up a topic",
                      "input_schema": {"properties": {"topic": {"type": "string"}}}}],
            call_result="<untrusted_data_abc123>Ada Lovelace: a mathematician</untrusted_data_abc123>",
        )
        from agentforge.reasoning.react_engine import react_loop
        with _patch_gateway(gw):
            react_loop("u1", "who is Ada Lovelace", max_steps=3)

        # The 2nd LLM call carries the post-tool message history. Under the
        # JSON-prompt protocol none of those messages may use the "tool" role.
        assert mock_client.chat.completions.create.call_count == 2
        second_call_messages = mock_client.chat.completions.create.call_args_list[1].kwargs["messages"]
        orphan_tool_msgs = [m for m in second_call_messages if m.get("role") == "tool"]
        assert orphan_tool_msgs == [], (
            "ReAct must feed observations back as a user message, not role:'tool' "
            f"(OpenAI rejects an orphan 'tool' role). Found: {orphan_tool_msgs}"
        )
        # The observation must still be present — fed back as a user turn.
        user_obs = [m for m in second_call_messages
                    if m.get("role") == "user" and "Observation from tool" in m.get("content", "")]
        assert user_obs, "the tool observation should be fed back as a user message"

    @patch("agentforge.reasoning.react_engine.log_event")
    @patch("agentforge.reasoning.react_engine.get_relevant_memories", return_value="")
    @patch("agentforge.reasoning.react_engine._client")
    def test_unknown_tool_observation_is_recoverable(self, mock_client, mock_mem, mock_log):
        """An unknown tool returns a readable error from the gateway (not a raise),
        which the loop feeds back as an observation."""
        tool_step = json.dumps({
            "thought": "try a tool",
            "action": {"type": "tool", "tool_name": "calculator", "tool_input": {"x": 1}},
            "reply": "",
        })
        final_step = json.dumps({
            "thought": "give up on the tool", "action": {"type": "final"}, "reply": "done",
        })
        msg1, msg2 = MagicMock(), MagicMock()
        msg1.content, msg2.content = tool_step, final_step
        mock_client.chat.completions.create.side_effect = [
            MagicMock(choices=[MagicMock(message=msg1)]),
            MagicMock(choices=[MagicMock(message=msg2)]),
        ]

        # Gateway has the tool catalog but call() returns an unknown-tool error.
        gw = _FakeGateway(call_result="Error: Unknown tool 'calculator'")
        from agentforge.reasoning.react_engine import react_loop
        with _patch_gateway(gw):
            result = react_loop("u1", "compute", max_steps=2)

        assert gw.calls == [("calculator", {"x": 1})]
        assert result == "done"


class TestReactErrorHandling:
    """Verify graceful error handling when the LLM call fails."""

    @patch("agentforge.reasoning.react_engine.log_event")
    @patch("agentforge.reasoning.react_engine.get_relevant_memories", return_value="")
    @patch("agentforge.reasoning.react_engine._client")
    def test_llm_exception_returns_friendly_error(self, mock_client, mock_mem, mock_log):
        mock_client.chat.completions.create.side_effect = Exception("API error")

        from agentforge.reasoning.react_engine import react_loop
        with _patch_gateway(_FakeGateway()):
            result = react_loop("u1", "Plan something", max_steps=1)
        assert "error" in result.lower()

    @patch("agentforge.reasoning.react_engine.log_event")
    @patch("agentforge.reasoning.react_engine.get_relevant_memories", return_value="")
    @patch("agentforge.reasoning.react_engine._client")
    def test_invalid_json_returns_error(self, mock_client, mock_mem, mock_log):
        mock_msg = MagicMock()
        mock_msg.content = "not valid json at all"
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=mock_msg)]
        )

        from agentforge.reasoning.react_engine import react_loop
        with _patch_gateway(_FakeGateway()):
            result = react_loop("u1", "Do something", max_steps=1)
        assert "error" in result.lower() or "Invalid" in result

    @patch("agentforge.reasoning.react_engine.log_event")
    @patch("agentforge.reasoning.react_engine.get_relevant_memories", return_value="")
    @patch("agentforge.reasoning.react_engine._client")
    def test_max_steps_exceeded(self, mock_client, mock_mem, mock_log):
        """If the model never returns action.type='final', loop stops at max_steps."""
        tool_response = json.dumps({
            "thought": "need more info",
            "action": {"type": "tool", "tool_name": "search_wikipedia", "tool_input": {"topic": "x"}},
            "reply": "",
        })
        mock_msg = MagicMock()
        mock_msg.content = tool_response
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=mock_msg)]
        )

        gw = _FakeGateway(
            catalog=[{"name": "search_wikipedia", "description": "Look up a topic",
                      "input_schema": {"properties": {"topic": {"type": "string"}}}}],
            call_result="some observation",
        )
        from agentforge.reasoning.react_engine import react_loop
        with _patch_gateway(gw):
            result = react_loop("u1", "keep going", max_steps=2)
        assert "too many" in result.lower()
