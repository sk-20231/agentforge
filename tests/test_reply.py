"""Tests for reply normalization (issue #49).

``response_format={"type": "json_object"}`` guarantees valid JSON, not field
types, so the model's ``reply`` can arrive as an object or list. Every
pipeline reads it through ``agentforge.reply.reply_text``, which always
returns a string. The run_agent-level check (the output guardrail now sees
and redacts an object reply) lives in test_output_guardrail.py.
"""
import contextlib
import json
from unittest.mock import MagicMock, patch

from agentforge.reply import reply_text


class TestReplyText:
    def test_string_passes_through_unlogged(self):
        with patch("agentforge.reply.log_event") as log:
            assert reply_text("hello", source="react") == "hello"
        log.assert_not_called()

    def test_empty_string_is_kept(self):
        assert reply_text("", source="react") == ""

    def test_none_becomes_empty_string(self):
        # A null reply used to reach len(None) in the react_end log and crash.
        with patch("agentforge.reply.log_event") as log:
            assert reply_text(None, source="react") == ""
        log.assert_not_called()

    def test_dict_becomes_readable_json(self):
        value = {"comparison": {"python": "3.14"}, "plan": ["upgrade"]}
        with patch("agentforge.reply.log_event"):
            out = reply_text(value, source="react")
        assert isinstance(out, str)
        assert json.loads(out) == value          # content kept exactly
        assert "\n" in out                       # indented, not one long line

    def test_list_becomes_json(self):
        with patch("agentforge.reply.log_event"):
            out = reply_text(["a", "b"], source="act")
        assert json.loads(out) == ["a", "b"]

    def test_non_ascii_is_not_escaped(self):
        with patch("agentforge.reply.log_event"):
            out = reply_text({"city": "Zürich"}, source="act")
        assert "Zürich" in out

    def test_other_types_become_str(self):
        with patch("agentforge.reply.log_event"):
            assert reply_text(42, source="act") == "42"

    def test_logs_type_and_source_never_the_value(self):
        with patch("agentforge.reply.log_event") as log:
            reply_text({"email": "ceo@corp.com"}, source="react", trace_id="t1")
        event, payload = log.call_args.args
        assert event == "reply_not_string"
        assert payload == {"source": "react", "type": "dict"}
        assert log.call_args.kwargs["trace_id"] == "t1"
        assert "ceo@corp.com" not in json.dumps(payload)


class _FakeGateway:
    catalog = []
    has_tools = False
    granted = set()
    nonce = "n"


@contextlib.asynccontextmanager
async def _fake_gateway(trace_id=None, approval_handler=None):
    yield _FakeGateway()


class TestReactObjectReply:
    def test_final_object_reply_is_returned_as_text(self):
        from agentforge.reasoning import react_engine

        final = json.dumps({"thought": "done", "action": {"type": "final"},
                            "reply": {"comparison": {}, "upgrade_plan": ["step 1"]},
                            "store_memory": False, "memory_text": ""})
        msg = MagicMock()
        msg.content = final
        with patch.object(react_engine, "mcp_gateway", _fake_gateway), \
             patch.object(react_engine, "get_relevant_memories", return_value=""), \
             patch.object(react_engine, "log_event"), \
             patch.object(react_engine, "log_token_usage"), \
             patch("agentforge.reply.log_event") as reply_log, \
             patch.object(react_engine, "_get_client") as client:
            client.return_value.chat.completions.create.return_value = MagicMock(
                choices=[MagicMock(message=msg)])
            out = react_engine.react_loop("u1", "plan my upgrade")

        assert isinstance(out, str)
        assert "step 1" in out
        assert reply_log.call_args.args[1]["source"] == "react"

    def test_null_reply_does_not_crash(self):
        from agentforge.reasoning import react_engine

        final = json.dumps({"thought": "done", "action": {"type": "final"},
                            "reply": None})
        msg = MagicMock()
        msg.content = final
        with patch.object(react_engine, "mcp_gateway", _fake_gateway), \
             patch.object(react_engine, "get_relevant_memories", return_value=""), \
             patch.object(react_engine, "log_event"), \
             patch.object(react_engine, "log_token_usage"), \
             patch.object(react_engine, "_get_client") as client:
            client.return_value.chat.completions.create.return_value = MagicMock(
                choices=[MagicMock(message=msg)])
            assert react_engine.react_loop("u1", "x") == ""
