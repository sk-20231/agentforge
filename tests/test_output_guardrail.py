"""Tests for the OUTPUT guardrail placement point (issue #22).

The REGEX engine: local PII redaction on the agent's final reply — plus a run_agent
integration test proving a tool reply's PII is redacted before return. These tests pin
the engine to "regex" so they stay deterministic regardless of whether the optional
``[pii]`` (Presidio) extra is installed locally; the Presidio engine has its own tests
in test_output_guardrail_presidio.py.
"""
import json
from unittest.mock import patch

import pytest

from agentforge import guardrail
from agentforge import output_guardrail
from agentforge.output_guardrail import scan_output
from agentforge.main import run_agent


@pytest.fixture(autouse=True)
def _force_regex_engine(monkeypatch):
    """Pin scan_output to the regex engine for this module (the Presidio engine, when
    installed, would otherwise handle these and change the expected redactions)."""
    monkeypatch.setattr(output_guardrail, "AGENT_OUTPUT_GUARDRAIL_ENGINE", "regex")


class TestScanOutput:
    def test_email_redacted(self):
        r = scan_output("Reach me at john.doe@example.com please.")
        assert "john.doe@example.com" not in r.redacted_text
        assert "[REDACTED_EMAIL]" in r.redacted_text
        assert r.found and "EMAIL" in r.types

    def test_openai_key_redacted(self):
        r = scan_output("Your key is sk-abcdef0123456789ABCDEFGHIJ now.")
        assert "sk-abcdef0123456789ABCDEFGHIJ" not in r.redacted_text
        assert "[REDACTED_SECRET]" in r.redacted_text

    def test_aws_key_and_bearer_redacted(self):
        r = scan_output("AKIAIOSFODNN7EXAMPLE and Bearer abcdefghijklmnopqrstuvwxyz123")
        assert "[REDACTED_SECRET]" in r.redacted_text
        assert "AKIAIOSFODNN7EXAMPLE" not in r.redacted_text

    def test_ssn_redacted(self):
        r = scan_output("SSN 123-45-6789 on file.")
        assert "123-45-6789" not in r.redacted_text
        assert "[REDACTED_SSN]" in r.redacted_text

    def test_credit_card_redacted(self):
        r = scan_output("Card: 4111 1111 1111 1111")
        assert "4111 1111 1111 1111" not in r.redacted_text
        assert "[REDACTED_CREDIT_CARD]" in r.redacted_text

    def test_phone_redacted_only_when_aggressive(self):
        text = "Call 555-123-4567 today."
        # Opt-in: not redacted by default (FP-prone on technical answers).
        assert scan_output(text, aggressive=False).redacted_text == text
        r = scan_output(text, aggressive=True)
        assert "555-123-4567" not in r.redacted_text
        assert "[REDACTED_PHONE]" in r.redacted_text

    def test_ipv4_redacted_only_when_aggressive(self):
        text = "Server at 192.168.0.1 is down."
        assert scan_output(text, aggressive=False).redacted_text == text
        r = scan_output(text, aggressive=True)
        assert "192.168.0.1" not in r.redacted_text
        assert "[REDACTED_IP]" in r.redacted_text

    def test_core_pii_redacted_by_default(self):
        # Secrets/email/SSN/card are core — redacted even with aggressive off.
        r = scan_output("Email a@b.com, SSN 123-45-6789.", aggressive=False)
        assert set(r.types) == {"EMAIL", "SSN"}
        assert r.count == 2

    def test_clean_text_unchanged(self):
        text = "The Eiffel Tower is in Paris and was completed in 1889."
        r = scan_output(text)
        assert r.redacted_text == text
        assert not r.found
        assert r.count == 0
        assert r.types == []

    def test_multiple_pii_counted_and_typed(self):
        r = scan_output("Email a@b.com and SSN 123-45-6789.")
        assert r.count == 2
        assert set(r.types) == {"EMAIL", "SSN"}

    def test_empty_text_safe(self):
        assert scan_output("").redacted_text == ""
        assert scan_output("").count == 0


class TestRunAgentOutputGuardrail:
    @patch("agentforge.main.run_llm_with_tools")
    @patch("agentforge.main.classify_intent")
    @patch("agentforge.main.guardrail.scan_external_text")
    def test_act_reply_pii_is_redacted(self, mock_in_scan, mock_classify, mock_tools):
        mock_in_scan.return_value = guardrail.GuardrailResult(guardrail.Verdict.ALLOW)
        mock_classify.return_value = {"intent": "ACT", "memory_candidate": "", "reason": "tool"}
        mock_tools.return_value = json.dumps(
            {"reply": "Sure — email the report to ceo@corp.com.", "store_memory": False}
        )
        out = run_agent("u1", "s1", "who do I email the report to?")
        assert "ceo@corp.com" not in out
        assert "[REDACTED_EMAIL]" in out

    @patch("agentforge.main.run_llm_with_tools")
    @patch("agentforge.main.classify_intent")
    @patch("agentforge.main.guardrail.scan_external_text")
    def test_act_object_reply_is_text_and_redacted(self, mock_in_scan, mock_classify,
                                                   mock_tools):
        # Issue #49: a dict `reply` used to skip _scan_output (non-string) and
        # reach the user unredacted. Normalized to text, the guardrail sees it.
        mock_in_scan.return_value = guardrail.GuardrailResult(guardrail.Verdict.ALLOW)
        mock_classify.return_value = {"intent": "ACT", "memory_candidate": "", "reason": "tool"}
        mock_tools.return_value = json.dumps(
            {"reply": {"contact": "ceo@corp.com", "steps": ["send it"]},
             "store_memory": False}
        )
        out = run_agent("u1", "s1", "who do I email the report to?")
        assert isinstance(out, str)
        assert "ceo@corp.com" not in out
        assert "[REDACTED_EMAIL]" in out
        assert "send it" in out                      # content kept, not dropped

    @patch("agentforge.main.log_event")
    @patch("agentforge.main.resume_tool_loop")
    def test_act_resume_object_reply_is_text(self, mock_resume, mock_log):
        from agentforge.main import resume_agent

        mock_resume.return_value = json.dumps(
            {"reply": {"steps": ["send it"]}, "store_memory": False})
        with patch("agentforge.main.pending_approval",
                   return_value=TestResumeAgentOutputGuardrail._pending("act")):
            out = resume_agent("act-t1", True)
        assert isinstance(out, str)
        assert "send it" in out

    @patch("agentforge.main.AGENT_OUTPUT_GUARDRAIL_ENABLED", False)
    @patch("agentforge.main.run_llm_with_tools")
    @patch("agentforge.main.classify_intent")
    @patch("agentforge.main.guardrail.scan_external_text")
    def test_disabled_passes_through(self, mock_in_scan, mock_classify, mock_tools):
        mock_in_scan.return_value = guardrail.GuardrailResult(guardrail.Verdict.ALLOW)
        mock_classify.return_value = {"intent": "ACT", "memory_candidate": "", "reason": "tool"}
        mock_tools.return_value = json.dumps(
            {"reply": "Email ceo@corp.com.", "store_memory": False}
        )
        out = run_agent("u1", "s1", "x")
        assert "ceo@corp.com" in out  # guardrail off -> not redacted


class TestResumeAgentOutputGuardrail:
    """Issue #51: a turn resumed after an approval is the same turn, so its
    reply gets the same output guardrail as run_agent's. These are the turns
    that called gated tools (e.g. fetch over untrusted web content)."""

    @staticmethod
    def _pending(pipeline):
        from agentforge.approval import ApprovalRequest
        from agentforge.hitl_graph import PendingApproval
        return PendingApproval(
            thread_id=f"{pipeline}-t1", pipeline=pipeline,
            request=ApprovalRequest("fetch", "ext", {"url": "http://x.com"}),
            user_id="u1", trace_id="t1", values={})

    def _resume(self, pipeline):
        from agentforge.main import resume_agent
        with patch("agentforge.main.pending_approval", return_value=self._pending(pipeline)):
            return resume_agent(f"{pipeline}-t1", True)

    @patch("agentforge.main.resume_tool_loop")
    def test_resumed_act_reply_pii_is_redacted(self, mock_resume):
        mock_resume.return_value = json.dumps(
            {"reply": "Email the report to ceo@corp.com.", "store_memory": False})
        out = self._resume("act")
        assert "ceo@corp.com" not in out
        assert "[REDACTED_EMAIL]" in out

    @patch("agentforge.main.resume_react_loop")
    def test_resumed_react_reply_pii_is_redacted(self, mock_resume):
        mock_resume.return_value = "Email the report to ceo@corp.com."
        out = self._resume("react")
        assert "ceo@corp.com" not in out
        assert "[REDACTED_EMAIL]" in out

    @patch("agentforge.main.log_event")
    @patch("agentforge.main.resume_react_loop")
    def test_resumed_redaction_is_logged_on_the_turn_trace(self, mock_resume, mock_log):
        mock_resume.return_value = "SSN 123-45-6789 on file."
        self._resume("react")
        redactions = [c for c in mock_log.call_args_list
                      if c.args[0] == "output_guardrail_redacted"]
        assert len(redactions) == 1
        assert redactions[0].kwargs["trace_id"] == "t1"   # same trace as the turn
        assert "123-45-6789" not in json.dumps(redactions[0].args[1])

    @patch("agentforge.main.AGENT_OUTPUT_GUARDRAIL_ENABLED", False)
    @patch("agentforge.main.resume_react_loop")
    def test_resumed_disabled_passes_through(self, mock_resume):
        mock_resume.return_value = "Email ceo@corp.com."
        assert "ceo@corp.com" in self._resume("react")


# --------------------- structured scanning (Step 21b.1) ---------------------

class TestScanStructured:
    """``scan_structured`` walks a dict/list and scrubs + bounds every string.

    Used to make tool ARGUMENTS safe to write to the trace log. The shape must
    survive the walk — a trajectory reader parses these records back.
    """

    def test_shape_is_preserved_and_clean_args_untouched(self):
        r = output_guardrail.scan_structured({"city": "Paris", "days": 3})
        assert r.value == {"city": "Paris", "days": 3}
        assert r.count == 0
        assert r.types == []
        assert r.truncated is False

    def test_string_value_is_redacted(self):
        r = output_guardrail.scan_structured({"to": "ceo@corp.com"})
        assert "ceo@corp.com" not in json.dumps(r.value)
        assert r.value["to"] == "[REDACTED_EMAIL]"
        assert r.types == ["EMAIL"]
        assert r.count == 1

    def test_keys_are_never_scanned(self):
        """Keys come from the tool's JSON Schema — vocabulary, not user data.
        Scanning them would rename fields and corrupt the record."""
        r = output_guardrail.scan_structured({"ceo@corp.com": "Paris"})
        assert list(r.value.keys()) == ["ceo@corp.com"]
        assert r.count == 0

    def test_nested_dict_and_list_are_walked(self):
        r = output_guardrail.scan_structured(
            {"outer": {"inner": ["ok", "ceo@corp.com"]}})
        assert r.value == {"outer": {"inner": ["ok", "[REDACTED_EMAIL]"]}}
        assert r.count == 1

    def test_long_string_is_capped_and_flagged(self):
        r = output_guardrail.scan_structured({"q": "x" * 50}, max_chars=10)
        assert r.value["q"] == "x" * 10 + "...[truncated]"
        assert r.truncated is True

    def test_zero_max_chars_disables_the_per_string_cap(self):
        r = output_guardrail.scan_structured({"q": "x" * 50}, max_chars=0)
        assert r.value["q"] == "x" * 50
        assert r.truncated is False

    def test_long_list_is_clipped_and_flagged(self):
        r = output_guardrail.scan_structured({"items": list(range(100))},
                                             max_items=3)
        assert r.value["items"] == [0, 1, 2]
        assert r.truncated is True

    def test_deep_nesting_is_depth_capped(self):
        deep = {"a": {"b": {"c": {"d": {"e": "too far"}}}}}
        r = output_guardrail.scan_structured(deep, max_depth=2)
        assert r.truncated is True
        assert "too far" not in json.dumps(r.value)

    def test_booleans_and_none_pass_through_unchanged(self):
        """bool is a subclass of int — it must not be stringified."""
        r = output_guardrail.scan_structured({"flag": True, "off": False,
                                              "nothing": None})
        assert r.value == {"flag": True, "off": False, "nothing": None}

    def test_ordinary_number_is_not_converted_to_string(self):
        r = output_guardrail.scan_structured({"days": 3, "temp": 21.5})
        assert r.value == {"days": 3, "temp": 21.5}

    def test_card_number_sent_as_an_int_is_still_redacted(self):
        """The regex floor only sees strings, so a numeric card would otherwise
        walk straight into the log."""
        r = output_guardrail.scan_structured({"card": 4111111111111111})
        assert r.value["card"] == "[REDACTED_CREDIT_CARD]"
        assert "CREDIT_CARD" in r.types

    def test_bare_string_input_is_supported(self):
        """A model can emit a non-dict tool_input; it is logged as attempted."""
        r = output_guardrail.scan_structured("mail ceo@corp.com")
        assert r.value == "mail [REDACTED_EMAIL]"

    def test_types_are_reported_without_values(self):
        r = output_guardrail.scan_structured(
            {"a": "ceo@corp.com", "b": "sk-abcdef0123456789ABCDEFGHIJ"})
        assert set(r.types) == {"EMAIL", "SECRET"}
        blob = json.dumps({"value": r.value, "types": r.types})
        assert "ceo@corp.com" not in blob
        assert "sk-abcdef0123456789ABCDEFGHIJ" not in blob


class TestScanOutputEngineOverride:
    def test_engine_argument_overrides_the_configured_engine(self, monkeypatch):
        """The per-call override is what keeps scan_structured off the model path."""
        monkeypatch.setattr(output_guardrail, "AGENT_OUTPUT_GUARDRAIL_ENGINE", "presidio")
        # Pretend the engine loaded, so the config alone WOULD take that branch.
        monkeypatch.setattr(output_guardrail, "_get_presidio", lambda: object())
        called = {"presidio": False}

        def _boom(*a, **kw):
            called["presidio"] = True
            raise AssertionError("presidio must not run when engine='regex'")

        monkeypatch.setattr(output_guardrail, "_scan_presidio", _boom)
        r = scan_output("mail ceo@corp.com", engine="regex")
        assert "[REDACTED_EMAIL]" in r.redacted_text
        assert called["presidio"] is False

    def test_engine_none_still_follows_config(self, monkeypatch):
        """Backward compat: existing callers pass no engine and are unaffected."""
        monkeypatch.setattr(output_guardrail, "AGENT_OUTPUT_GUARDRAIL_ENGINE", "presidio")
        monkeypatch.setattr(output_guardrail, "_get_presidio", lambda: object())
        monkeypatch.setattr(
            output_guardrail, "_scan_presidio",
            lambda text, aggressive: output_guardrail.OutputScanResult("PRESIDIO", 1, ["X"]))
        assert scan_output("mail ceo@corp.com").redacted_text == "PRESIDIO"


# --------------------- streamed replies (issue #24) ---------------------

def _chunks(text, size):
    """Split ``text`` into fixed-size tokens, the way a model stream arrives."""
    return [text[i:i + size] for i in range(0, len(text), size)]


def _stream(tokens, **kw):
    return "".join(r.redacted_text for r in output_guardrail.scan_stream(tokens, **kw))


class TestScanStream:
    SAMPLES = [
        "Email john.doe@example.com now. Thanks!",
        "Your key is sk-abcdef0123456789ABCDEFGHIJ.\nKeep it safe? Yes.",
        "SSN 123-45-6789 on file. Card: 4111 1111 1111 1111\nDone.",
        "The Eiffel Tower is in Paris. It was completed in 1889.",
        "Use Bearer\nabcdefghijklmnopqrstuvwxyz123 to call it.",
    ]

    @pytest.mark.parametrize("size", [1, 3, 7, 1000])
    @pytest.mark.parametrize("text", SAMPLES)
    def test_same_result_as_scanning_the_whole_reply(self, text, size):
        # The streamed pieces, joined, must equal one scan of the full reply —
        # for any token size. A cut inside a match would break this.
        assert _stream(_chunks(text, size)) == scan_output(text).redacted_text

    def test_email_split_across_tokens_is_redacted(self):
        out = _stream(_chunks("Email john.doe@example.com now. Thanks.", 3))
        assert "john.doe@example.com" not in out
        assert "[REDACTED_EMAIL]" in out

    def test_every_chunk_is_scanned_before_it_is_yielded(self, monkeypatch):
        seen = []
        real = output_guardrail.scan_output

        def spy(text, **kw):
            seen.append(text)
            return real(text, **kw)

        monkeypatch.setattr(output_guardrail, "scan_output", spy)
        out = list(output_guardrail.scan_stream(_chunks("Mail a.b@c.com. Then stop. End", 2)))
        assert len(out) == len(seen)                      # one scan per yielded chunk
        assert "".join(seen) == "Mail a.b@c.com. Then stop. End"
        assert any("a.b@c.com" in s for s in seen)        # the email was never split

    def test_last_sentence_without_an_end_is_still_scanned(self):
        out = _stream(_chunks("All good. My SSN is 123-45-6789", 4))
        assert "123-45-6789" not in out
        assert out.startswith("All good. ")

    def test_bearer_token_after_a_line_break_is_not_split(self):
        text = "Use Bearer\nabcdefghijklmnopqrstuvwxyz123 to call it."
        out = _stream(_chunks(text, 2))
        assert "abcdefghijklmnopqrstuvwxyz123" not in out

    def test_phone_across_a_line_break_is_not_split_when_aggressive(self):
        text = "Call 555\n123-4567 today."
        full = scan_output(text, aggressive=True).redacted_text
        assert "[REDACTED_PHONE]" in full                 # the pattern spans the \n
        assert _stream(_chunks(text, 2), aggressive=True) == full

    def test_long_text_without_a_sentence_end_is_cut_at_a_space(self):
        words = [f"w{i}x" for i in range(150)]             # ~750 chars, no "." at all
        text = " ".join(words)
        results = list(output_guardrail.scan_stream(_chunks(text, 5)))
        assert len(results) > 1                            # it streamed, not one block
        assert "".join(r.redacted_text for r in results) == text
        for r in results[:-1]:
            assert r.redacted_text[-1].isspace()           # never cut mid-word

    def test_card_at_the_hold_limit_is_not_split(self):
        filler = "x " * 196                                # 392 chars, no sentence end
        text = filler + "card 4111 1111 1111 1111 and more words follow here"
        out = _stream(_chunks(text, 3))
        assert "4111 1111 1111 1111" not in out
        assert out == scan_output(text).redacted_text

    def test_first_sentence_is_yielded_before_the_stream_ends(self):
        consumed = []

        def tokens():
            for t in ["First sentence. ", "Second ", "sentence."]:
                consumed.append(t)
                yield t

        stream = output_guardrail.scan_stream(tokens())
        assert next(stream).redacted_text == "First sentence. "
        assert consumed == ["First sentence. "]            # the rest not read yet

    def test_empty_and_none_tokens_are_skipped(self):
        assert list(output_guardrail.scan_stream([])) == []
        assert _stream(["", None, "Hi. ", "", "there"]) == "Hi. there"


class TestRunAgentStreamedReply:
    """run_agent's streamed ANSWER / DOCS_QA replies get the output guardrail too
    (issue #24): each chunk is redacted before the caller can show it."""

    def _run(self, intent, tokens):
        with patch("agentforge.main.guardrail.scan_external_text",
                   return_value=guardrail.GuardrailResult(guardrail.Verdict.ALLOW)), \
             patch("agentforge.main.classify_intent",
                   return_value={"intent": intent, "memory_candidate": "", "reason": "r"}), \
             patch("agentforge.main.answer_with_memory", return_value=iter(tokens)), \
             patch("agentforge.main.answer_from_docs", return_value=iter(tokens)):
            return run_agent("u1", "s1", "q", stream=True, trace_id="t1")

    @pytest.mark.parametrize("intent", ["ANSWER", "RESPOND_WITH_MEMORY", "DOCS_QA"])
    def test_streamed_reply_pii_is_redacted(self, intent):
        result = self._run(intent, ["Email ", "ceo@", "corp.com", ". Done."])
        assert not isinstance(result, str)                 # still a stream
        out = "".join(result)
        assert "ceo@corp.com" not in out
        assert "[REDACTED_EMAIL]" in out and out.endswith("Done.")

    @patch("agentforge.main.log_event")
    def test_one_event_per_reply_with_totals_and_no_values(self, mock_log):
        out = "".join(self._run("ANSWER", ["Mail a@b.com. ", "SSN 123-45-6789. ", "Bye."]))
        assert "a@b.com" not in out and "123-45-6789" not in out
        events = [c for c in mock_log.call_args_list
                  if c.args[0] == "output_guardrail_redacted"]
        assert len(events) == 1
        payload = events[0].args[1]
        assert payload["count"] == 2 and set(payload["types"]) == {"EMAIL", "SSN"}
        assert payload["streamed"] is True
        assert events[0].kwargs["trace_id"] == "t1"
        assert "a@b.com" not in json.dumps(payload)

    @patch("agentforge.main.log_event")
    def test_event_is_logged_when_the_reader_stops_early(self, mock_log):
        result = self._run("ANSWER", ["Mail a@b.com. ", "More text. ", "End."])
        assert "[REDACTED_EMAIL]" in next(result)
        result.close()                                     # e.g. the UI stopped reading
        assert any(c.args[0] == "output_guardrail_redacted"
                   for c in mock_log.call_args_list)

    @patch("agentforge.main.AGENT_OUTPUT_GUARDRAIL_ENABLED", False)
    def test_disabled_returns_the_stream_unchanged(self):
        out = "".join(self._run("ANSWER", ["Email ", "ceo@corp.com", "."]))
        assert out == "Email ceo@corp.com."
