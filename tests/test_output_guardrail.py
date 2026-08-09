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
