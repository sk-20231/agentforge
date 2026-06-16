"""Tests for the OUTPUT guardrail placement point (issue #22).

Pure, local regex PII redaction on the agent's final reply — plus a run_agent
integration test proving a tool reply's PII is redacted before return.
"""
import json
from unittest.mock import patch

from agentforge import guardrail
from agentforge.output_guardrail import scan_output
from agentforge.main import run_agent


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
