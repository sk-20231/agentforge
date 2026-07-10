"""Tests for the OUTPUT guardrail's engine dispatch + the Presidio/GLiNER engine (#24).

Two layers:
  - HERMETIC dispatcher tests (no model load): prove ``scan_output`` picks the right
    engine and ALWAYS degrades to the regex floor — never to "no guard" — when Presidio
    is absent or a scan errors. These run everywhere, including CI without ``[pii]``.
  - An INTEGRATION test that loads the real Presidio + GLiNER engine and confirms it
    redacts a person name regex can't see. It SKIPS when the ``[pii]`` extra/model are
    unavailable (same pattern as the live MCP contract tests).
"""
from unittest.mock import MagicMock

import pytest

from agentforge import output_guardrail
from agentforge.output_guardrail import OutputScanResult, scan_output


class TestEngineDispatch:
    """scan_output must route correctly and fail SAFE (to regex), never open."""

    def test_regex_engine_never_touches_presidio(self, monkeypatch):
        monkeypatch.setattr(output_guardrail, "AGENT_OUTPUT_GUARDRAIL_ENGINE", "regex")

        def _boom():
            raise AssertionError("_get_presidio must not be called when engine=regex")

        monkeypatch.setattr(output_guardrail, "_get_presidio", _boom)
        r = scan_output("Email John Smith at a@b.com.")
        # Regex catches the email but CANNOT see the name.
        assert "[REDACTED_EMAIL]" in r.redacted_text
        assert "John Smith" in r.redacted_text

    def test_auto_falls_back_to_regex_when_presidio_unavailable(self, monkeypatch):
        monkeypatch.setattr(output_guardrail, "AGENT_OUTPUT_GUARDRAIL_ENGINE", "auto")
        monkeypatch.setattr(output_guardrail, "_get_presidio", lambda: None)
        r = scan_output("Email John Smith at a@b.com.")
        assert "[REDACTED_EMAIL]" in r.redacted_text
        assert "John Smith" in r.redacted_text  # regex floor, name not caught

    def test_auto_uses_presidio_when_available(self, monkeypatch):
        monkeypatch.setattr(output_guardrail, "AGENT_OUTPUT_GUARDRAIL_ENGINE", "auto")
        monkeypatch.setattr(output_guardrail, "_get_presidio", lambda: MagicMock())
        sentinel = OutputScanResult("[REDACTED_PERSON] was here", 1, ["PERSON"])
        monkeypatch.setattr(output_guardrail, "_scan_presidio",
                            lambda text, aggressive: sentinel)
        assert scan_output("Jane Doe was here") is sentinel

    def test_presidio_scan_error_degrades_to_regex(self, monkeypatch):
        monkeypatch.setattr(output_guardrail, "AGENT_OUTPUT_GUARDRAIL_ENGINE", "auto")
        monkeypatch.setattr(output_guardrail, "_get_presidio", lambda: MagicMock())

        def _raise(text, aggressive):
            raise RuntimeError("model exploded mid-scan")

        monkeypatch.setattr(output_guardrail, "_scan_presidio", _raise)
        # Must not raise; must fall back to the regex engine (email still redacted).
        r = scan_output("Reach a@b.com now.")
        assert "[REDACTED_EMAIL]" in r.redacted_text

    def test_empty_text_short_circuits_before_engine(self, monkeypatch):
        def _boom():
            raise AssertionError("must not select an engine for empty text")

        monkeypatch.setattr(output_guardrail, "_get_presidio", _boom)
        assert scan_output("").redacted_text == ""
        assert scan_output("").count == 0


class TestPresidioEngineIntegration:
    """Loads the real Presidio + GLiNER engine. Skips without the [pii] extra/model."""

    @pytest.fixture()
    def presidio_engine(self, monkeypatch):
        pytest.importorskip("presidio_analyzer", reason="[pii] extra not installed")
        pytest.importorskip("gliner", reason="[pii] extra not installed")
        monkeypatch.setattr(output_guardrail, "AGENT_OUTPUT_GUARDRAIL_ENGINE", "presidio")
        output_guardrail.reset_engine_cache()
        if output_guardrail._get_presidio() is None:
            pytest.skip("Presidio engine unavailable (deps or model not downloaded)")
        yield
        output_guardrail.reset_engine_cache()

    def test_redacts_person_name_regex_cannot_see(self, presidio_engine):
        r = scan_output("Please email John Smith about the quarterly report.")
        assert "John Smith" not in r.redacted_text
        assert "[REDACTED_PERSON]" in r.redacted_text
        assert "PERSON" in r.types

    def test_card_luhn_avoids_regex_false_positive(self, presidio_engine):
        # A 16-digit run that FAILS Luhn — the regex engine flags it, Presidio must not.
        r = scan_output("Order number 1234567890123456 has shipped.")
        assert "1234567890123456" in r.redacted_text  # not redacted
        assert "CREDIT_CARD" not in r.types
