"""
Unit tests for agent.tools: calculator, execute_tool, wikipedia_lookup, and leading-zero handling.
"""
import json
import pytest
from unittest.mock import patch, MagicMock
import urllib.error

from agentforge.tools import calculator, execute_tool, wikipedia_lookup, TOOL_REGISTRY, _strip_leading_zeros, safe_eval_math


class TestCalculator:
    """Tests for the calculator tool (no API calls)."""

    def test_simple_expression(self):
        assert calculator("2 + 3") == "5"

    def test_multiplication(self):
        assert calculator("10 * 4") == "40"

    def test_division(self):
        assert calculator("15 / 3") == "5.0"

    def test_parentheses(self):
        assert calculator("(1 + 2) * 3") == "9"

    def test_invalid_expression_returns_error_message(self):
        result = calculator("1 + ")
        assert "Error" in result

    def test_unsafe_expression_rejected(self):
        # eval with restricted builtins should reject things like __import__
        result = calculator("__import__('os').system('echo')")
        assert "Error" in result


class TestStripLeadingZeros:
    """Tests for _strip_leading_zeros — normalises LLM-generated expressions."""

    def test_single_leading_zero(self):
        assert _strip_leading_zeros("0765") == "765"

    def test_multiple_leading_zeros(self):
        assert _strip_leading_zeros("007") == "7"

    def test_zero_itself_preserved(self):
        assert _strip_leading_zeros("0") == "0"

    def test_no_leading_zeros(self):
        assert _strip_leading_zeros("123 * 456") == "123 * 456"

    def test_mixed_expression(self):
        assert _strip_leading_zeros("0765 * 5243") == "765 * 5243"

    def test_both_operands_have_leading_zeros(self):
        assert _strip_leading_zeros("0012 + 0034") == "12 + 34"

    def test_decimal_preserved(self):
        assert _strip_leading_zeros("0.5 * 2") == "0.5 * 2"

    def test_empty_string(self):
        assert _strip_leading_zeros("") == ""


class TestCalculatorLeadingZeros:
    """Tests that calculator handles leading-zero expressions (the actual bug)."""

    def test_leading_zero_multiplication(self):
        assert calculator("0765 * 5243") == str(765 * 5243)

    def test_leading_zero_addition(self):
        assert calculator("007 + 003") == "10"

    def test_leading_zero_large_number(self):
        assert calculator("00100 * 00200") == "20000"


class TestExecuteTool:
    """Tests for execute_tool (mocking log_event to avoid side effects)."""

    @patch("agentforge.tools.log_event")
    def test_execute_calculator(self, mock_log):
        assert execute_tool("calculator", {"expression": "7 * 8"}) == "56"
        mock_log.assert_called()

    @patch("agentforge.tools.log_event")
    def test_unknown_tool_returns_error(self, mock_log):
        result = execute_tool("nonexistent_tool", {})
        assert "Unknown tool" in result
        assert "nonexistent_tool" in result

    @patch("agentforge.tools.log_event")
    def test_calculator_missing_arg_returns_error(self, mock_log):
        result = execute_tool("calculator", {})
        assert "Error" in result or "error" in result.lower()

    @patch("agentforge.tools.log_event")
    @patch("agentforge.tools.urllib.request.urlopen")
    def test_execute_wikipedia_lookup(self, mock_urlopen, mock_log):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "title": "Python",
            "extract": "Python is a programming language."
        }).encode("utf-8")
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = execute_tool("wikipedia_lookup", {"topic": "Python"})
        assert "Python" in result
        assert "programming language" in result


class TestWikipediaLookup:
    """Tests for the wikipedia_lookup tool (mocked, no real HTTP calls)."""

    def _mock_response(self, data: dict):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(data).encode("utf-8")
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    @patch("agentforge.tools.urllib.request.urlopen")
    def test_successful_lookup(self, mock_urlopen):
        mock_urlopen.return_value = self._mock_response({
            "title": "Machine learning",
            "extract": "Machine learning is a branch of AI."
        })
        result = wikipedia_lookup("machine learning")
        assert "Machine learning" in result
        assert "branch of AI" in result

    @patch("agentforge.tools.urllib.request.urlopen")
    def test_topic_not_found_returns_error(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="", code=404, msg="Not Found", hdrs=None, fp=None
        )
        result = wikipedia_lookup("xyznonexistent")
        assert "No Wikipedia article found" in result
        assert "xyznonexistent" in result

    @patch("agentforge.tools.urllib.request.urlopen")
    def test_network_error_returns_message(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.URLError("DNS lookup failed")
        result = wikipedia_lookup("Python")
        assert "Could not reach Wikipedia" in result

    @patch("agentforge.tools.urllib.request.urlopen")
    def test_empty_extract_returns_no_summary(self, mock_urlopen):
        mock_urlopen.return_value = self._mock_response({
            "title": "Test", "extract": ""
        })
        result = wikipedia_lookup("Test")
        assert "No summary available" in result

    def test_empty_topic_returns_error(self):
        result = wikipedia_lookup("")
        assert "Error" in result

    def test_wikipedia_in_tool_registry(self):
        assert "wikipedia_lookup" in TOOL_REGISTRY
