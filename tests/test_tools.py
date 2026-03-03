"""
Unit tests for agent.tools: calculator and execute_tool.
"""
import pytest
from unittest.mock import patch

from agentforge.tools import calculator, execute_tool, TOOL_REGISTRY


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
