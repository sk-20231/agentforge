"""
Unit tests for agentforge.tools: tool registry, execute_tool, wikipedia_lookup.
"""
import json
import urllib.error
from unittest.mock import MagicMock, patch

from agentforge.tools import TOOL_REGISTRY, execute_tool, wikipedia_lookup


class TestToolRegistry:
    """The registry is built from TOOL_MODULES at import time."""

    def test_wikipedia_in_registry(self):
        assert "wikipedia_lookup" in TOOL_REGISTRY

    def test_registry_entries_are_callable(self):
        for name, func in TOOL_REGISTRY.items():
            assert callable(func), f"{name} is not callable"


class TestExecuteTool:
    """Tests for execute_tool (mocking log_event to avoid side effects)."""

    @patch("agentforge.tools.log_event")
    def test_unknown_tool_returns_error(self, mock_log):
        result = execute_tool("nonexistent_tool", {})
        assert "Unknown tool" in result
        assert "nonexistent_tool" in result

    @patch("agentforge.tools.log_event")
    @patch("agentforge.tools.wikipedia.urllib.request.urlopen")
    def test_execute_wikipedia_lookup(self, mock_urlopen, mock_log):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "title": "Python",
            "extract": "Python is a programming language.",
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

    @patch("agentforge.tools.wikipedia.urllib.request.urlopen")
    def test_successful_lookup(self, mock_urlopen):
        mock_urlopen.return_value = self._mock_response({
            "title": "Machine learning",
            "extract": "Machine learning is a branch of AI.",
        })
        result = wikipedia_lookup("machine learning")
        assert "Machine learning" in result
        assert "branch of AI" in result

    @patch("agentforge.tools.wikipedia.urllib.request.urlopen")
    def test_topic_not_found_returns_error(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="", code=404, msg="Not Found", hdrs=None, fp=None
        )
        result = wikipedia_lookup("xyznonexistent")
        assert "No Wikipedia article found" in result
        assert "xyznonexistent" in result

    @patch("agentforge.tools.wikipedia.urllib.request.urlopen")
    def test_network_error_returns_message(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.URLError("DNS lookup failed")
        result = wikipedia_lookup("Python")
        assert "Could not reach Wikipedia" in result

    @patch("agentforge.tools.wikipedia.urllib.request.urlopen")
    def test_empty_extract_returns_no_summary(self, mock_urlopen):
        mock_urlopen.return_value = self._mock_response({
            "title": "Test", "extract": ""
        })
        result = wikipedia_lookup("Test")
        assert "No summary available" in result

    def test_empty_topic_returns_error(self):
        result = wikipedia_lookup("")
        assert "Error" in result
