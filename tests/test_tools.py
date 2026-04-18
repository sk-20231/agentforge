"""
Unit tests for agentforge.tools: tool registry, execute_tool, wikipedia_lookup.
"""
import json
import urllib.error
from unittest.mock import MagicMock, patch

from agentforge.tools import TOOL_REGISTRY, execute_tool, get_weather, wikipedia_lookup


class TestToolRegistry:
    """The registry is built from TOOL_MODULES at import time."""

    def test_wikipedia_in_registry(self):
        assert "wikipedia_lookup" in TOOL_REGISTRY

    def test_weather_in_registry(self):
        assert "get_weather" in TOOL_REGISTRY

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


class TestWeather:
    """Tests for the get_weather tool (mocked, no real HTTP calls)."""

    def _mock_response(self, data: dict):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(data).encode("utf-8")
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    def _geocode_ok(self, name="Tokyo", lat=35.69, lon=139.69, country="Japan"):
        return self._mock_response({
            "results": [{
                "name": name, "latitude": lat, "longitude": lon, "country": country
            }]
        })

    def _forecast_ok(self, temp_c=18.3, code=2, wind=12.4):
        return self._mock_response({
            "current": {
                "temperature_2m": temp_c,
                "weather_code": code,
                "wind_speed_10m": wind,
            }
        })

    @patch("agentforge.tools.weather.urllib.request.urlopen")
    def test_successful_lookup_formats_both_units(self, mock_urlopen):
        mock_urlopen.side_effect = [self._geocode_ok(), self._forecast_ok()]
        result = get_weather("Tokyo")
        assert "Tokyo" in result
        assert "Japan" in result
        assert "°C" in result and "°F" in result
        assert "partly cloudy" in result  # weather_code 2
        assert "km/h" in result

    @patch("agentforge.tools.weather.urllib.request.urlopen")
    def test_city_not_found_returns_error(self, mock_urlopen):
        mock_urlopen.return_value = self._mock_response({"results": []})
        result = get_weather("Xyznonexistentville")
        assert "not recognized" in result or "No weather data" in result

    @patch("agentforge.tools.weather.urllib.request.urlopen")
    def test_weather_code_mapped_to_text(self, mock_urlopen):
        mock_urlopen.side_effect = [self._geocode_ok(), self._forecast_ok(code=95)]
        result = get_weather("Tokyo")
        assert "thunderstorm" in result

    @patch("agentforge.tools.weather.urllib.request.urlopen")
    def test_unknown_weather_code_falls_back(self, mock_urlopen):
        mock_urlopen.side_effect = [self._geocode_ok(), self._forecast_ok(code=999)]
        result = get_weather("Tokyo")
        assert "999" in result

    @patch("agentforge.tools.weather.urllib.request.urlopen")
    def test_network_error_returns_message(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.URLError("DNS lookup failed")
        result = get_weather("Tokyo")
        assert "Could not reach weather service" in result

    @patch("agentforge.tools.weather.urllib.request.urlopen")
    def test_http_error_returns_message(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="", code=503, msg="Service Unavailable", hdrs=None, fp=None
        )
        result = get_weather("Tokyo")
        assert "503" in result

    @patch("agentforge.tools.weather.urllib.request.urlopen")
    def test_forecast_missing_fields(self, mock_urlopen):
        mock_urlopen.side_effect = [
            self._geocode_ok(),
            self._mock_response({"current": {}}),
        ]
        result = get_weather("Tokyo")
        assert "incomplete" in result

    def test_empty_city_returns_error(self):
        result = get_weather("")
        assert "Error" in result
