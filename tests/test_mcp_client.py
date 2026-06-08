"""Unit tests for the shared MCP gateway (agentforge.mcp_client).

The gateway is the one place both ACT and ReAct use to discover tools and
dispatch calls. These tests mock the MCP transport (stdio_client / ClientSession)
so they are fast and hermetic — the real-subprocess contract tests for the actual
servers live in tests/test_*_mcp_server.py.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from agentforge.mcp_client import MCPGateway, mcp_gateway, mcp_tool_to_openai_schema


# --------------------------- mock helpers ---------------------------

def _make_mcp_tool(name, description="A tool", input_schema=None):
    tool = MagicMock()
    tool.name = name
    tool.description = description
    tool.inputSchema = input_schema or {
        "type": "object",
        "properties": {"topic": {"type": "string"}},
        "required": ["topic"],
    }
    return tool


def _make_session_mock(tools, call_text="<untrusted_data>result</untrusted_data>", is_error=False):
    session = AsyncMock()
    session.initialize = AsyncMock()

    list_result = MagicMock()
    list_result.tools = tools
    session.list_tools = AsyncMock(return_value=list_result)

    content = MagicMock()
    content.text = call_text
    call_result = MagicMock()
    call_result.isError = is_error
    call_result.content = [content]
    session.call_tool = AsyncMock(return_value=call_result)

    return session


def _patch_mcp_transport(session_mock):
    stdio_cm = MagicMock()
    stdio_cm.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
    stdio_cm.__aexit__ = AsyncMock(return_value=None)
    mock_stdio = MagicMock(return_value=stdio_cm)

    session_cm = MagicMock()
    session_cm.__aenter__ = AsyncMock(return_value=session_mock)
    session_cm.__aexit__ = AsyncMock(return_value=None)
    mock_cs = MagicMock(return_value=session_cm)

    return mock_stdio, mock_cs


# One-server config in the standard mcpServers dict shape (Step 17d). trusted=True
# so these routing/dispatch tests don't trigger client-side wrapping; the wrap
# behaviour is covered separately with a trusted=False server.
_DUMMY_SERVERS = {"dummy": {"command": "python", "args": ["dummy/server.py"], "trusted": True}}

# A third-party (untrusted) server: the gateway must guard its URL args and wrap
# its output. Tests use numeric-IP URLs so is_safe_url needs no real DNS lookup.
_UNTRUSTED_SERVERS = {"ext": {"command": "uvx", "args": ["some-tool"], "trusted": False}}


def _run_with_gateway(session_mock, servers, body):
    """Open a gateway over a mocked transport, run ``body(gw)``, return its result."""
    mock_stdio, mock_cs = _patch_mcp_transport(session_mock)

    async def _go():
        async with mcp_gateway() as gw:
            return await body(gw)

    with patch("agentforge.mcp_client.MCP_SERVERS", servers), \
         patch("agentforge.mcp_client.log_event"), \
         patch("agentforge.mcp_client.stdio_client", mock_stdio), \
         patch("agentforge.mcp_client.ClientSession", mock_cs):
        return asyncio.run(_go())


# --------------------------- schema conversion ---------------------------

class TestSchemaConversion:
    def test_shape_is_correct(self):
        schema = mcp_tool_to_openai_schema(_make_mcp_tool("search_wikipedia", "Search Wikipedia"))
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "search_wikipedia"
        assert schema["function"]["description"] == "Search Wikipedia"
        assert schema["function"]["parameters"]["properties"]["topic"]["type"] == "string"

    def test_missing_description_becomes_empty_string(self):
        schema = mcp_tool_to_openai_schema(_make_mcp_tool("no_desc", description=None))
        assert schema["function"]["description"] == ""


# --------------------------- discovery ---------------------------

class TestDiscovery:
    def test_discovery_populates_catalog_and_schemas(self):
        session = _make_session_mock([
            _make_mcp_tool("search_wikipedia", "Search Wikipedia"),
            _make_mcp_tool("get_weather", "Weather by city", {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            }),
        ])

        async def body(gw):
            return gw

        gw = _run_with_gateway(session, _DUMMY_SERVERS, body)

        assert gw.has_tools is True
        names = [t["name"] for t in gw.catalog]
        assert names == ["search_wikipedia", "get_weather"]
        # catalog carries the input schema so the ReAct prompt can show arg shapes
        weather = next(t for t in gw.catalog if t["name"] == "get_weather")
        assert weather["input_schema"]["properties"]["city"]["type"] == "string"
        # openai_schemas are in function-calling shape for the ACT loop
        assert {s["function"]["name"] for s in gw.openai_schemas} == {"search_wikipedia", "get_weather"}

    def test_discovery_failure_yields_no_tools_no_crash(self):
        """A server that fails to start must not crash the turn — has_tools=False."""
        stdio_cm = MagicMock()
        stdio_cm.__aenter__ = AsyncMock(side_effect=OSError("No such file"))
        stdio_cm.__aexit__ = AsyncMock(return_value=None)
        mock_stdio = MagicMock(return_value=stdio_cm)

        async def _go():
            async with mcp_gateway() as gw:
                return gw.has_tools, gw.catalog

        with patch("agentforge.mcp_client.MCP_SERVERS",
                   {"bad": {"command": "python", "args": ["bad/server.py"], "trusted": True}}), \
             patch("agentforge.mcp_client.log_event"), \
             patch("agentforge.mcp_client.stdio_client", mock_stdio):
            has_tools, catalog = asyncio.run(_go())

        assert has_tools is False
        assert catalog == []


# --------------------------- dispatch ---------------------------

class TestCall:
    def test_call_dispatches_through_session(self):
        session = _make_session_mock([_make_mcp_tool("search_wikipedia")],
                                     call_text="<untrusted_data>Python is a language</untrusted_data>")

        async def body(gw):
            return await gw.call("search_wikipedia", {"topic": "Python"})

        result = _run_with_gateway(session, _DUMMY_SERVERS, body)
        session.call_tool.assert_called_once_with("search_wikipedia", {"topic": "Python"})
        assert "Python is a language" in result

    def test_unknown_tool_returns_error_string_and_does_not_dispatch(self):
        session = _make_session_mock([_make_mcp_tool("search_wikipedia")])

        async def body(gw):
            return await gw.call("get_weather", {"city": "Tokyo"})  # never discovered

        result = _run_with_gateway(session, _DUMMY_SERVERS, body)
        assert "Unknown tool 'get_weather'" in result
        session.call_tool.assert_not_called()

    def test_tool_error_result_is_prefixed(self):
        session = _make_session_mock([_make_mcp_tool("search_wikipedia")],
                                     call_text="boom", is_error=True)

        async def body(gw):
            return await gw.call("search_wikipedia", {"topic": "x"})

        result = _run_with_gateway(session, _DUMMY_SERVERS, body)
        assert result.startswith("Tool error:")
        assert "boom" in result

    def test_call_swallows_session_exception_into_error_string(self):
        session = _make_session_mock([_make_mcp_tool("search_wikipedia")])
        session.call_tool = AsyncMock(side_effect=RuntimeError("transport died"))

        async def body(gw):
            return await gw.call("search_wikipedia", {"topic": "x"})

        result = _run_with_gateway(session, _DUMMY_SERVERS, body)
        assert "MCP tool error" in result
        assert "transport died" in result


class TestUntrustedGuards:
    """Untrusted (third-party) servers get an SSRF URL guard + output wrapping (17d)."""

    def test_unsafe_url_blocked_before_dispatch(self):
        session = _make_session_mock([_make_mcp_tool("fetch")])

        async def body(gw):
            # link-local / cloud-metadata address — numeric, no DNS needed
            return await gw.call("fetch", {"url": "http://169.254.169.254/latest/meta-data"})

        result = _run_with_gateway(session, _UNTRUSTED_SERVERS, body)
        assert "refused" in result.lower()
        session.call_tool.assert_not_called()

    def test_safe_url_is_dispatched_and_output_wrapped(self):
        session = _make_session_mock([_make_mcp_tool("fetch")], call_text="PAGE BODY")

        async def body(gw):
            # public numeric IP — passes is_safe_url without a real DNS lookup
            return await gw.call("fetch", {"url": "http://8.8.8.8/page"})

        result = _run_with_gateway(session, _UNTRUSTED_SERVERS, body)
        session.call_tool.assert_called_once_with("fetch", {"url": "http://8.8.8.8/page"})
        assert "<untrusted_data" in result          # client-side spotlight applied
        assert "PAGE BODY" in result

    def test_trusted_output_is_not_double_wrapped(self):
        # Trusted servers already wrap server-side; the gateway must NOT wrap again.
        session = _make_session_mock(
            [_make_mcp_tool("search_wikipedia")],
            call_text="<untrusted_data>wiki</untrusted_data>",
        )

        async def body(gw):
            return await gw.call("search_wikipedia", {"topic": "x"})

        result = _run_with_gateway(session, _DUMMY_SERVERS, body)
        assert result == "<untrusted_data>wiki</untrusted_data>"

    def test_non_url_args_to_untrusted_server_pass_through(self):
        # The guard only blocks unsafe URLs; ordinary args must still work.
        session = _make_session_mock([_make_mcp_tool("fetch")], call_text="ok")

        async def body(gw):
            return await gw.call("fetch", {"topic": "not a url"})

        result = _run_with_gateway(session, _UNTRUSTED_SERVERS, body)
        session.call_tool.assert_called_once()
        assert "<untrusted_data" in result
