"""
Contract tests for the news MCP server.

Spawns news_server.py as a real subprocess via the MCP SDK's stdio_client and
asserts on the actual wire responses. No mocking of the SDK — the SDK is what we
are testing through. Mirrors test_wikipedia_mcp_server.py.

Three contracts:
  1. initialize  → negotiated protocolVersion matches the pinned constant
  2. tools/list  → get_top_news present with a topic string parameter
  3. tools/call  → isError: true for invalid input; isError: false + untrusted
                   wrapper for a real topic
"""
import asyncio
import sys
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

# Pin the protocol version negotiated by the installed SDK. If an SDK upgrade
# changes the negotiated version this test fails loudly — review before bumping.
EXPECTED_PROTOCOL_VERSION = "2025-11-25"

SERVER_PATH = str(Path(__file__).parent.parent / "mcp_servers" / "news_server.py")


def _params() -> StdioServerParameters:
    return StdioServerParameters(command=sys.executable, args=[SERVER_PATH])


def _run(coro):
    """Run an async coroutine synchronously so pytest-asyncio is not required."""
    return asyncio.run(coro)


def test_protocol_version_is_pinned():
    """initialize result must carry the expected protocol version."""
    async def _run_inner():
        async with stdio_client(_params()) as (read, write):
            async with ClientSession(read, write) as session:
                result = await session.initialize()
                return result.protocolVersion

    version = _run(_run_inner())
    assert version == EXPECTED_PROTOCOL_VERSION, (
        f"Protocol version changed: expected {EXPECTED_PROTOCOL_VERSION!r}, got {version!r}. "
        "Update EXPECTED_PROTOCOL_VERSION after reviewing the SDK changelog."
    )


def test_tools_list_exposes_get_top_news():
    """tools/list must return get_top_news with a topic string parameter."""
    async def _run_inner():
        async with stdio_client(_params()) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await session.list_tools()

    result = _run(_run_inner())
    tool_names = [t.name for t in result.tools]
    assert "get_top_news" in tool_names, f"get_top_news missing from tools: {tool_names}"

    tool = next(t for t in result.tools if t.name == "get_top_news")
    schema_props = tool.inputSchema.get("properties", {})
    assert "topic" in schema_props, f"'topic' missing from inputSchema properties: {schema_props}"
    assert schema_props["topic"].get("type") == "string", "'topic' must be type string"


def test_tool_error_on_invalid_input():
    """tools/call with an empty topic must return isError: true (no network needed)."""
    async def _run_inner():
        async with stdio_client(_params()) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await session.call_tool("get_top_news", {"topic": ""})

    result = _run(_run_inner())
    assert result.isError is True, "Expected isError: true for empty topic"


def test_tool_success_wraps_result_as_untrusted():
    """tools/call with a real topic must return isError: false and an untrusted_data tag."""
    async def _run_inner():
        async with stdio_client(_params()) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await session.call_tool("get_top_news", {"topic": "Python"})

    result = _run(_run_inner())
    assert result.isError is False, "Expected isError: false for a real topic"
    text = " ".join(c.text for c in result.content if hasattr(c, "text"))
    assert "<untrusted_data" in text, "Result must be wrapped in <untrusted_data> tag"
