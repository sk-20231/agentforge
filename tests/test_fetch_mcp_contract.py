"""Contract tests for the THIRD-PARTY fetch MCP server (Step 17d).

Unlike our own servers, this one we did not write — it's `mcp-server-fetch`,
pinned to a PyPI version and launched via `uvx`. These tests spawn the real
server as a subprocess (no SDK mocks) and assert that the contract we depend on
still holds: the negotiated protocol version, that a `fetch` tool with a `url`
string exists, and that calling it on a real URL returns content. This is the
"dependency-on-strangers" guard — if a future version drops/changes the tool, or
the protocol shifts, these fail loudly instead of the agent breaking silently.

Because it's a network/third-party dependency (first run downloads the package),
every test SKIPS — not fails — when uvx is missing or the server can't be reached.
"""
import asyncio
import shutil
import subprocess

import pytest
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

# Pin: the exact PyPI version we consume. Bump deliberately (supply-chain hygiene).
FETCH_VERSION = "2026.6.4"
FETCH_ARGS = ["--from", f"mcp-server-fetch=={FETCH_VERSION}", "mcp-server-fetch"]

# Same SDK-negotiated protocol version our other contract tests pin.
EXPECTED_PROTOCOL_VERSION = "2025-11-25"

_UVX = shutil.which("uvx")
pytestmark = pytest.mark.skipif(_UVX is None, reason="uvx not installed — third-party fetch server unavailable")


@pytest.fixture(scope="module", autouse=True)
def _prewarm():
    """Download/resolve the pinned package once so per-test handshakes don't time
    out on a cold cache. Best-effort: ignore failures (offline CI just skips)."""
    if _UVX is None:
        return
    try:
        subprocess.run(
            [_UVX, "--from", f"mcp-server-fetch=={FETCH_VERSION}", "python", "-c", "pass"],
            capture_output=True, timeout=180,
        )
    except Exception:
        pass


def _params() -> StdioServerParameters:
    return StdioServerParameters(command="uvx", args=FETCH_ARGS)


def _connect(coro_fn):
    """Run coro_fn(session) against the live server; skip on any launch/connection
    failure (offline, download failed, uvx resolve, handshake timeout)."""
    async def _inner():
        async with stdio_client(_params()) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await coro_fn(session)

    try:
        return asyncio.run(_inner())
    except Exception as exc:  # noqa: BLE001 — third-party/network: skip, don't fail
        pytest.skip(f"third-party fetch server unavailable (network/download/uvx): {exc}")


def test_protocol_version_is_pinned():
    async def _inner():
        async with stdio_client(_params()) as (read, write):
            async with ClientSession(read, write) as session:
                return (await session.initialize()).protocolVersion

    try:
        version = asyncio.run(_inner())
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"third-party fetch server unavailable: {exc}")
    assert version == EXPECTED_PROTOCOL_VERSION, (
        f"Negotiated protocol changed: expected {EXPECTED_PROTOCOL_VERSION!r}, got {version!r}. "
        "Review the SDK/server changelog before bumping."
    )


def test_tools_list_exposes_fetch_with_url():
    result = _connect(lambda s: s.list_tools())
    names = [t.name for t in result.tools]
    assert "fetch" in names, f"'fetch' tool missing — third-party contract broke: {names}"

    tool = next(t for t in result.tools if t.name == "fetch")
    props = (tool.inputSchema or {}).get("properties", {})
    assert "url" in props, f"'url' missing from fetch inputSchema: {props}"
    assert props["url"].get("type") == "string", "'url' must be type string"


def test_fetch_returns_content_for_real_url():
    result = _connect(lambda s: s.call_tool("fetch", {"url": "https://example.com"}))
    text = " ".join(c.text for c in result.content if hasattr(c, "text"))
    if result.isError:
        pytest.skip(f"fetch upstream unavailable, not a contract failure: {text}")
    assert text.strip(), "fetch returned empty content for a real URL"
