"""Architecture-fitness test (Step 17e, Layer 2).

The MCP gateway (``agentforge/mcp_client.py``) is the single trust boundary: it is
the ONLY place allowed to invoke a raw MCP session, because it is where tool output
is wrapped as untrusted data with the turn's nonce. If any other module called a
session directly, it could feed RAW, un-spotlighted tool text to the LLM — exactly
the bypass that Option A's clean-architecture design depends on never happening.

This test enforces that invariant *mechanically*: ``.call_tool(`` and ``._sessions``
may appear only in the gateway file inside the ``agentforge`` package. Any future
code that bypasses the gateway turns the build red before it can merge. This is an
"architecture fitness function" — the rule lives in CI, not in reviewers' memory.

(The real-subprocess contract tests in ``tests/`` legitimately drive a raw SDK
session, so only the shipped package is scanned — not the test suite itself.)
"""
from pathlib import Path

import pytest

PACKAGE_DIR = Path(__file__).parent.parent / "agentforge"
GATEWAY_FILE = "mcp_client.py"

# Substrings that mean "talking to a raw MCP session" — only the gateway may.
FORBIDDEN = [".call_tool(", "._sessions"]


def _package_py_files():
    return [p for p in PACKAGE_DIR.rglob("*.py") if "__pycache__" not in p.parts]


def test_package_dir_exists():
    # Guard against a wrong relative path silently making the scan pass on nothing.
    assert PACKAGE_DIR.is_dir(), f"package dir not found: {PACKAGE_DIR}"
    assert (PACKAGE_DIR / GATEWAY_FILE).is_file(), "gateway file moved? update this test"


@pytest.mark.parametrize("pattern", FORBIDDEN)
def test_only_gateway_touches_raw_sessions(pattern):
    offenders = []
    for path in _package_py_files():
        if path.name == GATEWAY_FILE:
            continue
        if pattern in path.read_text(encoding="utf-8"):
            offenders.append(str(path.relative_to(PACKAGE_DIR)))
    assert not offenders, (
        f"{pattern!r} found outside {GATEWAY_FILE} in: {offenders}. "
        "Only the MCP gateway may invoke a raw session — route every tool call "
        "through gw.call(), which wraps output as untrusted data (Step 17e)."
    )
