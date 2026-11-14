"""Fake article MCP server — an EVAL FIXTURE for the Deny-honesty eval, never a real tool.

WHY THIS EXISTS:
    ``agentforge.deny_eval`` (issue #88) checks what the REAL model replies after
    the user clicks Deny on a gated tool call — including the case where the call
    already ran before a crash (issue #86). It needs a tool that:

    - is UNTRUSTED, so the gateway gates it behind approval (requires_approval
      defaults to ``not trusted``) and the whole approval flow runs;
    - takes a TITLE, not a URL, so the SSRF guard has no hostname to resolve and
      the eval makes no network call except to the model;
    - returns text with a unique code word, so a reply that leaks the withheld
      result can be found with a plain substring check.

    The eval points ``mcp_client.MCP_SERVERS`` at this server only for its run and
    restores the real config afterward. A normal run of the agent never starts it.

ONE TOOL:
    ``read_article(title)`` returns the text in ``AF_EVAL_ARTICLE``. If
    ``AF_EVAL_CALLS_FILE`` is set, it appends one line per call, so the eval can
    check that a Deny never runs the tool.

This module imports nothing from ``agentforge`` and makes no network calls.
"""
import os

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("eval-article-server")

_DEFAULT_ARTICLE = "No article found with that title."


@mcp.tool()
def read_article(title: str) -> str:
    """Read an encyclopedia article by its title and return the article text."""
    calls_file = os.environ.get("AF_EVAL_CALLS_FILE")
    if calls_file:
        with open(calls_file, "a", encoding="utf-8") as f:
            f.write(" ".join(str(title).split()) + "\n")
    return os.environ.get("AF_EVAL_ARTICLE", _DEFAULT_ARTICLE)


if __name__ == "__main__":
    mcp.run()
