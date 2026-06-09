from mcp.server.fastmcp import FastMCP

# Reuse the existing tool implementation — the MCP server is a thin protocol
# wrapper, not a reimplementation. Aliased on import to avoid colliding with the
# decorated tool function below.
from agentforge.tools.news import get_top_news as _get_top_news

mcp = FastMCP("news-server")


@mcp.tool()
def get_top_news(topic: str) -> str:
    """Search HackerNews for recent top stories on a topic. Returns up to 5 stories with titles, points, and source domains. Use for tech news and developer-focused topics."""
    # The tool function returns raw, sanitized text on success and *raises* on
    # any failure; FastMCP maps the raised exception to isError:true (an
    # LLM-recoverable tool error). The MCP gateway — not this server — wraps the
    # result as untrusted data with the turn's nonce (Step 17e).
    return _get_top_news(topic)


if __name__ == "__main__":
    mcp.run()
