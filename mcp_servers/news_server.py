from mcp.server.fastmcp import FastMCP

# Reuse the existing tool implementation — the MCP server is a thin protocol
# wrapper, not a reimplementation. Aliased on import to avoid colliding with the
# decorated tool function below.
from agentforge.tools.news import get_top_news as _get_top_news

mcp = FastMCP("news-server")


@mcp.tool()
def get_top_news(topic: str) -> str:
    """Search HackerNews for recent top stories on a topic. Returns up to 5 stories with titles, points, and source domains. Use for tech news and developer-focused topics."""
    result = _get_top_news(topic)
    # Uniform rule across all our MCP servers: a successful result is always
    # wrapped in <untrusted_data>. Anything else (error string, no-stories-found)
    # is raised so FastMCP reports it as isError:true — an LLM-recoverable tool
    # error, not a protocol error.
    if not result.startswith("<untrusted_data"):
        raise ValueError(result)
    return result


if __name__ == "__main__":
    mcp.run()
