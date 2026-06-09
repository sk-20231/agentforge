from mcp.server.fastmcp import FastMCP

from agentforge.tools.wikipedia import wikipedia_lookup

mcp = FastMCP("wikipedia-server")


@mcp.tool()
def search_wikipedia(topic: str) -> str:
    """Look up a topic on Wikipedia and return a short summary."""
    # The tool function returns raw, sanitized text on success and *raises* on
    # any failure; FastMCP maps the raised exception to isError:true (an
    # LLM-recoverable tool error). The MCP gateway — not this server — wraps the
    # result as untrusted data with the turn's nonce (Step 17e).
    return wikipedia_lookup(topic)


if __name__ == "__main__":
    mcp.run()
