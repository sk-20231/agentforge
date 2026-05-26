from mcp.server.fastmcp import FastMCP

from agentforge.tools.wikipedia import wikipedia_lookup

mcp = FastMCP("wikipedia-server")


@mcp.tool()
def search_wikipedia(topic: str) -> str:
    """Look up a topic on Wikipedia and return a short summary."""
    result = wikipedia_lookup(topic)
    if not result.startswith("<untrusted_data"):
        raise ValueError(result)
    return result


if __name__ == "__main__":
    mcp.run()
