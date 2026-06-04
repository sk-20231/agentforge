from mcp.server.fastmcp import FastMCP

# Reuse the existing tool implementation — the MCP server is a thin protocol
# wrapper, not a reimplementation. Aliased on import to avoid colliding with the
# decorated tool function below.
from agentforge.tools.weather import get_weather as _get_weather

mcp = FastMCP("weather-server")


@mcp.tool()
def get_weather(city: str) -> str:
    """Get current weather (temperature, conditions, wind) for a city by name."""
    result = _get_weather(city)
    # Uniform rule across all our MCP servers: a successful result is always
    # wrapped in <untrusted_data>. Anything else (error string, city-not-found)
    # is raised so FastMCP reports it as isError:true — an LLM-recoverable tool
    # error, not a protocol error.
    if not result.startswith("<untrusted_data"):
        raise ValueError(result)
    return result


if __name__ == "__main__":
    mcp.run()
