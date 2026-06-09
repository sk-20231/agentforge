from mcp.server.fastmcp import FastMCP

# Reuse the existing tool implementation — the MCP server is a thin protocol
# wrapper, not a reimplementation. Aliased on import to avoid colliding with the
# decorated tool function below.
from agentforge.tools.weather import get_weather as _get_weather

mcp = FastMCP("weather-server")


@mcp.tool()
def get_weather(city: str) -> str:
    """Get current weather (temperature, conditions, wind) for a city by name."""
    # The tool function returns raw, sanitized text on success and *raises* on
    # any failure; FastMCP maps the raised exception to isError:true (an
    # LLM-recoverable tool error). The MCP gateway — not this server — wraps the
    # result as untrusted data with the turn's nonce (Step 17e).
    return _get_weather(city)


if __name__ == "__main__":
    mcp.run()
