from mcp.server.fastmcp import FastMCP

mcp = FastMCP("wikipedia-server")

if __name__ == "__main__":
    mcp.run()
