"""
Configuration for the agent. Reads from environment variables with sensible defaults.
Optional: create a .env file or set env vars (OPENAI_API_KEY, OPENAI_MODEL, etc.).
"""
import os
import sys

# Optional: load .env file if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# OpenAI
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")  # None = use default

# Paths
AGENT_MEMORY_DIR = os.environ.get("AGENT_MEMORY_DIR", "memory")
AGENT_LOG_FILE = os.environ.get("AGENT_LOG_FILE", "agent_logs.jsonl")
# Single JSON file where the document corpus (chunks + embeddings) is stored for RAG.
AGENT_CORPUS_FILE = os.environ.get("AGENT_CORPUS_FILE", "corpus.json")
# Tool "pins": baseline fingerprints of each untrusted MCP tool's definition,
# recorded on first sight (trust-on-first-use). The gateway compares against these
# on later turns to detect rug pulls (a tool's definition silently changing after
# we trusted it). Runtime data — gitignored. (Step 17e gap C.)
AGENT_TOOL_PINS_FILE = os.environ.get("AGENT_TOOL_PINS_FILE", "tool_pins.json")

# Maximum estimated tokens allowed in conversation history before trimming.
# Conservative default leaves room for system prompts + RAG chunks (which can
# consume 1,000–2,000 tokens on DOCS_QA calls) within most model context windows.
HISTORY_TOKEN_BUDGET = int(os.environ.get("HISTORY_TOKEN_BUDGET", "2000"))

# MCP servers the agent connects to at runtime to discover and call tools.
#
# Follows the cross-vendor standard "mcpServers" shape (Claude Desktop / Cursor /
# VS Code): a named map where each entry declares HOW TO LAUNCH the server and
# whether we trust it. Tool names are NOT listed here — they are discovered via
# tools/list at runtime.
#
#   command : executable to spawn the server (stdio transport)
#   args    : arguments passed to that executable, in order
#   env     : extra environment variables for THIS server only (optional;
#             merged over the inherited default environment)
#   trusted : True for our own first-party servers; False for a third-party server
#             we did not write. The gateway treats an untrusted server's output as
#             untrusted data and guards its URL arguments against SSRF (Step 17d).
#
# Add a new server by adding an entry here — no other code changes needed.
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).parent.parent
_MCP_DIR = _REPO_ROOT / "mcp_servers"
MCP_SERVERS: dict = {
    "wikipedia": {"command": sys.executable, "args": [str(_MCP_DIR / "wikipedia_server.py")], "trusted": True},
    "weather":   {"command": sys.executable, "args": [str(_MCP_DIR / "weather_server.py")],   "trusted": True},
    "news":      {"command": sys.executable, "args": [str(_MCP_DIR / "news_server.py")],       "trusted": True},
    # Third-party: Anthropic reference web-fetch server. Launched via uvx and
    # pinned to a specific PyPI version (supply-chain hygiene). Untrusted: the
    # gateway wraps its output and blocks internal/private URLs (SSRF guard).
    "fetch":     {"command": "uvx", "args": ["--from", "mcp-server-fetch==2026.6.4", "mcp-server-fetch"], "trusted": False},
}
