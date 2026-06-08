"""Shared MCP client gateway.

A single place that knows how to spawn the configured MCP servers, open one
session per server for the lifetime of an agent turn, discover their tools, and
dispatch tool calls. Both the ACT pipeline (``tools.run_llm_with_tools``) and the
ReAct pipeline (``reasoning.react_engine``) open one gateway per turn and share
it — so discovery, dispatch, logging, and the trust boundary live in exactly one
place instead of being copy-pasted into each pipeline.

WHY this exists (the AI-engineering concept):
    MCP decouples the agent from tool *implementations* — the agent discovers
    capabilities at runtime over a protocol instead of importing functions. This
    module is the agent's MCP *client*: the thin layer a host (Claude Desktop,
    Cursor, our agent) uses to talk to a fleet of servers. Production MCP SDKs
    expose the same shape (discover → schemas → call); LangChain calls its
    equivalent ``MultiServerMCPClient``.

Lifecycle — per turn:
    ``async with mcp_gateway(trace_id) as gw:`` opens every server subprocess
    once on enter and tears them all down on exit. All tool calls within the turn
    reuse the open sessions — no reconnect overhead between steps. This matters
    most for ReAct, which may call tools across several reasoning steps in one
    turn. Enter and exit happen in the same task (the calling coroutine), which
    keeps the SDK's anyio cancel scopes well-behaved.

Security boundary — tool output is untrusted:
    Our own MCP servers wrap their results in ``<untrusted_data>`` before
    returning (see ``tools/_safety.py`` / ``tools/wikipedia.py``); the gateway
    passes that through unchanged. The wrap is independent of trust-of-source —
    a server can be compromised or proxy attacker text — so it applies even to
    servers we wrote ourselves.
"""
import contextlib
import logging
from typing import Any, Dict, List, Optional

from mcp import ClientSession
from mcp.client.stdio import (
    StdioServerParameters,
    get_default_environment,
    stdio_client,
)

from agentforge.config import MCP_SERVERS
from agentforge.logger import log_event
from agentforge.safety import is_safe_url, wrap_untrusted

logger = logging.getLogger(__name__)


def mcp_tool_to_openai_schema(tool) -> dict:
    """Convert an MCP tool definition to the OpenAI function-calling schema shape.

    The same JSON Schema (``tool.inputSchema``) feeds both MCP and OpenAI
    function calling — one schema, multiple consumers, no re-marshaling.
    """
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema,
        },
    }


class MCPGateway:
    """Per-turn MCP client. Holds discovered tools and dispatches calls.

    Build it via the ``mcp_gateway`` async context manager, which manages the
    session lifecycle. After entering, these are populated:

    - ``catalog``        — ``[{"name", "description", "input_schema"}]`` for
                           rendering into a prompt (used by the ReAct loop).
    - ``openai_schemas`` — OpenAI function-calling schemas (used by the ACT loop).
    - ``has_tools``      — whether any tool was discovered at all.

    and ``await gw.call(name, args)`` dispatches one tool call.
    """

    def __init__(self, trace_id: Optional[str] = None):
        self.trace_id = trace_id
        self._sessions: Dict[str, Any] = {}        # tool_name -> live ClientSession
        self._trusted: Dict[str, bool] = {}        # tool_name -> is its server trusted?
        self._server_of: Dict[str, str] = {}       # tool_name -> server name (for logging)
        self.catalog: List[dict] = []              # [{name, description, input_schema}]
        self.openai_schemas: List[dict] = []       # OpenAI function-calling schemas

    @property
    def has_tools(self) -> bool:
        return bool(self.openai_schemas)

    async def _discover(self, stack: contextlib.AsyncExitStack) -> None:
        """Connect to each configured server, initialize, and list its tools.

        Reads the standard ``mcpServers`` config shape (Step 17d): a name -> config
        map where each config gives ``command`` / ``args`` / optional ``env`` and a
        ``trusted`` flag. A server that fails to start or respond is logged and
        skipped — one bad server must not blind the agent to the others. Sessions
        are entered on the caller-supplied ``stack`` so they stay open for the turn.
        """
        for name, cfg in MCP_SERVERS.items():
            # env: start from the SDK's minimal-safe default (which includes PATH so
            # launchers like `uvx` resolve) and layer this server's own env on top.
            # Per-server env keeps one server's secrets out of another's process.
            extra_env = cfg.get("env")
            env = {**get_default_environment(), **extra_env} if extra_env else None
            params = StdioServerParameters(
                command=cfg["command"], args=cfg.get("args", []), env=env
            )
            trusted = bool(cfg.get("trusted", False))
            try:
                read, write = await stack.enter_async_context(stdio_client(params))
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                tools_result = await session.list_tools()
                for tool in tools_result.tools:
                    self._sessions[tool.name] = session
                    self._trusted[tool.name] = trusted
                    self._server_of[tool.name] = name
                    self.catalog.append({
                        "name": tool.name,
                        "description": tool.description or "",
                        "input_schema": tool.inputSchema,
                    })
                    self.openai_schemas.append(mcp_tool_to_openai_schema(tool))
                log_event(
                    "mcp_discovery",
                    {
                        "server": name,
                        "trusted": trusted,
                        "tools": [t.name for t in tools_result.tools],
                    },
                    trace_id=self.trace_id,
                )
            except Exception as exc:
                logger.warning("MCP discovery failed for server '%s': %s", name, exc)

    @staticmethod
    def _unsafe_url_in(arguments: Dict[str, Any]):
        """Return ``(url, reason)`` for the first http(s) URL argument that fails the
        SSRF guard, or ``None`` if every URL-looking argument is safe.

        Scans top-level string argument values that look like http(s) URLs — broad
        enough to catch a fetch-style tool's ``url`` without hardcoding an arg name.
        """
        for value in (arguments or {}).values():
            if isinstance(value, str) and value.strip().lower().startswith(("http://", "https://")):
                ok, reason = is_safe_url(value)
                if not ok:
                    return value, reason
        return None

    async def call(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Dispatch one tool call on the already-open session for that tool.

        Returns the tool's text result. An unknown tool name (not discovered from
        any server) returns an *error string* — not a raised exception — so the
        LLM can read it and recover. Models occasionally hallucinate a tool that
        doesn't exist; this turns that into a recoverable observation, matching
        the dual-error model MCP itself uses (``isError`` results the LLM sees vs.
        protocol errors it never does).

        For **untrusted** servers (ones we didn't write), two extra guards apply
        (Step 17d): (1) an **SSRF URL guard** refuses to dispatch if an argument is
        an http(s) URL pointing at an internal/private address, and (2) the tool's
        output is **wrapped as untrusted data** client-side, since a third-party
        server won't mark its own output. Our own (trusted) servers already wrap
        server-side, so we don't double-wrap them.
        """
        session = self._sessions.get(tool_name)
        if session is None:
            log_event("mcp_unknown_tool", {"tool": tool_name}, trace_id=self.trace_id)
            return f"Error: Unknown tool '{tool_name}'"

        trusted = self._trusted.get(tool_name, False)
        server = self._server_of.get(tool_name, "external")

        if not trusted:
            blocked = self._unsafe_url_in(arguments)
            if blocked:
                url, reason = blocked
                log_event(
                    "mcp_url_blocked",
                    {"tool": tool_name, "server": server, "reason": reason},
                    trace_id=self.trace_id,
                )
                return f"Error: refused to request '{url}' — {reason}"

        try:
            result = await session.call_tool(tool_name, arguments)
            text = " ".join(c.text for c in result.content if hasattr(c, "text"))
            log_event(
                "mcp_tool_call",
                {"tool": tool_name, "server": server, "trusted": trusted, "is_error": result.isError},
                trace_id=self.trace_id,
            )
            if result.isError:
                return f"Tool error: {text}"
            # Spotlight untrusted output (Step 17d). Trusted servers self-wrap.
            return text if trusted else wrap_untrusted(text, source=server)
        except Exception as exc:
            return f"MCP tool error: {exc}"


@contextlib.asynccontextmanager
async def mcp_gateway(trace_id: Optional[str] = None):
    """Open an :class:`MCPGateway` for one agent turn, then tear it down.

    Usage::

        async with mcp_gateway(trace_id) as gw:
            if gw.has_tools:
                result = await gw.call(name, args)

    The whole server fleet is spawned on enter and closed on exit. Using a single
    ``AsyncExitStack`` driven inside this generator (entered and exited from the
    caller's task) keeps the MCP SDK's anyio cancel scopes in one task — the same
    proven pattern the original inline ACT loop used.
    """
    gw = MCPGateway(trace_id)
    async with contextlib.AsyncExitStack() as stack:
        await gw._discover(stack)
        yield gw
