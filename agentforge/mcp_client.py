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
import sys
from typing import Any, Dict, List, Optional

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from agentforge.config import MCP_SERVERS
from agentforge.logger import log_event

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
        self.catalog: List[dict] = []              # [{name, description, input_schema}]
        self.openai_schemas: List[dict] = []       # OpenAI function-calling schemas

    @property
    def has_tools(self) -> bool:
        return bool(self.openai_schemas)

    async def _discover(self, stack: contextlib.AsyncExitStack) -> None:
        """Connect to each configured server, initialize, and list its tools.

        A server that fails to start or respond is logged and skipped — one bad
        server must not blind the agent to the others. The sessions are entered
        on the caller-supplied ``stack`` so they stay open for the whole turn.
        """
        for server_path in MCP_SERVERS:
            params = StdioServerParameters(command=sys.executable, args=[server_path])
            try:
                read, write = await stack.enter_async_context(stdio_client(params))
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                tools_result = await session.list_tools()
                for tool in tools_result.tools:
                    self._sessions[tool.name] = session
                    self.catalog.append({
                        "name": tool.name,
                        "description": tool.description or "",
                        "input_schema": tool.inputSchema,
                    })
                    self.openai_schemas.append(mcp_tool_to_openai_schema(tool))
                log_event(
                    "mcp_discovery",
                    {"server": server_path, "tools": [t.name for t in tools_result.tools]},
                    trace_id=self.trace_id,
                )
            except Exception as exc:
                logger.warning("MCP discovery failed for %s: %s", server_path, exc)

    async def call(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Dispatch one tool call on the already-open session for that tool.

        Returns the tool's text result. An unknown tool name (not discovered from
        any server) returns an *error string* — not a raised exception — so the
        LLM can read it and recover. Models occasionally hallucinate a tool that
        doesn't exist; this turns that into a recoverable observation, matching
        the dual-error model MCP itself uses (``isError`` results the LLM sees vs.
        protocol errors it never does).
        """
        session = self._sessions.get(tool_name)
        if session is None:
            log_event("mcp_unknown_tool", {"tool": tool_name}, trace_id=self.trace_id)
            return f"Error: Unknown tool '{tool_name}'"
        try:
            result = await session.call_tool(tool_name, arguments)
            text = " ".join(c.text for c in result.content if hasattr(c, "text"))
            log_event(
                "mcp_tool_call",
                {"tool": tool_name, "is_error": result.isError},
                trace_id=self.trace_id,
            )
            return f"Tool error: {text}" if result.isError else text
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
