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

Security boundary — the gateway owns the untrusted-data wrap (Step 17e):
    The gateway is the single doorway between any MCP tool and the LLM, so it is
    the one place that wraps tool output as ``<untrusted_data_<nonce>>`` before it
    can reach the model. It wraps EVERY server's output — trusted and untrusted
    alike — because the wrap is independent of trust-of-source: a server can be
    compromised or proxy attacker text. The servers themselves no longer wrap
    (their tool functions return raw, sanitized text and *raise* on failure).
    Each turn gets a fresh random ``nonce`` so untrusted content cannot forge the
    closing delimiter (see ``safety.wrap_untrusted``). Two test layers protect
    this invariant: ``test_mcp_client`` asserts ``call()`` always wraps (incl. the
    delimiter-forging case), and ``test_architecture`` asserts nothing outside
    this module ever touches a raw session — so the gateway can't be bypassed.
"""
import contextlib
import json
import logging
import time
from typing import Any, Dict, List, Optional

from mcp import ClientSession
from mcp.client.stdio import (
    StdioServerParameters,
    get_default_environment,
    stdio_client,
)

from agentforge.approval import ApprovalHandler, ApprovalRequest
from agentforge.config import MCP_SERVERS, AGENT_TOOL_PINS_FILE
from agentforge.logger import log_event
from agentforge.safety import (
    fingerprint_tool,
    is_safe_url,
    new_spotlight_nonce,
    sanitize_external_block,
    wrap_untrusted,
)

logger = logging.getLogger(__name__)

# Max chars of an untrusted server's tool DESCRIPTION we keep before it enters a
# prompt. Tool descriptions are an attack surface ("tool poisoning"), so untrusted
# ones are sanitized + bounded like any other untrusted text (Step 17e gap C).
_MAX_UNTRUSTED_DESC = 1000


def _load_pins() -> Dict[str, str]:
    """Load the trust-on-first-use tool fingerprints, or {} if none yet."""
    try:
        with open(AGENT_TOOL_PINS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_pins(pins: Dict[str, str]) -> None:
    """Persist tool fingerprints. Best-effort: a write failure must not break a turn."""
    try:
        with open(AGENT_TOOL_PINS_FILE, "w", encoding="utf-8") as f:
            json.dump(pins, f, indent=2, sort_keys=True)
    except OSError as exc:
        logger.warning("Could not write tool pins file %s: %s", AGENT_TOOL_PINS_FILE, exc)


def mcp_tool_to_openai_schema(tool, description: Optional[str] = None) -> dict:
    """Convert an MCP tool definition to the OpenAI function-calling schema shape.

    The same JSON Schema (``tool.inputSchema``) feeds both MCP and OpenAI
    function calling — one schema, multiple consumers, no re-marshaling.

    ``description`` overrides ``tool.description`` when given — the gateway passes a
    sanitized description for untrusted servers (Step 17e gap C) so a poisoned
    description never reaches the model unfiltered.
    """
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": description if description is not None else (tool.description or ""),
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

    def __init__(self, trace_id: Optional[str] = None,
                 approval_handler: Optional[ApprovalHandler] = None):
        self.trace_id = trace_id
        # Front-end-supplied human-in-the-loop callback (Step 17f). None means
        # "no way to ask a human" — gated calls are then DENIED by default;
        # nothing silently auto-approves.
        self.approval_handler = approval_handler
        # One unguessable nonce per turn. Every untrusted-data wrap this gateway
        # emits uses it, so attacker-controlled tool output can't forge the
        # closing delimiter (Step 17e spotlighting).
        self.nonce = new_spotlight_nonce()
        self._sessions: Dict[str, Any] = {}        # tool_name -> live ClientSession
        self._trusted: Dict[str, bool] = {}        # tool_name -> is its server trusted?
        self._requires_approval: Dict[str, bool] = {}  # tool_name -> gate behind a human? (17f)
        self._server_of: Dict[str, str] = {}       # tool_name -> server name (for logging)
        self._blocked: Dict[str, str] = {}         # tool_name -> reason it was refused (17e C)
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
        pins = _load_pins()          # trust-on-first-use fingerprints (untrusted tools)
        pins_changed = False

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
            # Step 17f: gate this server's calls behind a human? Defaults to
            # "yes for third-party servers, no for our own" (see config.py).
            requires_approval = bool(cfg.get("requires_approval", not trusted))
            try:
                read, write = await stack.enter_async_context(stdio_client(params))
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                tools_result = await session.list_tools()
                for tool in tools_result.tools:
                    if self._register_tool(tool, session, name, trusted, pins,
                                           requires_approval=requires_approval):
                        pins_changed = True
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

        if pins_changed:
            _save_pins(pins)

    def _register_tool(self, tool, session, server: str, trusted: bool, pins: Dict[str, str],
                       requires_approval: bool = False) -> bool:
        """Vet one discovered tool and register it (Step 17e gap C).

        Refuses (does not register, so the model never sees it) a tool that is:
          - **shadowing** — another server already claimed this tool name (the first
            server to register a name wins; list trusted servers first in config); or
          - a **rug pull** — an untrusted tool whose fingerprint differs from the one
            we pinned the first time we saw it.
        Untrusted tool **descriptions** are sanitized + length-bounded before they can
        enter a prompt (tool-poisoning defence). Returns True if ``pins`` was updated
        (a new trust-on-first-use baseline was recorded) so the caller can persist it.
        """
        tname = tool.name

        # 1. Tool shadowing — a name collision across servers is suspicious. The
        # FIRST server to claim a name wins (list trusted servers first in config);
        # the later duplicate is dropped + logged, NOT registered. We must not block
        # the name itself here, or we'd disable the legitimate first tool too.
        if tname in self._sessions:
            log_event(
                "mcp_tool_blocked",
                {"tool": tname, "server": server,
                 "reason": f"shadowed: name already provided by '{self._server_of.get(tname)}'"},
                trace_id=self.trace_id,
            )
            return False

        # 2. Description is an attack surface — sanitize untrusted ones.
        description = tool.description or ""
        if not trusted:
            description = sanitize_external_block(description, max_length=_MAX_UNTRUSTED_DESC)

        # 3. Rug-pull detection (untrusted only): trust-on-first-use fingerprint.
        pins_changed = False
        if not trusted:
            fp = fingerprint_tool(tname, tool.description or "", tool.inputSchema)
            pin_key = f"{server}::{tname}"
            if pin_key not in pins:
                pins[pin_key] = fp                      # first sight — trust + record
                pins_changed = True
            elif pins[pin_key] != fp:
                reason = "pin_mismatch: tool definition changed since first trusted (possible rug pull)"
                self._blocked[tname] = reason
                log_event("mcp_tool_blocked", {"tool": tname, "server": server, "reason": reason},
                          trace_id=self.trace_id)
                return pins_changed

        # 4. Passed all checks — register it.
        self._sessions[tname] = session
        self._trusted[tname] = trusted
        self._requires_approval[tname] = requires_approval
        self._server_of[tname] = server
        self.catalog.append({"name": tname, "description": description, "input_schema": tool.inputSchema})
        self.openai_schemas.append(mcp_tool_to_openai_schema(tool, description=description))
        return pins_changed

    @staticmethod
    def _unsafe_url_in(arguments: Dict[str, Any]):
        """Return ``(url, reason)`` for the first http(s) URL anywhere in ``arguments``
        that fails the SSRF guard, or ``None`` if every URL-looking value is safe.

        Recurses through nested dicts and lists — not just top-level args (Step 17e
        gap D) — so a URL tucked inside an object/array (e.g. ``{"options": {"url":
        ...}}`` or ``{"urls": [...]}``) is still caught. Matches by value shape
        (starts with http/https), not by argument name, so it works for any tool.

        Known limit (documented, not fixed here): this is a **check-time (TOCTOU)**
        defence. The third-party server performs the actual fetch, so a hostname
        that resolves to a public IP at check time could resolve to a private one at
        request time (**DNS rebinding**). Closing that needs the fetching client to
        pin the resolved IP — which lives inside the server we don't control. The
        guard stays best-effort; full containment of a hostile server is gap G
        (sandboxing).
        """
        def _walk(value):
            if isinstance(value, str):
                if value.strip().lower().startswith(("http://", "https://")):
                    ok, reason = is_safe_url(value)
                    if not ok:
                        return value, reason
                return None
            if isinstance(value, dict):
                for v in value.values():
                    hit = _walk(v)
                    if hit:
                        return hit
            elif isinstance(value, (list, tuple)):
                for v in value:
                    hit = _walk(v)
                    if hit:
                        return hit
            return None

        return _walk(arguments or {})

    async def call(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Dispatch one tool call on the already-open session for that tool.

        Returns the tool's text result. An unknown tool name (not discovered from
        any server) returns an *error string* — not a raised exception — so the
        LLM can read it and recover. Models occasionally hallucinate a tool that
        doesn't exist; this turns that into a recoverable observation, matching
        the dual-error model MCP itself uses (``isError`` results the LLM sees vs.
        protocol errors it never does).

        Every successful result is **wrapped as untrusted data** here, with this
        turn's random nonce, before it is returned toward the LLM (Step 17e). The
        wrap applies to trusted and untrusted servers alike — the gateway is the
        single trust boundary and does not rely on a server to mark its own
        output. For **untrusted** servers (ones we didn't write) one extra guard
        applies (Step 17d): an **SSRF URL guard** refuses to dispatch if an
        argument is an http(s) URL pointing at an internal/private address.
        """
        # Audit args by KEY NAME only — values can hold PII/secrets (a user query,
        # a tokened URL) and must never reach the log sink (Step 17e gap A).
        arg_keys = sorted((arguments or {}).keys())
        server = self._server_of.get(tool_name, "external")
        trusted = self._trusted.get(tool_name, False)

        # Refused at discovery (shadowing / rug-pull pin mismatch). Such tools are
        # never advertised, so the model shouldn't name one — but if it does, this
        # turns it into a recoverable, audited observation (Step 17e gap C).
        if tool_name in self._blocked:
            reason = self._blocked[tool_name]
            self._audit(tool_name, server, trusted, arg_keys, "blocked", reason=reason)
            return f"Error: tool '{tool_name}' is blocked — {reason}"

        session = self._sessions.get(tool_name)
        if session is None:
            self._audit(tool_name, "?", False, arg_keys, "unknown_tool")
            return f"Error: Unknown tool '{tool_name}'"

        if not trusted:
            blocked = self._unsafe_url_in(arguments)
            if blocked:
                url, reason = blocked
                self._audit(tool_name, server, trusted, arg_keys, "url_blocked", reason=reason)
                return f"Error: refused to request '{url}' — {reason}"

        # Step 17f — human-in-the-loop gate (after the automated guards, so a
        # human is never asked to bless a call SSRF/pinning would refuse anyway).
        # Automated checks catch known-bad patterns; this gate covers the calls
        # that look legitimate to every rule but only a human can judge (e.g.
        # fetch of an attacker's *public* URL carrying exfiltrated context).
        # The handler may raise ApprovalRequired — that deliberately unwinds the
        # turn so a front-end that can't block (Streamlit) can ask via its own
        # UI and replay; it must not be caught here.
        if self._requires_approval.get(tool_name, False):
            request = ApprovalRequest(tool=tool_name, server=server,
                                      arguments=arguments or {})
            self._audit(tool_name, server, trusted, arg_keys, "approval_requested")
            if self.approval_handler is None:
                # No way to ask a human -> deny by default. A readable string,
                # not an exception, so the model can observe and adapt.
                self._audit(tool_name, server, trusted, arg_keys, "denied",
                            reason="no approval handler configured")
                return (f"Error: tool '{tool_name}' requires human approval and no "
                        f"approval handler is configured — the call was not made.")
            if not self.approval_handler(request):
                self._audit(tool_name, server, trusted, arg_keys, "denied",
                            reason="user declined")
                return (f"Error: the user declined permission for tool "
                        f"'{tool_name}' — the call was not made.")
            self._audit(tool_name, server, trusted, arg_keys, "approved")

        start = time.perf_counter()
        try:
            result = await session.call_tool(tool_name, arguments)
            text = " ".join(c.text for c in result.content if hasattr(c, "text"))
            duration_ms = (time.perf_counter() - start) * 1000
            if result.isError:
                self._audit(tool_name, server, trusted, arg_keys, "tool_error",
                            result_len=len(text), duration_ms=duration_ms)
                return f"Tool error: {text}"
            # Untrusted (third-party) output reaches us un-sanitized — strip the
            # machine-level tricks (control / zero-width chars) before wrapping
            # (Step 17e gap B). Our own tools already sanitize content-aware, so we
            # don't re-sanitize trusted output here.
            if not trusted:
                text = sanitize_external_block(text)
            self._audit(tool_name, server, trusted, arg_keys, "ok",
                        result_len=len(text), duration_ms=duration_ms)
            # Spotlight ALL tool output with this turn's nonce (Step 17e). The
            # gateway is the single trust boundary: it wraps trusted and untrusted
            # servers alike rather than trusting any server to mark its own output.
            return wrap_untrusted(text, source=server, nonce=self.nonce)
        except Exception as exc:
            duration_ms = (time.perf_counter() - start) * 1000
            self._audit(tool_name, server, trusted, arg_keys, "exception",
                        error=str(exc), duration_ms=duration_ms)
            return f"MCP tool error: {exc}"

    def _audit(self, tool, server, trusted, arg_keys, outcome, duration_ms=None, **extra):
        """Emit one uniform audit line per tool dispatch (Step 17e gap A).

        Every exit path of ``call()`` routes through here, so the JSONL log carries
        a complete, consistent trail: which server's tool the agent invoked, with
        which argument *names* (never values), and the outcome — ``ok`` /
        ``tool_error`` / ``url_blocked`` / ``unknown_tool`` / ``exception``.
        """
        log_event(
            "mcp_audit",
            {
                "tool": tool,
                "server": server,
                "trusted": trusted,
                "arg_keys": arg_keys,
                "outcome": outcome,
                **extra,
            },
            trace_id=self.trace_id,
            duration_ms=duration_ms,
        )


@contextlib.asynccontextmanager
async def mcp_gateway(trace_id: Optional[str] = None,
                      approval_handler: Optional[ApprovalHandler] = None):
    """Open an :class:`MCPGateway` for one agent turn, then tear it down.

    Usage::

        async with mcp_gateway(trace_id) as gw:
            if gw.has_tools:
                result = await gw.call(name, args)

    The whole server fleet is spawned on enter and closed on exit. Using a single
    ``AsyncExitStack`` driven inside this generator (entered and exited from the
    caller's task) keeps the MCP SDK's anyio cancel scopes in one task — the same
    proven pattern the original inline ACT loop used.

    ``approval_handler`` is the front-end's human-in-the-loop callback (Step
    17f); see ``agentforge.approval``. Omitted → gated tools are denied.
    """
    gw = MCPGateway(trace_id, approval_handler=approval_handler)
    async with contextlib.AsyncExitStack() as stack:
        await gw._discover(stack)
        yield gw
