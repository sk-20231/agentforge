"""Unit tests for the human-in-the-loop approval gate + resume (Step 17f).

The gate lives in MCPGateway.call() — the single trust boundary — and consults a
front-end-supplied approval handler for tools whose server has
``requires_approval`` (default: every untrusted server). A handler that cannot
answer synchronously (Streamlit) raises ApprovalRequired; the pipeline loop
attaches a continuation (frozen loop state) and the front-end later resumes the
turn with the human's decision. These tests mock the MCP transport the same way
tests/test_mcp_client.py does, so they are fast and hermetic.

The invariants under test:
  - gated + no handler        -> DENIED with a readable string (never dispatched)
  - gated + handler False     -> declined observation (never dispatched)
  - gated + handler True      -> dispatched, output still nonce-wrapped
  - not gated                 -> handler never consulted
  - handler raises ApprovalRequired -> propagates with the loop's continuation
  - resume settles the STORED call exactly once; new gated calls re-interrupt
  - automated guards run FIRST -> a human is never asked about an SSRF-blocked call
  - every request/decision audited; arg NAMES only, never values
  - APPROVE_TURN (issue #6) grants ONE (server, tool) for the REST OF THE TURN:
    asked once, later calls skip the gate, the grant rides the continuation
    across interrupts and dies with the turn (never leaks into a new gateway)
"""
import asyncio
import contextlib
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentforge.approval import (
    APPROVE_TURN,
    ApprovalRequest,
    ApprovalRequired,
    make_resume_handler,
    unwrap_approval_required,
)
from agentforge.mcp_client import mcp_gateway


# --------------------------- mock helpers ---------------------------
# Same shapes as tests/test_mcp_client.py (duplicated: test modules stay
# standalone rather than importing each other's private helpers).

def _make_mcp_tool(name, description="A tool", input_schema=None):
    tool = MagicMock()
    tool.name = name
    tool.description = description
    tool.inputSchema = input_schema or {
        "type": "object",
        "properties": {"topic": {"type": "string"}},
        "required": ["topic"],
    }
    return tool


def _make_session_mock(tools, call_text="result text", is_error=False):
    session = AsyncMock()
    session.initialize = AsyncMock()

    list_result = MagicMock()
    list_result.tools = tools
    session.list_tools = AsyncMock(return_value=list_result)

    content = MagicMock()
    content.text = call_text
    call_result = MagicMock()
    call_result.isError = is_error
    call_result.content = [content]
    session.call_tool = AsyncMock(return_value=call_result)

    return session


def _patch_mcp_transport(session_mock):
    stdio_cm = MagicMock()
    stdio_cm.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
    stdio_cm.__aexit__ = AsyncMock(return_value=None)
    mock_stdio = MagicMock(return_value=stdio_cm)

    session_cm = MagicMock()
    session_cm.__aenter__ = AsyncMock(return_value=session_mock)
    session_cm.__aexit__ = AsyncMock(return_value=None)
    mock_cs = MagicMock(return_value=session_cm)

    return mock_stdio, mock_cs


def _llm_response(content, tool_calls=None):
    """Build an OpenAI-shaped chat completion response mock."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.choices[0].message.tool_calls = tool_calls
    return response


# Server configs covering the requires_approval matrix. The flag DEFAULTS to
# ``not trusted``: untrusted servers are gated, first-party ones are not, and
# either default can be overridden explicitly.
_UNTRUSTED = {"ext": {"command": "uvx", "args": ["x"], "trusted": False}}
_TRUSTED = {"own": {"command": "python", "args": ["s.py"], "trusted": True}}
_TRUSTED_GATED = {"own": {"command": "python", "args": ["s.py"], "trusted": True,
                          "requires_approval": True}}
_UNTRUSTED_RELAXED = {"ext": {"command": "uvx", "args": ["x"], "trusted": False,
                              "requires_approval": False}}


def _gateway_patches(stack, session, servers, capture=None):
    """Enter the standard mocked-transport patches on an ExitStack."""
    mock_stdio, mock_cs = _patch_mcp_transport(session)

    def fake_log(event, payload, **kw):
        if capture is not None:
            capture.append((event, payload, kw))

    stack.enter_context(patch("agentforge.mcp_client.MCP_SERVERS", servers))
    stack.enter_context(patch("agentforge.mcp_client.log_event", fake_log))
    stack.enter_context(patch("agentforge.mcp_client._load_pins", lambda: {}))
    stack.enter_context(patch("agentforge.mcp_client._save_pins", lambda pins: None))
    stack.enter_context(patch("agentforge.mcp_client.stdio_client", mock_stdio))
    stack.enter_context(patch("agentforge.mcp_client.ClientSession", mock_cs))


def _call_through_gateway(session, servers, tool, args, handler=None, capture=None):
    """Open a gateway over a mocked transport and dispatch one call."""
    async def _go():
        async with mcp_gateway("tid", approval_handler=handler) as gw:
            return await gw.call(tool, args)

    with contextlib.ExitStack() as stack:
        _gateway_patches(stack, session, servers, capture)
        return asyncio.run(_go())


def _raising_handler(request):
    """The Streamlit-style handler: can never answer synchronously."""
    raise ApprovalRequired(request)


# --------------------------- the request contract ---------------------------

class TestApprovalRequest:
    def test_approval_required_carries_the_request(self):
        req = ApprovalRequest("fetch", "ext", {"url": "http://x.com"})
        exc = ApprovalRequired(req)
        assert exc.request is req
        assert "fetch" in str(exc)

    def test_continuation_defaults_to_none(self):
        exc = ApprovalRequired(ApprovalRequest("fetch", "ext", {}))
        assert exc.continuation is None


class TestMakeResumeHandler:
    def _pending(self):
        return ApprovalRequest("fetch", "ext", {"url": "http://x.com", "max": 5})

    def test_grants_the_decision_for_the_exact_pending_call(self):
        handler = make_resume_handler(True, self._pending())
        # same tool/server/args (fresh but equal dict) -> the stored decision
        assert handler(ApprovalRequest("fetch", "ext", {"max": 5, "url": "http://x.com"})) is True

    def test_deny_decision_is_returned_too(self):
        handler = make_resume_handler(False, self._pending())
        assert handler(self._pending()) is False

    def test_grant_is_one_shot(self):
        fallback = MagicMock(return_value=False)
        handler = make_resume_handler(True, self._pending(), fallback)
        assert handler(self._pending()) is True
        assert handler(self._pending()) is False   # second time -> fallback
        fallback.assert_called_once()

    def test_different_call_goes_to_fallback(self):
        fallback = MagicMock(return_value=True)
        handler = make_resume_handler(True, self._pending(), fallback)
        other = ApprovalRequest("fetch", "ext", {"url": "http://evil.com"})
        assert handler(other) is True
        fallback.assert_called_once_with(other)

    def test_no_fallback_denies_unknown_calls(self):
        handler = make_resume_handler(True, self._pending())
        other = ApprovalRequest("send", "ext", {"q": "x"})
        assert handler(other) is False


# --------------------------- the gate in gw.call() ---------------------------

class TestApprovalGate:
    def test_gated_with_no_handler_is_denied_by_default(self):
        # The fail-safe invariant: nothing ever silently auto-approves.
        session = _make_session_mock([_make_mcp_tool("fetch")])
        result = _call_through_gateway(session, _UNTRUSTED, "fetch", {"topic": "x"})
        assert "requires human approval" in result
        session.call_tool.assert_not_called()

    def test_gated_handler_deny_returns_declined_observation(self):
        # Deny is a readable observation the model can adapt to — not a crash.
        session = _make_session_mock([_make_mcp_tool("fetch")])
        handler = MagicMock(return_value=False)
        result = _call_through_gateway(session, _UNTRUSTED, "fetch", {"topic": "x"},
                                       handler=handler)
        assert "declined" in result.lower()
        session.call_tool.assert_not_called()
        handler.assert_called_once()
        req = handler.call_args.args[0]
        assert req.tool == "fetch" and req.server == "ext"
        assert req.arguments == {"topic": "x"}

    def test_gated_handler_approve_dispatches_and_output_is_wrapped(self):
        # Approval lets the call through; the 17e spotlight invariant still holds.
        session = _make_session_mock([_make_mcp_tool("fetch")], call_text="PAGE")
        handler = MagicMock(return_value=True)
        result = _call_through_gateway(session, _UNTRUSTED, "fetch", {"topic": "x"},
                                       handler=handler)
        session.call_tool.assert_called_once_with("fetch", {"topic": "x"})
        assert "<untrusted_data" in result
        assert "PAGE" in result

    def test_trusted_server_is_not_gated_by_default(self):
        session = _make_session_mock([_make_mcp_tool("search_wikipedia")], call_text="hi")
        handler = MagicMock(return_value=False)  # would deny if consulted
        result = _call_through_gateway(session, _TRUSTED, "search_wikipedia",
                                       {"topic": "x"}, handler=handler)
        handler.assert_not_called()
        session.call_tool.assert_called_once()
        assert "hi" in result

    def test_explicit_requires_approval_gates_a_trusted_server(self):
        session = _make_session_mock([_make_mcp_tool("send_email")])
        handler = MagicMock(return_value=False)
        result = _call_through_gateway(session, _TRUSTED_GATED, "send_email",
                                       {"to": "a@b.c"}, handler=handler)
        handler.assert_called_once()
        assert "declined" in result.lower()
        session.call_tool.assert_not_called()

    def test_explicit_relaxation_ungates_an_untrusted_server(self):
        session = _make_session_mock([_make_mcp_tool("fetch")], call_text="ok")
        handler = MagicMock(return_value=False)  # would deny if consulted
        _call_through_gateway(session, _UNTRUSTED_RELAXED, "fetch",
                              {"topic": "not a url"}, handler=handler)
        handler.assert_not_called()
        session.call_tool.assert_called_once()

    def test_approval_required_from_handler_propagates(self):
        # The interrupt contract (Streamlit): the exception must unwind out of
        # gw.call() untouched so the loop can attach its continuation.
        session = _make_session_mock([_make_mcp_tool("fetch")])

        with pytest.raises(ApprovalRequired) as excinfo:
            _call_through_gateway(session, _UNTRUSTED, "fetch", {"topic": "x"},
                                  handler=_raising_handler)
        assert excinfo.value.request.tool == "fetch"
        session.call_tool.assert_not_called()

    def test_automated_guards_run_before_the_human_is_asked(self):
        # An SSRF-blocked call must be refused WITHOUT consulting the handler —
        # a human is never asked to bless a call the machine would refuse anyway.
        session = _make_session_mock([_make_mcp_tool("fetch")])
        handler = MagicMock(return_value=True)
        result = _call_through_gateway(
            session, _UNTRUSTED, "fetch",
            {"url": "http://169.254.169.254/latest/meta-data"}, handler=handler)
        assert "refused" in result.lower()
        handler.assert_not_called()
        session.call_tool.assert_not_called()


# --------------------------- audit trail ---------------------------

class TestApprovalAudit:
    def _audits(self, events):
        return [p for (e, p, _kw) in events if e == "mcp_audit"]

    def test_no_handler_denial_is_audited(self):
        session = _make_session_mock([_make_mcp_tool("fetch")])
        events = []
        _call_through_gateway(session, _UNTRUSTED, "fetch",
                              {"topic": "S3CRET-VALUE"}, capture=events)
        outcomes = [a["outcome"] for a in self._audits(events)]
        assert outcomes == ["approval_requested", "denied"]
        # arg names only — the value must never reach the log sink
        assert "S3CRET-VALUE" not in json.dumps(events, default=str)

    def test_user_denial_is_audited(self):
        session = _make_session_mock([_make_mcp_tool("fetch")])
        events = []
        _call_through_gateway(session, _UNTRUSTED, "fetch", {"topic": "x"},
                              handler=MagicMock(return_value=False), capture=events)
        audits = self._audits(events)
        outcomes = [a["outcome"] for a in audits]
        assert outcomes == ["approval_requested", "denied"]
        assert audits[-1]["reason"] == "user declined"

    def test_approval_is_audited_then_call_proceeds(self):
        session = _make_session_mock([_make_mcp_tool("fetch")], call_text="ok")
        events = []
        # Disable the gap-E content guardrail so this asserts the pure APPROVAL
        # audit sequence. Otherwise an untrusted call also emits a
        # guardrail_unavailable line (fail-open) when the classifier model isn't
        # installed, which is the guardrail suite's concern, not this one.
        with patch("agentforge.mcp_client.AGENT_GUARDRAIL_ENABLED", False):
            _call_through_gateway(session, _UNTRUSTED, "fetch", {"topic": "x"},
                                  handler=MagicMock(return_value=True), capture=events)
        outcomes = [a["outcome"] for a in self._audits(events)]
        assert outcomes == ["approval_requested", "approved", "ok"]

    def test_ungated_call_emits_no_approval_events(self):
        session = _make_session_mock([_make_mcp_tool("search_wikipedia")], call_text="hi")
        events = []
        _call_through_gateway(session, _TRUSTED, "search_wikipedia", {"topic": "x"},
                              capture=events)
        outcomes = [a["outcome"] for a in self._audits(events)]
        assert outcomes == ["ok"]


# --------------------------- ExceptionGroup unwrapping ---------------------------
# Regression for a bug found in LIVE Streamlit testing that the mocked-transport
# tests above cannot reproduce: the real MCP stdio transport runs inside anyio
# task groups, so an ApprovalRequired raised by the handler reaches the front-end
# wrapped in an ExceptionGroup ("unhandled errors in a TaskGroup") — and
# `except ApprovalRequired` never matches. The sync→async bridges must unwrap.

class TestExceptionGroupUnwrap:
    def _req(self):
        return ApprovalRequest("fetch", "ext", {"url": "http://x.com"})

    def test_unwrap_returns_bare_exception_unchanged(self):
        exc = ApprovalRequired(self._req())
        assert unwrap_approval_required(exc) is exc

    def test_unwrap_finds_interrupt_inside_a_group(self):
        inner = ApprovalRequired(self._req())
        group = ExceptionGroup("unhandled errors in a TaskGroup", [inner])
        assert unwrap_approval_required(group) is inner

    def test_unwrap_finds_interrupt_inside_a_nested_group(self):
        inner = ApprovalRequired(self._req())
        nested = ExceptionGroup("outer", [ExceptionGroup("inner", [inner])])
        assert unwrap_approval_required(nested) is inner

    def test_unwrap_returns_none_for_unrelated_errors(self):
        group = ExceptionGroup("boom", [RuntimeError("transport died")])
        assert unwrap_approval_required(group) is None
        assert unwrap_approval_required(RuntimeError("x")) is None

    def test_react_bridge_reraises_bare_approval_required(self):
        from agentforge.reasoning import react_engine

        inner = ApprovalRequired(self._req())
        inner.continuation = {"pipeline": "react"}     # must survive the unwrap

        async def raising_loop(*args, **kwargs):
            raise ExceptionGroup("unhandled errors in a TaskGroup", [inner])

        with patch.object(react_engine, "_react_loop_async", raising_loop):
            with pytest.raises(ApprovalRequired) as excinfo:
                react_engine.react_loop("u", "fetch something")
        assert excinfo.value is inner          # the bare interrupt, not the group
        assert excinfo.value.continuation == {"pipeline": "react"}

    def test_act_bridge_reraises_bare_approval_required(self):
        from agentforge import tools as tools_pkg

        inner = ApprovalRequired(self._req())

        async def raising_loop(*args, **kwargs):
            raise ExceptionGroup("unhandled errors in a TaskGroup", [inner])

        with patch.object(tools_pkg, "_run_tool_loop", raising_loop), \
             patch.object(tools_pkg, "get_relevant_memories", return_value=""):
            with pytest.raises(ApprovalRequired) as excinfo:
                tools_pkg.run_llm_with_tools("u", "fetch something")
        assert excinfo.value is inner

    def test_bridges_leave_unrelated_groups_alone(self):
        # A group with no interrupt inside must propagate unchanged — the
        # bridge only normalizes the approval contract, it is not a catch-all.
        from agentforge.reasoning import react_engine

        async def raising_loop(*args, **kwargs):
            raise ExceptionGroup("boom", [RuntimeError("transport died")])

        with patch.object(react_engine, "_react_loop_async", raising_loop):
            with pytest.raises(ExceptionGroup):
                react_engine.react_loop("u", "anything")


# --------------------------- ReAct interrupt -> resume ---------------------------

_REACT_TOOL_STEP = json.dumps({
    "thought": "I need the page",
    "action": {"type": "tool", "tool_name": "fetch",
               "tool_input": {"url": "http://8.8.8.8/page"}},
})
_REACT_SECOND_TOOL_STEP = json.dumps({
    "thought": "Now another page",
    "action": {"type": "tool", "tool_name": "fetch",
               "tool_input": {"url": "http://9.9.9.9/other"}},
})
_REACT_FINAL_STEP = json.dumps({
    "thought": "done", "action": {"type": "final"},
    "reply": "Trip planned", "store_memory": False, "memory_text": "",
})


class TestReactResume:
    """End-to-end interrupt -> resume through the REAL gateway over a mocked
    transport, so the gate, the one-shot grant, and the spotlight wrap are all
    exercised for real (no fake gateway)."""

    def _react_env(self, stack, session, llm_contents):
        from agentforge.reasoning import react_engine
        _gateway_patches(stack, session, _UNTRUSTED)
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            _llm_response(c) for c in llm_contents
        ]
        stack.enter_context(patch.object(react_engine, "log_event"))
        stack.enter_context(patch.object(react_engine, "log_token_usage"))
        stack.enter_context(patch.object(react_engine, "get_relevant_memories",
                                         return_value="no memories"))
        stack.enter_context(patch.object(react_engine, "_get_client",
                                         return_value=client))
        return react_engine, client

    def _capture_interrupt(self, react_engine):
        with pytest.raises(ApprovalRequired) as excinfo:
            react_engine.react_loop("u1", "plan a trip",
                                    approval_handler=_raising_handler)
        return excinfo.value

    def test_interrupt_carries_the_react_continuation(self):
        session = _make_session_mock([_make_mcp_tool("fetch")])
        with contextlib.ExitStack() as stack:
            react_engine, _ = self._react_env(stack, session, [_REACT_TOOL_STEP])
            interrupt = self._capture_interrupt(react_engine)

        cont = interrupt.continuation
        assert cont["pipeline"] == "react"
        assert cont["user_id"] == "u1"
        assert cont["step"] == 0
        assert cont["tool_name"] == "fetch"
        assert cont["tool_input"] == {"url": "http://8.8.8.8/page"}
        assert cont["raw"] == _REACT_TOOL_STEP          # the step that proposed it
        assert any(m.get("content") == "plan a trip" for m in cont["messages"]
                   if m.get("role") == "user")
        session.call_tool.assert_not_called()           # interrupted before dispatch

    def test_resume_approve_dispatches_stored_call_and_finishes(self):
        session = _make_session_mock([_make_mcp_tool("fetch")], call_text="PAGE BODY")
        with contextlib.ExitStack() as stack:
            react_engine, client = self._react_env(
                stack, session, [_REACT_TOOL_STEP, _REACT_FINAL_STEP])
            interrupt = self._capture_interrupt(react_engine)
            reply = react_engine.resume_react_loop(interrupt, True,
                                                   approval_handler=_raising_handler)

        assert reply == "Trip planned"
        # The STORED call was dispatched — exactly once, exact arguments.
        session.call_tool.assert_called_once_with("fetch", {"url": "http://8.8.8.8/page"})
        # The continued loop saw the observation (wrapped) and the stored raw step.
        followup_messages = client.chat.completions.create.call_args.kwargs["messages"]
        assert any(m["role"] == "assistant" and m["content"] == _REACT_TOOL_STEP
                   for m in followup_messages)
        observation = next(m["content"] for m in followup_messages
                           if m["role"] == "user" and "Observation from tool" in m["content"])
        assert "PAGE BODY" in observation
        assert "<untrusted_data" in observation          # 17e wrap survives resume

    def test_resume_deny_feeds_declined_observation_and_continues(self):
        session = _make_session_mock([_make_mcp_tool("fetch")])
        with contextlib.ExitStack() as stack:
            react_engine, client = self._react_env(
                stack, session, [_REACT_TOOL_STEP, _REACT_FINAL_STEP])
            interrupt = self._capture_interrupt(react_engine)
            reply = react_engine.resume_react_loop(interrupt, False,
                                                   approval_handler=_raising_handler)

        assert reply == "Trip planned"                  # the model adapted
        session.call_tool.assert_not_called()           # denied -> never dispatched
        followup_messages = client.chat.completions.create.call_args.kwargs["messages"]
        observation = next(m["content"] for m in followup_messages
                           if m["role"] == "user" and "Observation from tool" in m["content"])
        assert "declined" in observation.lower()

    def test_new_gated_call_during_resume_interrupts_again(self):
        # The one-shot grant covers ONLY the stored call. If the resumed loop
        # proposes a DIFFERENT gated call, the fallback handler raises a fresh
        # interrupt — with its own continuation at the later step.
        session = _make_session_mock([_make_mcp_tool("fetch")], call_text="PAGE")
        with contextlib.ExitStack() as stack:
            react_engine, _ = self._react_env(
                stack, session, [_REACT_TOOL_STEP, _REACT_SECOND_TOOL_STEP])
            interrupt = self._capture_interrupt(react_engine)
            with pytest.raises(ApprovalRequired) as excinfo:
                react_engine.resume_react_loop(interrupt, True,
                                               approval_handler=_raising_handler)

        second = excinfo.value
        assert second.request.arguments == {"url": "http://9.9.9.9/other"}
        assert second.continuation["step"] == 1          # one step further in
        session.call_tool.assert_called_once()           # only the approved call ran

    def test_resume_with_approve_turn_covers_later_calls_to_same_tool(self):
        """Issue #6 end-to-end: same scenario as the test above, but the human
        answers "allow for the rest of this turn". The second fetch (different
        URL!) rides the grant instead of interrupting again — one card for the
        whole paging loop. The grant travels: continuation -> resumed gateway."""
        session = _make_session_mock([_make_mcp_tool("fetch")], call_text="PAGE")
        with contextlib.ExitStack() as stack:
            react_engine, _ = self._react_env(
                stack, session,
                [_REACT_TOOL_STEP, _REACT_SECOND_TOOL_STEP, _REACT_FINAL_STEP])
            interrupt = self._capture_interrupt(react_engine)
            assert interrupt.continuation["granted"] == set()   # nothing granted yet
            reply = react_engine.resume_react_loop(interrupt, APPROVE_TURN,
                                                   approval_handler=_raising_handler)

        assert reply == "Trip planned"                   # turn completed, no 2nd card
        assert session.call_tool.await_count == 2        # BOTH fetches dispatched
        # The grant was recorded in the turn's own state (the shared set).
        assert interrupt.continuation["granted"] == {("ext", "fetch")}


# --------------------------- turn-scoped grants (issue #6) ---------------------------

class TestTurnScopedGrant:
    """APPROVE_TURN allows one (server, tool) for the rest of the turn.

    The gateway interprets the decision: it records the pair in its ``granted``
    set; later calls to a granted tool dispatch without consulting the handler.
    The set is shared with the continuation, so the grant survives
    interrupt→resume — and a NEW turn builds a new gateway with a fresh set,
    so a grant can never outlive its turn.
    """

    def _run_calls(self, session, handler, calls, servers=_UNTRUSTED,
                   granted=None, capture=None):
        """Open ONE gateway (= one turn segment) and dispatch ``calls`` in order."""
        async def _go():
            async with mcp_gateway("tid", approval_handler=handler,
                                   granted=granted) as gw:
                return [await gw.call(tool, args) for tool, args in calls]

        with contextlib.ExitStack() as stack:
            _gateway_patches(stack, session, servers, capture)
            return asyncio.run(_go())

    def test_approve_turn_asks_once_then_skips_the_gate(self):
        """The paging case: 5 cards for one article becomes 1. Different args
        per call — the grant is per-TOOL, unlike the one-shot exact-match."""
        session = _make_session_mock([_make_mcp_tool("fetch")], call_text="PAGE")
        asked = []

        def handler(request):
            asked.append(request)
            return APPROVE_TURN

        results = self._run_calls(session, handler, [
            ("fetch", {"topic": "page 1"}),
            ("fetch", {"topic": "page 2"}),
            ("fetch", {"topic": "page 3"}),
        ])

        assert len(asked) == 1                       # human asked exactly once
        assert session.call_tool.await_count == 3    # all three dispatched
        for r in results:
            assert "<untrusted_data" in r            # 17e wrap still applies

    def test_allow_once_still_asks_every_time(self):
        """Plain True keeps the pre-#6 semantics — per-call approval."""
        session = _make_session_mock([_make_mcp_tool("fetch")], call_text="PAGE")
        asked = []

        def handler(request):
            asked.append(request)
            return True

        self._run_calls(session, handler, [
            ("fetch", {"topic": "page 1"}),
            ("fetch", {"topic": "page 2"}),
        ])

        assert len(asked) == 2

    def test_grant_does_not_cross_tools(self):
        """Granting fetch must not bless a different gated tool."""
        session = _make_session_mock(
            [_make_mcp_tool("fetch"), _make_mcp_tool("send_email")],
            call_text="OK")
        asked = []

        def handler(request):
            asked.append(request.tool)
            return APPROVE_TURN

        self._run_calls(session, handler, [
            ("fetch", {"topic": "a"}),
            ("fetch", {"topic": "b"}),       # covered by the fetch grant
            ("send_email", {"topic": "c"}),  # different tool -> ask again
        ])

        assert asked == ["fetch", "send_email"]

    def test_grant_dies_with_the_turn(self):
        """A new gateway (= a new turn) starts with a fresh empty set — the
        grant can never leak into the next question."""
        session = _make_session_mock([_make_mcp_tool("fetch")], call_text="PAGE")
        asked = []

        def handler(request):
            asked.append(request)
            return APPROVE_TURN

        self._run_calls(session, handler, [("fetch", {"topic": "turn 1"})])
        self._run_calls(session, handler, [("fetch", {"topic": "turn 2"})])

        assert len(asked) == 2                       # asked once PER TURN

    def test_grant_set_is_shared_with_the_caller(self):
        """The pipeline passes the continuation's set in; the gateway must
        mutate THAT object (not a copy) so resume sees the grant."""
        session = _make_session_mock([_make_mcp_tool("fetch")], call_text="PAGE")
        granted = set()

        self._run_calls(session, lambda r: APPROVE_TURN,
                        [("fetch", {"topic": "a"})], granted=granted)

        assert granted == {("ext", "fetch")}

    def test_resumed_gateway_honors_a_prior_grant(self):
        """A grant made before the interrupt covers calls after the resume:
        the resumed gateway receives the stored set and never asks."""
        session = _make_session_mock([_make_mcp_tool("fetch")], call_text="PAGE")

        def handler(request):  # pragma: no cover - must never run
            raise AssertionError("handler consulted despite an existing grant")

        results = self._run_calls(session, handler, [("fetch", {"topic": "a"})],
                                  granted={("ext", "fetch")})

        assert session.call_tool.await_count == 1
        assert "<untrusted_data" in results[0]

    def test_audit_distinguishes_grant_scopes(self):
        """The trail must show which calls a human looked at (turn_grant_created)
        versus which rode an earlier grant (turn_grant_used) — issue #6."""
        session = _make_session_mock([_make_mcp_tool("fetch")], call_text="PAGE")
        capture = []

        self._run_calls(session, lambda r: APPROVE_TURN, [
            ("fetch", {"topic": "a"}),
            ("fetch", {"topic": "b"}),
        ], capture=capture)

        scopes = [p.get("scope") for e, p, kw in capture
                  if e == "mcp_audit" and p.get("outcome") == "approved"]
        assert scopes == ["turn_grant_created", "turn_grant_used"]

    def test_once_approval_audited_as_once(self):
        session = _make_session_mock([_make_mcp_tool("fetch")], call_text="PAGE")
        capture = []

        self._run_calls(session, lambda r: True, [("fetch", {"topic": "a"})],
                        capture=capture)

        scopes = [p.get("scope") for e, p, kw in capture
                  if e == "mcp_audit" and p.get("outcome") == "approved"]
        assert scopes == ["once"]

    def test_make_resume_handler_transports_approve_turn(self):
        """The one-shot handler only carries the decision; APPROVE_TURN must
        arrive at the gateway unchanged."""
        pending = ApprovalRequest("fetch", "ext", {"url": "http://x.com"})
        handler = make_resume_handler(APPROVE_TURN, pending)
        assert handler(ApprovalRequest("fetch", "ext", {"url": "http://x.com"})) == APPROVE_TURN


# --------------------------- ACT interrupt -> resume ---------------------------

def _make_openai_tool_call(name, args_json, call_id="tc1"):
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = args_json
    return tc


_ACT_FOLLOWUP_JSON = json.dumps({
    "reply": "Here is the page summary", "store_memory": False, "memory_text": "",
})


class TestActResume:
    def _act_env(self, stack, session, llm_responses):
        from agentforge import tools as tools_pkg
        _gateway_patches(stack, session, _UNTRUSTED)
        client = MagicMock()
        client.chat.completions.create.side_effect = llm_responses
        stack.enter_context(patch.object(tools_pkg, "log_event"))
        stack.enter_context(patch.object(tools_pkg, "log_token_usage"))
        stack.enter_context(patch.object(tools_pkg, "get_relevant_memories",
                                         return_value="no memories"))
        stack.enter_context(patch.object(tools_pkg, "_get_client",
                                         return_value=client))
        return tools_pkg, client

    def test_act_interrupt_then_resume_approve(self):
        session = _make_session_mock([_make_mcp_tool("fetch")], call_text="PAGE")
        tool_call = _make_openai_tool_call("fetch", json.dumps({"url": "http://8.8.8.8/p"}))
        with contextlib.ExitStack() as stack:
            tools_pkg, _ = self._act_env(stack, session, [
                _llm_response(None, tool_calls=[tool_call]),   # picks the tool
                _llm_response(_ACT_FOLLOWUP_JSON),             # follow-up answer
            ])
            with pytest.raises(ApprovalRequired) as excinfo:
                tools_pkg.run_llm_with_tools("u1", "fetch the page",
                                             approval_handler=_raising_handler)
            interrupt = excinfo.value
            cont = interrupt.continuation
            assert cont["pipeline"] == "act"
            assert cont["pending_index"] == 0
            assert cont["user_id"] == "u1"
            session.call_tool.assert_not_called()

            result = tools_pkg.resume_tool_loop(interrupt, True,
                                                approval_handler=_raising_handler)

        session.call_tool.assert_called_once_with("fetch", {"url": "http://8.8.8.8/p"})
        assert json.loads(result)["reply"] == "Here is the page summary"

    def test_act_resume_deny_still_answers(self):
        session = _make_session_mock([_make_mcp_tool("fetch")])
        tool_call = _make_openai_tool_call("fetch", json.dumps({"url": "http://8.8.8.8/p"}))
        with contextlib.ExitStack() as stack:
            tools_pkg, client = self._act_env(stack, session, [
                _llm_response(None, tool_calls=[tool_call]),
                _llm_response(_ACT_FOLLOWUP_JSON),
            ])
            with pytest.raises(ApprovalRequired) as excinfo:
                tools_pkg.run_llm_with_tools("u1", "fetch the page",
                                             approval_handler=_raising_handler)
            result = tools_pkg.resume_tool_loop(excinfo.value, False,
                                                approval_handler=_raising_handler)

        session.call_tool.assert_not_called()
        # The declined observation reached the follow-up call as the tool message.
        followup_messages = client.chat.completions.create.call_args.kwargs["messages"]
        tool_msg = next(m for m in followup_messages if m.get("role") == "tool")
        assert "declined" in tool_msg["content"].lower()
        assert json.loads(result)["reply"] == "Here is the page summary"


# --------------------------- main.resume_agent dispatch ---------------------------

class TestResumeAgent:
    def _interrupt(self, pipeline, **extra):
        exc = ApprovalRequired(ApprovalRequest("fetch", "ext", {"url": "http://x.com"}))
        exc.continuation = {"pipeline": pipeline, "user_id": "u1",
                            "trace_id": "tid1", **extra}
        return exc

    def test_react_resume_is_dispatched(self):
        from agentforge import main as main_mod
        with patch.object(main_mod, "resume_react_loop", return_value="done") as rr, \
             patch.object(main_mod, "log_event"):
            result = main_mod.resume_agent(self._interrupt("react"), True)
        assert result == "done"
        rr.assert_called_once()

    def test_act_resume_parses_json_and_stores_memory(self):
        from agentforge import main as main_mod
        payload = json.dumps({"reply": "hi", "store_memory": True, "memory_text": "likes Greece"})
        with patch.object(main_mod, "resume_tool_loop", return_value=payload), \
             patch.object(main_mod, "store_memory") as sm, \
             patch.object(main_mod, "log_event"):
            result = main_mod.resume_agent(self._interrupt("act"), True)
        assert result == "hi"
        sm.assert_called_once_with("u1", "likes Greece")

    def test_act_resume_invalid_json_is_graceful(self):
        from agentforge import main as main_mod
        with patch.object(main_mod, "resume_tool_loop", return_value="not json"), \
             patch.object(main_mod, "log_event"):
            result = main_mod.resume_agent(self._interrupt("act"), True)
        assert "invalid tool response" in result.lower()

    def test_missing_continuation_raises(self):
        from agentforge import main as main_mod
        exc = ApprovalRequired(ApprovalRequest("fetch", "ext", {}))
        with pytest.raises(ValueError):
            main_mod.resume_agent(exc, True)


# --------------------------- pipeline threading ---------------------------

class TestHandlerThreading:
    def test_react_loop_passes_handler_to_gateway(self):
        # The loops stay security-ignorant: they pass the handler through to the
        # gateway untouched. Patch the gateway factory and inspect the kwargs.
        from agentforge.reasoning import react_engine

        captured = {}

        def fake_gateway(trace_id=None, approval_handler=None):
            captured["approval_handler"] = approval_handler

            class _FakeGw:
                catalog = []
                has_tools = False

                async def __aenter__(self):
                    return self

                async def __aexit__(self, *a):
                    return None

            return _FakeGw()

        sentinel = MagicMock(name="handler")
        fake_response = _llm_response(json.dumps(
            {"thought": "done", "action": {"type": "final"}, "reply": "hi",
             "store_memory": False, "memory_text": ""}))

        with patch.object(react_engine, "mcp_gateway", fake_gateway), \
             patch.object(react_engine, "get_relevant_memories", return_value=""), \
             patch.object(react_engine, "log_event"), \
             patch.object(react_engine, "log_token_usage"), \
             patch.object(react_engine, "_get_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = fake_response
            react_engine.react_loop("u", "do something", approval_handler=sentinel)

        assert captured["approval_handler"] is sentinel

    def test_act_loop_passes_handler_to_gateway(self):
        from agentforge import tools as tools_pkg

        captured = {}

        def fake_gateway(trace_id=None, approval_handler=None):
            captured["approval_handler"] = approval_handler

            class _FakeGw:
                catalog = []
                openai_schemas = []
                has_tools = False

                async def __aenter__(self):
                    return self

                async def __aexit__(self, *a):
                    return None

            return _FakeGw()

        sentinel = MagicMock(name="handler")
        with patch.object(tools_pkg, "mcp_gateway", fake_gateway), \
             patch.object(tools_pkg, "get_relevant_memories", return_value=""), \
             patch.object(tools_pkg, "log_event"):
            tools_pkg.run_llm_with_tools("u", "weather?", approval_handler=sentinel)

        assert captured["approval_handler"] is sentinel
