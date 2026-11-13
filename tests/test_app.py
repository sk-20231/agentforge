"""Tests for the Streamlit page, app.py (issue #88).

AppTest runs the REAL page script. The agent behind it is faked: the page
imports run_agent / resume_agent / pending_approval / … from agentforge.main
on every run, so replacing those attributes is enough. No LLM, no MCP server,
no network. What these check is the page's own logic — which card it shows,
which decision and interrupt ID a click sends, and how each outcome reads.
"""
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from agentforge import hitl_graph, main
from agentforge.approval import APPROVE_TURN, ApprovalRequest, ApprovalRequired

APP = str(Path(__file__).resolve().parent.parent / "app.py")
ARGS = {"url": "https://a.test/page"}


def _pending(thread_id="act-1", interrupt_id="i1", already_ran=False, question="q"):
    return hitl_graph.PendingApproval(
        thread_id=thread_id, pipeline="act",
        request=ApprovalRequest(tool="fetch", server="fetch", arguments=ARGS),
        user_id="demo_user", trace_id="t1", values={"user_input": question},
        interrupt_id=interrupt_id, already_ran=already_ran)


class FakeAgent:
    """Stands in for agentforge.main. Records what the page calls."""

    def __init__(self):
        self.calls = []
        self.pending = {}            # thread_id -> PendingApproval
        self.restorable = []         # what pending_approvals() returns
        self.read_error = None       # raised by pending_approval()
        self.resume_error = None     # raised by resume_agent()
        self.already_ran = False

    def run_agent(self, user_id, session_id, user_input, **kw):
        self.calls.append(("run", user_input))
        self.pending["act-1"] = _pending(already_ran=self.already_ran, question=user_input)
        raise ApprovalRequired(self.pending["act-1"].request, thread_id="act-1")

    def pending_approval(self, thread_id):
        if self.read_error is not None:
            raise self.read_error
        if thread_id not in self.pending:
            raise ValueError("This approval is no longer pending and cannot be resumed.")
        return self.pending[thread_id]

    def resume_agent(self, thread_id, decision, approval_handler=None, interrupt_id=None):
        self.calls.append(("resume", thread_id, decision, interrupt_id))
        if self.resume_error is not None:
            raise self.resume_error
        self.pending.pop(thread_id, None)
        return "RESUMED ANSWER"

    def pending_approvals(self, user_id):
        self.calls.append(("list", user_id))
        return self.restorable

    def discard_approval(self, thread_id, reason="abandoned"):
        self.calls.append(("discard", thread_id, reason))
        self.pending.pop(thread_id, None)

    def sweep_stale_runs(self):
        self.calls.append(("sweep",))
        return {}


@pytest.fixture
def agent(monkeypatch):
    fake = FakeAgent()
    for name in ("run_agent", "resume_agent", "pending_approval", "pending_approvals",
                 "discard_approval", "sweep_stale_runs"):
        monkeypatch.setattr(main, name, getattr(fake, name))
    monkeypatch.setattr("agentforge.tools.prime_tool_catalog", lambda: None)
    monkeypatch.setattr(hitl_graph, "is_durable", lambda: False)
    st.cache_resource.clear()        # the once-per-process sweep, fresh per test
    return fake


def _app():
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _ask(at, text="fetch the page"):
    at.chat_input[0].set_value(text).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _click(at, label_start):
    """Click, then run once more. A click here reruns the page twice
    (st.rerun after the decision, again after the turn finishes), and AppTest
    then keeps the cut-short run's elements — the old card — although the
    session state is already cleared. A browser drops those; one plain run
    does the same here. It calls nothing new: tests that count calls still
    see exactly one resume."""
    [button] = [b for b in at.button if b.label.startswith(label_start)]
    button.click().run()
    assert not at.exception, [e.value for e in at.exception]
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _card_shown(at):
    return any('"fetch"' in c.value and "https://a.test/page" in c.value for c in at.code)


def _texts(at):
    """Every markdown, info and error text on the page."""
    return ([m.value for m in at.markdown] + [i.value for i in at.info]
            + [e.value for e in at.error])


# ------------------------------- pause and card -------------------------------

def test_a_paused_question_shows_the_card_and_blocks_new_input(agent):
    at = _ask(_app())
    assert ("run", "fetch the page") in agent.calls
    assert _card_shown(at)
    assert at.chat_input[0].disabled
    assert not any("already ran" in e.value for e in at.error)


def test_the_card_warns_when_the_call_already_ran(agent):
    agent.already_ran = True
    at = _ask(_app())
    assert any("already ran before the app restarted" in e.value for e in at.error)


@pytest.mark.parametrize("label, decision", [
    ("✅ Allow once", True),
    ("🔁 Allow for this turn", APPROVE_TURN),
    ("🚫 Deny", False),
])
def test_each_button_resumes_with_its_decision_and_the_cards_interrupt_id(agent, label, decision):
    at = _click(_ask(_app()), label)
    assert [c for c in agent.calls if c[0] == "resume"] == [("resume", "act-1", decision, "i1")]
    assert "RESUMED ANSWER" in _texts(at)
    assert not _card_shown(at)
    assert not at.chat_input[0].disabled


# ------------------------------- restart -------------------------------

def test_durable_mode_restores_the_users_paused_card(agent, monkeypatch):
    monkeypatch.setattr(hitl_graph, "is_durable", lambda: True)
    agent.pending["act-9"] = _pending("act-9", "i9", question="the old question")
    agent.restorable = [agent.pending["act-9"]]
    at = _app()
    assert ("list", "demo_user") in agent.calls
    assert _card_shown(at)
    assert "the old question" in [m.value for m in at.markdown]
    at = _click(at, "✅ Allow once")
    assert ("resume", "act-9", True, "i9") in agent.calls


def test_memory_mode_does_not_look_for_paused_cards(agent):
    agent.restorable = [_pending("act-9", "i9")]
    at = _app()
    assert not any(c[0] == "list" for c in agent.calls)
    assert not _card_shown(at)


def test_the_startup_sweep_runs(agent):
    _app()
    assert ("sweep",) in agent.calls


# ------------------------------- refusals and failures -------------------------------

def test_an_expired_approval_says_so_and_clears_the_card(agent):
    at = _ask(_app())
    agent.read_error = hitl_graph.ApprovalExpired("This approval has expired and cannot be resumed.")
    at.run()
    assert "That approval expired. Please ask your question again." in [i.value for i in at.info]
    assert not _card_shown(at)
    assert not at.chat_input[0].disabled


def test_a_settled_approval_says_so_and_clears_the_card(agent):
    at = _ask(_app())
    agent.pending.clear()                         # settled elsewhere
    at.run()
    assert any("no longer pending" in i.value for i in at.info)
    assert not _card_shown(at)


def test_a_refused_resume_is_information_not_a_failure(agent):
    # Another tab holds the claim: nothing ran, so no "Something went wrong" (#86).
    message = "This approval is already being resumed, for example in another tab. It was not run again."
    agent.resume_error = hitl_graph.ApprovalNotPending(message)
    at = _click(_ask(_app()), "✅ Allow once")
    assert message in _texts(at)
    assert not any("Something went wrong" in t for t in _texts(at))


def test_a_real_failure_during_resume_is_still_reported(agent):
    agent.resume_error = RuntimeError("tool server crashed")
    at = _click(_ask(_app()), "✅ Allow once")
    assert "Something went wrong: tool server crashed" in _texts(at)


# ------------------------------- clear chat -------------------------------

def test_clear_chat_discards_the_pending_card(agent):
    at = _click(_ask(_app()), "Clear chat")
    assert ("discard", "act-1", "clear_chat") in agent.calls
    assert not _card_shown(at)
    assert not at.chat_input[0].disabled
