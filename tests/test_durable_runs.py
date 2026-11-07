"""Tests for durable runs (Step 31): the SQLite checkpointer, approval expiry,
the startup sweep, and finding a paused turn again after a restart.

A "restart" here is a second saver opened on the same file, with the graph
caches cleared so every graph recompiles against it — nothing carries over in
memory. The real process restart (kill, start again) is the live kill test.
"""
import asyncio
import contextlib
import importlib.util
import json
from datetime import datetime, timedelta, timezone
from typing import TypedDict
from unittest.mock import MagicMock, patch

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from agentforge import hitl_graph, main
from agentforge.approval import ApprovalRequest, ApprovalRequired

needs_sqlite = pytest.mark.skipif(
    importlib.util.find_spec("langgraph.checkpoint.sqlite") is None,
    reason='needs the [durable] extra: pip install -e ".[durable]"')


class _S(TypedDict, total=False):
    user_id: str
    trace_id: str
    answer: str


def _ask(state: _S) -> dict:
    decision = interrupt({"tool": "fetch", "server": "ext", "arguments": {"n": 1}})
    return {"answer": f"decided {decision}"}


def _crash(state: _S) -> dict:
    raise RuntimeError("process died here")


def _graph(saver, second=_ask):
    builder = StateGraph(_S)
    builder.add_node("first", lambda state: {})
    builder.add_node("second", second)
    builder.add_edge(START, "first")
    builder.add_edge("first", "second")
    builder.add_edge("second", END)
    return builder.compile(checkpointer=saver)


def _cfg(thread_id):
    return {"configurable": {"thread_id": thread_id}}


def _pause(graph, thread_id):
    asyncio.run(graph.ainvoke({"user_id": "u1", "trace_id": "t1"}, _cfg(thread_id),
                              durability="sync"))


def _use_saver(monkeypatch, saver):
    """Point every graph at ``saver`` and drop the compiled graphs, so they
    recompile against it — what a process restart does."""
    from agentforge import supervisor
    from agentforge import tools as tools_pkg
    from agentforge.reasoning import react_engine
    for module in (hitl_graph, tools_pkg, react_engine, supervisor):
        monkeypatch.setattr(module, "CHECKPOINTER", saver)
    monkeypatch.setattr(tools_pkg, "_ACT_GRAPH", None)
    monkeypatch.setattr(react_engine, "_GRAPH", None)
    monkeypatch.setattr(supervisor, "_GRAPH", None)


@pytest.fixture(autouse=True)
def _fresh_saver(monkeypatch):
    """An empty in-memory saver per test: turns other tests left paused in the
    shared one would otherwise change what the sweep and listing find."""
    monkeypatch.setattr(hitl_graph, "CHECKPOINTER", InMemorySaver())


@pytest.fixture
def open_savers():
    """Open SQLite savers on demand; close them all at the end."""
    savers = []

    def _open(path):
        saver = hitl_graph.build_checkpointer("sqlite", str(path))
        savers.append(saver)
        return saver
    yield _open
    for saver in savers:
        saver.conn.close()


# ----------------------------- build_checkpointer -----------------------------

class TestBuildCheckpointer:
    def test_suite_runs_on_the_in_memory_saver(self):
        # conftest forces AGENT_CHECKPOINTER=memory, so a developer's .env can
        # never make the suite write into their real checkpoints.sqlite.
        assert isinstance(hitl_graph.CHECKPOINTER, InMemorySaver)
        assert not hitl_graph.is_durable()

    def test_memory_is_the_default(self):
        assert isinstance(hitl_graph.build_checkpointer(), InMemorySaver)

    def test_unknown_kind_is_refused(self):
        with pytest.raises(ValueError, match="Unknown AGENT_CHECKPOINTER"):
            hitl_graph.build_checkpointer("postgres")

    def test_sqlite_without_the_extra_fails_loudly(self, monkeypatch):
        # A silent fallback to memory would drop the durability that was asked for.
        monkeypatch.setitem(__import__("sys").modules, "agentforge.sqlite_saver", None)
        with pytest.raises(RuntimeError, match=r"\[durable\]"):
            hitl_graph.build_checkpointer("sqlite", "unused.sqlite")

    @needs_sqlite
    def test_sqlite_creates_the_file(self, tmp_path, open_savers):
        path = tmp_path / "ck.sqlite"
        saver = open_savers(path)
        assert path.exists()
        assert saver.thread_ids() == []


# ------------------------- SQLite saver: loops + restart -------------------------

@needs_sqlite
class TestSqliteSaver:
    def test_survives_a_new_event_loop_per_turn(self, tmp_path, open_savers):
        # Every turn runs under its own asyncio.run. AsyncSqliteSaver hangs on
        # the second loop; this saver must not — wait_for turns a hang into a failure.
        graph = _graph(open_savers(tmp_path / "ck.sqlite"))
        _pause(graph, "test-1")

        async def read():
            return await asyncio.wait_for(graph.aget_state(_cfg("test-1")), timeout=10)
        snapshot = asyncio.run(read())
        assert [i.value["tool"] for t in snapshot.tasks for i in t.interrupts] == ["fetch"]

    def test_paused_turn_survives_a_restart_and_resumes(self, tmp_path, open_savers):
        path = tmp_path / "ck.sqlite"
        _pause(_graph(open_savers(path)), "test-1")

        restarted = open_savers(path)                      # new connection, new graph
        graph = _graph(restarted)
        assert restarted.thread_ids() == ["test-1"]
        pending = hitl_graph.pending_approval(graph, "test-1", "test")   # sync, like the card
        assert pending.request.tool == "fetch"
        out = asyncio.run(graph.ainvoke(Command(resume=True), _cfg("test-1"), durability="sync"))
        assert out["answer"] == "decided True"

    def test_delete_thread_removes_every_namespace(self, tmp_path, open_savers):
        saver = open_savers(tmp_path / "ck.sqlite")
        _pause(_graph(saver), "test-1")
        asyncio.run(saver.adelete_thread("test-1"))
        assert saver.thread_ids() == []
        assert saver.get_tuple(_cfg("test-1")) is None


# --------------------------------- run_graph ---------------------------------

def test_run_graph_writes_checkpoints_synchronously():
    class _FakeGraph:
        async def ainvoke(self, graph_input, config, **kwargs):
            self.kwargs = kwargs
            return {"out": "done"}

    graph = _FakeGraph()
    result = asyncio.run(hitl_graph.run_graph(
        graph, {}, thread_id="test-1", recursion_limit=5, context=None,
        pipeline="test", trace_id=None, output_key="out", progress_key="step"))
    assert result == "done"
    assert graph.kwargs["durability"] == "sync"


# ------------------------------- approval expiry -------------------------------

class TestApprovalExpiry:
    def _paused(self):
        graph = _graph(hitl_graph.CHECKPOINTER)
        thread_id = hitl_graph.new_thread_id("test")
        _pause(graph, thread_id)
        return graph, thread_id

    def test_no_ttl_means_never_expires(self, monkeypatch):
        graph, tid = self._paused()
        old = datetime.now(timezone.utc) - timedelta(days=30)
        monkeypatch.setattr(hitl_graph, "last_activity", lambda _tid: old)
        assert hitl_graph.pending_approval(graph, tid, "test").paused_at == old
        hitl_graph.discard(tid, "test")

    def test_fresh_approval_is_pending(self, monkeypatch):
        monkeypatch.setattr(hitl_graph, "AGENT_APPROVAL_TTL_SECONDS", 3600)
        graph, tid = self._paused()
        pending = hitl_graph.pending_approval(graph, tid, "test")
        assert pending.paused_at is not None
        assert (datetime.now(timezone.utc) - pending.paused_at).total_seconds() < 60
        hitl_graph.discard(tid, "test")

    def test_expired_approval_is_refused_deleted_and_logged(self, monkeypatch):
        monkeypatch.setattr(hitl_graph, "AGENT_APPROVAL_TTL_SECONDS", 3600)
        graph, tid = self._paused()
        old = datetime.now(timezone.utc) - timedelta(hours=2)
        monkeypatch.setattr(hitl_graph, "last_activity", lambda _tid: old)
        with patch.object(hitl_graph, "log_event") as log:
            with pytest.raises(hitl_graph.ApprovalExpired):
                asyncio.run(hitl_graph.apending_approval(graph, tid, "test"))   # resume path
        assert tid not in hitl_graph.thread_ids()
        event, payload = log.call_args.args
        assert event == "approval_expired"
        assert payload["ttl_seconds"] == 3600 and payload["age_seconds"] >= 7200
        assert log.call_args.kwargs["trace_id"] == "t1"

    def test_expired_is_still_a_value_error(self):
        # Callers that catch ValueError for "no longer pending" keep working.
        assert issubclass(hitl_graph.ApprovalExpired, ValueError)

    def test_last_activity_counts_nested_graph_checkpoints(self):
        # A nested graph saves its own, later checkpoints; the TTL must count
        # from those, not from the parent's last one.
        saver = InMemorySaver()
        child = _graph(saver)

        async def run_child(state):
            return {"answer": (await child.ainvoke({"user_id": "u1"}))["answer"]}

        parent = StateGraph(_S)
        parent.add_node("w", run_child)
        parent.add_edge(START, "w")
        parent.add_edge("w", END)
        parent = parent.compile(checkpointer=saver)
        _pause(parent, "sup-1")
        with patch.object(hitl_graph, "CHECKPOINTER", saver):
            parent_ts = datetime.fromisoformat(saver.get_tuple(_cfg("sup-1")).checkpoint["ts"])
            assert hitl_graph.last_activity("sup-1") > parent_ts
            assert hitl_graph.is_paused("sup-1")


# ------------------------------- startup sweep -------------------------------

class TestSweep:
    def test_keeps_paused_deletes_abandoned(self):
        _pause(_graph(hitl_graph.CHECKPOINTER), "test-paused")
        with pytest.raises(RuntimeError):     # stopped mid-node, like a crash
            asyncio.run(_graph(hitl_graph.CHECKPOINTER, second=_crash).ainvoke(
                {"user_id": "u1", "trace_id": "t2"}, _cfg("test-crashed"), durability="sync"))

        with patch.object(hitl_graph, "log_event") as log:
            counts = hitl_graph.sweep()

        assert counts == {"kept": 1, "expired": 0, "abandoned": 1}
        assert hitl_graph.thread_ids() == ["test-paused"]
        events = {call.args[0]: call for call in log.call_args_list}
        assert events["run_abandoned"].args[1] == {"thread_id": "test-crashed"}
        assert events["run_abandoned"].kwargs["trace_id"] == "t2"
        assert events["checkpoint_sweep"].args[1]["durable"] is False
        hitl_graph.discard("test-paused", "test")

    def test_deletes_expired_approvals(self, monkeypatch):
        monkeypatch.setattr(hitl_graph, "AGENT_APPROVAL_TTL_SECONDS", 60)
        _pause(_graph(hitl_graph.CHECKPOINTER), "test-old")
        later = datetime.now(timezone.utc) + timedelta(minutes=5)
        assert hitl_graph.sweep(now=later) == {"kept": 0, "expired": 1, "abandoned": 0}
        assert hitl_graph.thread_ids() == []

    def test_in_memory_thread_ids_skip_deleted_threads(self):
        # InMemorySaver.storage is a defaultdict: reading a deleted thread
        # leaves an empty entry that must not look like a saved turn.
        hitl_graph.CHECKPOINTER.get_tuple(_cfg("test-gone"))
        assert "test-gone" not in hitl_graph.thread_ids()


# --------------------------- main.pending_approvals ---------------------------

def _pending(thread_id, user_id, minutes_ago):
    return hitl_graph.PendingApproval(
        thread_id=thread_id, pipeline="act",
        request=ApprovalRequest(tool="fetch", server="ext", arguments={}),
        user_id=user_id, trace_id=None, values={},
        paused_at=datetime.now(timezone.utc) - timedelta(minutes=minutes_ago))


def test_pending_approvals_filters_by_user_newest_first(monkeypatch):
    found = {"act-a": _pending("act-a", "u1", 30), "act-b": _pending("act-b", "u2", 1),
             "act-c": _pending("act-c", "u1", 5)}

    def read(thread_id):
        if thread_id == "act-gone":
            raise hitl_graph.ApprovalExpired("expired")
        return found[thread_id]

    monkeypatch.setattr(hitl_graph, "thread_ids", lambda: [*found, "act-gone", "act-crashed"])
    monkeypatch.setattr(hitl_graph, "is_paused", lambda tid: tid != "act-crashed")
    monkeypatch.setattr(main, "pending_approval", read)
    assert [p.thread_id for p in main.pending_approvals("u1")] == ["act-c", "act-a"]


# --------------------- real ACT pipeline across a restart ---------------------

@needs_sqlite
def test_act_turn_survives_a_restart(tmp_path, monkeypatch, open_savers):
    from tests.test_approval import (
        _ACT_FOLLOWUP_JSON, TestActResume, _llm_response, _make_mcp_tool,
        _make_openai_tool_call, _make_session_mock, _raising_handler,
    )
    path = tmp_path / "ck.sqlite"
    _use_saver(monkeypatch, open_savers(path))
    session = _make_session_mock([_make_mcp_tool("fetch")], call_text="PAGE")
    tool_call = _make_openai_tool_call("fetch", json.dumps({"url": "http://8.8.8.8/p"}))
    with contextlib.ExitStack() as stack:
        tools_pkg, _ = TestActResume()._act_env(stack, session, [
            _llm_response(None, tool_calls=[tool_call]),
            _llm_response(_ACT_FOLLOWUP_JSON),
        ])
        with pytest.raises(ApprovalRequired) as excinfo:
            tools_pkg.run_llm_with_tools("u1", "fetch the page",
                                         approval_handler=_raising_handler)
        thread_id = excinfo.value.thread_id

        _use_saver(monkeypatch, open_savers(path))        # the restart
        assert hitl_graph.sweep() == {"kept": 1, "expired": 0, "abandoned": 0}
        assert main.pending_approvals("someone-else") == []
        [pending] = main.pending_approvals("u1")
        assert pending.thread_id == thread_id
        assert pending.values["user_input"] == "fetch the page"
        session.call_tool.assert_not_called()

        result = tools_pkg.resume_tool_loop(thread_id, True,
                                            approval_handler=_raising_handler)

    session.call_tool.assert_called_once_with("fetch", {"url": "http://8.8.8.8/p"})
    assert json.loads(result)["reply"] == "Here is the page summary"
    assert hitl_graph.thread_ids() == []                 # finished turn deleted


def test_react_state_keeps_the_question():
    from types import SimpleNamespace

    from agentforge.reasoning.react_engine import build_react_input
    with patch("agentforge.reasoning.react_engine.get_relevant_memories", return_value=""), \
         patch("agentforge.reasoning.react_engine.log_event", MagicMock()):
        state = build_react_input("u1", "weather in Paris?", SimpleNamespace(catalog=[]))
    assert state["user_input"] == "weather in Paris?"
