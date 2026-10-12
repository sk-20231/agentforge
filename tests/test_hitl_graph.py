"""Tests for the shared LangGraph HITL runtime (Step 21.1).

The pipelines' end-to-end pause/resume behaviour is tested in test_approval.py
through the real gateway. These cover the runtime helpers directly, on a tiny
graph — including the case the pipelines are designed never to reach.
"""
import asyncio
import uuid
from typing import TypedDict
from unittest.mock import patch

import pytest
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from agentforge import hitl_graph


class _S(TypedDict):
    user_id: str
    trace_id: str
    answers: list


def _ask(name):
    return {"tool": name, "server": "ext", "arguments": {"n": name}}


async def _two_pauses(state: _S) -> dict:
    first = interrupt(_ask("first"))
    second = interrupt(_ask("second"))
    return {"answers": [first, second]}


def _graph():
    builder = StateGraph(_S)
    builder.add_node("n", _two_pauses)
    builder.add_edge(START, "n")
    builder.add_edge("n", END)
    return builder.compile(checkpointer=hitl_graph.CHECKPOINTER)


def _start():
    graph = _graph()
    thread_id = hitl_graph.new_thread_id("test")
    config = {"configurable": {"thread_id": thread_id}}
    asyncio.run(graph.ainvoke({"user_id": "u1", "trace_id": "t1", "answers": []}, config))
    return graph, thread_id, config


def test_thread_id_names_its_pipeline():
    tid = hitl_graph.new_thread_id("act")
    assert tid.startswith("act-")
    assert hitl_graph.pipeline_of(tid) == "act"
    assert hitl_graph.new_thread_id("act") != tid


@pytest.mark.parametrize("bad", ["nodash", "", None, 42])
def test_pipeline_of_rejects_ids_without_a_prefix(bad):
    with pytest.raises(ValueError):
        hitl_graph.pipeline_of(bad)


def test_pending_approval_reads_request_and_ids_back():
    graph, tid, _ = _start()
    pending = hitl_graph.pending_approval(graph, tid, "test")
    assert pending.request.tool == "first"
    assert pending.request.arguments == {"n": "first"}
    assert (pending.user_id, pending.trace_id, pending.pipeline) == ("u1", "t1", "test")
    hitl_graph.discard(tid, "test")


def test_second_interrupt_in_one_node_still_counts_as_pending():
    # After a second interrupt() in the same node run, LangGraph leaves
    # snapshot.next empty while the interrupt is still waiting. The check
    # must look at the interrupts, or it refuses a resume it should allow.
    graph, tid, config = _start()
    asyncio.run(graph.ainvoke(Command(resume="a"), config))            # pause 2
    pending = asyncio.run(hitl_graph.apending_approval(graph, tid, "test"))
    assert pending.request.tool == "second"
    result = asyncio.run(graph.ainvoke(Command(resume="b"), config))
    assert result["answers"] == ["a", "b"]
    hitl_graph.discard(tid, "test")


def test_discard_removes_a_paused_turn():
    graph, tid, _ = _start()
    hitl_graph.discard(tid, "test")
    with pytest.raises(ValueError):
        hitl_graph.pending_approval(graph, tid, "test")


def test_discard_is_logged_on_the_turns_trace():
    # The discard event must link to the turn it ends, like every other
    # record of that turn (issue #7) — the trace ID comes from saved state.
    _, tid, _ = _start()
    with patch.object(hitl_graph, "log_event") as log:
        hitl_graph.discard(tid, "clear_chat")
    event, payload = log.call_args.args
    assert event == "approval_discarded"
    assert payload == {"thread_id": tid, "reason": "clear_chat"}
    assert log.call_args.kwargs["trace_id"] == "t1"


def test_discard_of_an_unknown_thread_is_safe():
    with patch.object(hitl_graph, "log_event") as log:
        hitl_graph.discard(f"test-{uuid.uuid4().hex}", "clear_chat")
    assert log.call_args.kwargs["trace_id"] is None


def test_finished_or_unknown_thread_is_not_pending():
    graph = _graph()
    with pytest.raises(ValueError):
        hitl_graph.pending_approval(graph, f"test-{uuid.uuid4().hex}", "test")
