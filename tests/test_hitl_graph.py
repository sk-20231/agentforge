"""Tests for the shared LangGraph HITL runtime (Step 21.1).

The pipelines' end-to-end pause/resume behaviour is tested in test_approval.py
through the real gateway. These cover ``pending_state`` directly, on a tiny
graph, for the case the pipelines are designed never to reach.
"""
import asyncio
import uuid
from typing import TypedDict

import pytest
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from agentforge import hitl_graph


class _S(TypedDict):
    answers: list


async def _two_pauses(state: _S) -> dict:
    first = interrupt("first")
    second = interrupt("second")
    return {"answers": [first, second]}


def _graph():
    builder = StateGraph(_S)
    builder.add_node("n", _two_pauses)
    builder.add_edge(START, "n")
    builder.add_edge("n", END)
    return builder.compile(checkpointer=hitl_graph.CHECKPOINTER)


def test_second_interrupt_in_one_node_still_counts_as_pending():
    # After a second interrupt() in the same node run, LangGraph leaves
    # snapshot.next empty while the interrupt is still waiting. The check
    # must look at the interrupts, or it refuses a resume it should allow.
    graph = _graph()
    config = {"configurable": {"thread_id": uuid.uuid4().hex}}
    tid = config["configurable"]["thread_id"]

    async def go():
        await graph.ainvoke({"answers": []}, config)              # pause 1
        await graph.ainvoke(Command(resume="a"), config)          # pause 2
        values = await hitl_graph.pending_state(graph, tid, "test", None)
        result = await graph.ainvoke(Command(resume="b"), config)
        return values, result

    values, result = asyncio.run(go())
    assert values == {"answers": []}
    assert result["answers"] == ["a", "b"]
    asyncio.run(hitl_graph.CHECKPOINTER.adelete_thread(tid))


def test_finished_or_unknown_thread_is_not_pending():
    graph = _graph()
    with pytest.raises(ValueError):
        asyncio.run(hitl_graph.pending_state(graph, uuid.uuid4().hex, "test", None))
