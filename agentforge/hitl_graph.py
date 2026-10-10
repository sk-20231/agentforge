"""Shared LangGraph runtime for the human-in-the-loop tool pipelines (Step 21.1).

Both tool-calling pipelines — ACT (``agentforge.tools``) and REACT
(``agentforge.reasoning.react_engine``) — run as LangGraph graphs whose tool
node can pause for human approval. What they share lives here, once:

- ``GraphApprovalHandler`` bridges the front-end's approval contract onto
  LangGraph ``interrupt()``.
- ``CHECKPOINTER`` saves each turn's state between nodes, keyed by a
  per-turn ``thread_id``.
- ``run_graph`` invokes or resumes a graph, turns a pause into
  ApprovalRequired for the front-end, and deletes a finished turn's thread.
- ``pending_state`` loads a paused turn for resume and refuses one that is
  no longer pending.

The rule both graphs are built around: LangGraph saves state only BETWEEN
nodes, and on resume re-runs the paused node from its first line. So the
non-deterministic LLM calls live in nodes that never pause, and a node that
can pause makes at most ONE gated call per run, reading it from saved state.
"""
from typing import Any, Optional

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt

from agentforge.approval import ApprovalRequest, ApprovalRequired
from agentforge.logger import log_event

# One in-process checkpointer for both graphs. Thread IDs are random per turn,
# so turns never share state. A persistent checkpointer (SQLite / Postgres)
# drops in here when multi-user needs state to outlive the process.
CHECKPOINTER = InMemorySaver()


class GraphApprovalHandler:
    """Bridges the front-end's approval contract onto LangGraph ``interrupt()``.

    The gateway calls this like any handler. Fresh run: the front-end handler
    answers synchronously (CLI) or raises ApprovalRequired (Streamlit, which
    cannot block) — the latter becomes ``interrupt()``, which pauses the graph
    and checkpoints it. Resume (``resuming=True``): the first gated call is
    the re-run of the stored call, so ``interrupt()`` RETURNS the human's
    decision instead of pausing. Consumed once: any later gated call in the
    resumed turn goes back to the front-end handler and can pause again.
    """

    def __init__(self, fallback=None, resuming: bool = False):
        self.fallback = fallback
        self.resuming = resuming

    def __call__(self, request: ApprovalRequest):
        payload = {"tool": request.tool, "server": request.server,
                   "arguments": request.arguments}
        if self.resuming:
            self.resuming = False
            return interrupt(payload)
        if self.fallback is None:
            return False  # no way to ask a human -> deny
        try:
            return self.fallback(request)
        except ApprovalRequired:
            return interrupt(payload)


def _config(thread_id: str, recursion_limit: int) -> dict:
    return {"configurable": {"thread_id": thread_id},
            "recursion_limit": recursion_limit}


async def pending_state(graph, thread_id: str, pipeline: str,
                        trace_id: Optional[str]) -> dict:
    """The saved state of a paused turn. Raises ValueError if the turn is no
    longer pending — already resumed (e.g. a double click), finished, or
    never paused — rather than starting a turn with no state, so a settled
    call is never dispatched again."""
    snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    # Ask the precise question — "is an interrupt waiting?" — not "is a node
    # scheduled next?". A second interrupt() inside one node run leaves
    # ``snapshot.next`` empty while the interrupt is still pending.
    if not any(task.interrupts for task in snapshot.tasks):
        log_event(f"{pipeline}_resume_rejected",
                  {"thread_id": thread_id, "reason": "no pending interrupt"},
                  trace_id=trace_id)
        raise ValueError("This approval is no longer pending and cannot be resumed.")
    return snapshot.values


async def run_graph(graph, graph_input, *, thread_id: str, recursion_limit: int,
                    context: Any, pipeline: str, user_id: str,
                    trace_id: Optional[str], output_key: str,
                    progress_key: str) -> Any:
    """Invoke (or resume) ``graph``; translate a pause into ApprovalRequired.

    Until the front-ends move to ``thread_id`` (Step 21.1 task F), a paused
    graph still surfaces as ApprovalRequired — its continuation carries only
    the handle; the turn state lives in the checkpointer. A finished turn's
    thread is deleted, so the in-memory checkpointer does not grow with every
    turn. ``progress_key`` names the state field logged at a pause (REACT's
    step counter, ACT's tool-call index).
    """
    paused = False
    try:
        result = await graph.ainvoke(graph_input, _config(thread_id, recursion_limit),
                                     context=context)
        pending = result.get("__interrupt__")
        if pending:
            paused = True
            payload = pending[0].value
            request = ApprovalRequest(tool=payload["tool"], server=payload["server"],
                                      arguments=payload["arguments"])
            # Tool NAME only — arguments can hold user data (Step 17e gap A).
            log_event(f"{pipeline}_interrupt",
                      {"thread_id": thread_id, progress_key: result.get(progress_key),
                       "tool": request.tool},
                      trace_id=trace_id)
            exc = ApprovalRequired(request)
            exc.continuation = {"pipeline": pipeline, "thread_id": thread_id,
                                "user_id": user_id, "trace_id": trace_id}
            raise exc
        return result[output_key]
    finally:
        if not paused:
            await CHECKPOINTER.adelete_thread(thread_id)
