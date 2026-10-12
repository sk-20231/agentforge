"""Shared LangGraph runtime for the human-in-the-loop tool pipelines (Step 21.1).

Both tool-calling pipelines — ACT (``agentforge.tools``) and REACT
(``agentforge.reasoning.react_engine``) — run as LangGraph graphs whose tool
node can pause for human approval. What they share lives here, once:

- ``GraphApprovalHandler`` bridges the front-end's approval contract onto
  LangGraph ``interrupt()``.
- ``CHECKPOINTER`` saves each turn's state between nodes, keyed by a
  per-turn ``thread_id``.
- ``new_thread_id`` / ``pipeline_of`` — a thread ID names its pipeline
  (``react-…`` / ``act-…``), so the ID alone is enough to resume a turn.
- ``run_graph`` invokes or resumes a graph, turns a pause into
  ApprovalRequired (carrying the thread ID) for the front-end, and deletes a
  finished turn's thread.
- ``pending_approval`` / ``apending_approval`` read a paused turn back as a
  ``PendingApproval``, and refuse one that is no longer pending.
- ``discard`` deletes a paused turn the user abandoned.

The rule both graphs are built around: LangGraph saves state only BETWEEN
nodes, and on resume re-runs the paused node from its first line. So the
non-deterministic LLM calls live in nodes that never pause, and a node that
can pause makes at most ONE gated call per run, reading it from saved state.
"""
import uuid
from dataclasses import dataclass
from typing import Any, Optional

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt

from agentforge.approval import ApprovalRequest, ApprovalRequired
from agentforge.logger import log_event

# One in-process checkpointer for both graphs. Thread IDs are random per turn,
# so turns never share state. A persistent checkpointer (SQLite / Postgres)
# drops in here when multi-user needs state to outlive the process.
CHECKPOINTER = InMemorySaver()


def new_thread_id(pipeline: str) -> str:
    """A fresh per-turn thread ID that names its pipeline, e.g. ``act-3f9a…``."""
    return f"{pipeline}-{uuid.uuid4().hex}"


def pipeline_of(thread_id: str) -> str:
    """The pipeline a thread ID belongs to (the part before the first ``-``).
    Raises ValueError for anything that is not a pipeline-prefixed ID."""
    if not isinstance(thread_id, str) or "-" not in thread_id:
        raise ValueError(f"Not a pipeline thread ID: {thread_id!r}")
    return thread_id.split("-", 1)[0]


@dataclass(frozen=True)
class PendingApproval:
    """A paused turn, read back from the checkpointer: what the human is
    being asked about, and what a resume needs. ``values`` is the saved
    graph state."""
    thread_id: str
    pipeline: str
    request: ApprovalRequest
    user_id: str
    trace_id: Optional[str]
    values: dict


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


def _pending_from_snapshot(snapshot, thread_id: str, pipeline: str) -> PendingApproval:
    # Ask the precise question — "is an interrupt waiting?" — not "is a node
    # scheduled next?". A second interrupt() inside one node run leaves
    # ``snapshot.next`` empty while the interrupt is still pending.
    waiting = [i for task in snapshot.tasks for i in task.interrupts]
    if not waiting:
        log_event(f"{pipeline}_resume_rejected",
                  {"thread_id": thread_id, "reason": "no pending interrupt"},
                  trace_id=snapshot.values.get("trace_id"))
        raise ValueError("This approval is no longer pending and cannot be resumed.")
    payload = waiting[0].value
    values = dict(snapshot.values)
    return PendingApproval(
        thread_id=thread_id, pipeline=pipeline,
        request=ApprovalRequest(tool=payload["tool"], server=payload["server"],
                                arguments=payload["arguments"]),
        user_id=values.get("user_id", ""), trace_id=values.get("trace_id"),
        values=values)


def pending_approval(graph, thread_id: str, pipeline: str) -> PendingApproval:
    """The paused turn ``thread_id`` (sync — for the front-end's card).
    Raises ValueError if the turn is no longer pending — already resumed
    (e.g. a double click), finished, discarded, or never paused — so a
    settled call is never dispatched again."""
    snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
    return _pending_from_snapshot(snapshot, thread_id, pipeline)


async def apending_approval(graph, thread_id: str, pipeline: str) -> PendingApproval:
    """Async twin of ``pending_approval``, for the resume path."""
    snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    return _pending_from_snapshot(snapshot, thread_id, pipeline)


def discard(thread_id: str, reason: str) -> None:
    """Delete a paused turn the user abandoned (e.g. "Clear chat" while a
    card is pending), so it does not stay in memory until the app restarts.
    Safe to call for a thread that no longer exists. The event is logged on
    the turn's own trace (read from its saved state before deletion), so it
    is not orphaned from the turn it ends (issue #7)."""
    saved = CHECKPOINTER.get_tuple({"configurable": {"thread_id": thread_id}})
    trace_id = (saved.checkpoint.get("channel_values", {}).get("trace_id")
                if saved is not None else None)
    CHECKPOINTER.delete_thread(thread_id)
    log_event("approval_discarded", {"thread_id": thread_id, "reason": reason},
              trace_id=trace_id)


async def run_graph(graph, graph_input, *, thread_id: str, recursion_limit: int,
                    context: Any, pipeline: str,
                    trace_id: Optional[str], output_key: str,
                    progress_key: str) -> Any:
    """Invoke (or resume) ``graph``; translate a pause into ApprovalRequired.

    A paused graph surfaces as ApprovalRequired carrying ``thread_id`` — the
    only handle a resume needs; the turn state lives in the checkpointer. A
    finished turn's
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
            raise ApprovalRequired(request, thread_id=thread_id)
        return result[output_key]
    finally:
        if not paused:
            await CHECKPOINTER.adelete_thread(thread_id)
