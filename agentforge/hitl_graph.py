"""Shared LangGraph runtime for the human-in-the-loop tool pipelines (Step 21.1).

Both tool-calling pipelines — ACT (``agentforge.tools``) and REACT
(``agentforge.reasoning.react_engine``) — run as LangGraph graphs whose tool
node can pause for human approval. What they share lives here, once:

- ``GraphApprovalHandler`` bridges the front-end's approval contract onto
  LangGraph ``interrupt()``.
- ``CHECKPOINTER`` saves each turn's state between nodes, keyed by a
  per-turn ``thread_id``. In memory by default; ``AGENT_CHECKPOINTER=sqlite``
  saves it to a file so paused turns survive a restart (Step 31).
- ``new_thread_id`` / ``pipeline_of`` — a thread ID names its pipeline
  (``react-…`` / ``act-…``), so the ID alone is enough to resume a turn.
- ``run_graph`` invokes or resumes a graph, turns a pause into
  ApprovalRequired (carrying the thread ID) for the front-end, and deletes a
  finished turn's thread.
- ``pending_approval`` / ``apending_approval`` read a paused turn back as a
  ``PendingApproval``, and refuse one that is no longer pending or whose
  approval expired (``AGENT_APPROVAL_TTL_SECONDS``).
- ``discard`` deletes a paused turn the user abandoned.
- ``thread_ids`` / ``is_paused`` / ``sweep`` find saved turns after a restart:
  paused ones are kept, the rest (a crash mid-node) are deleted, not re-driven.
- ``journal`` / ``journal_slot`` / ``claimed`` — the tool-call journal and
  resume claims (``agentforge.tool_journal``, Step 31 part 2): a call that ran
  before a crash is not run again, and one approval is resumed only once.

The rule both graphs are built around: LangGraph saves state only BETWEEN
nodes, and on resume re-runs the paused node from its first line. So the
non-deterministic LLM calls live in nodes that never pause, and a node that
can pause makes at most ONE gated call per run, reading it from saved state.
"""
import asyncio
import contextlib
import uuid
import weakref
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_config
from langgraph.types import interrupt

from agentforge.approval import ApprovalRequest, ApprovalRequired
from agentforge.config import (
    AGENT_APPROVAL_TTL_SECONDS,
    AGENT_CHECKPOINT_DB,
    AGENT_CHECKPOINTER,
)
from agentforge.logger import log_event
from agentforge.tool_journal import JournalSlot, MemoryJournal, call_key

# The channel LangGraph saves a pause under: a pending write on the turn's
# top-level checkpoint — also when the pause happened inside a nested graph
# (the supervisor's REACT worker), so no graph is needed to find one.
_INTERRUPT_CHANNEL = "__interrupt__"


def build_checkpointer(kind: str = "memory", path: str = "checkpoints.sqlite"):
    """The checkpointer for ``kind``: ``memory`` (in the process) or
    ``sqlite`` (saved to ``path``, survives a restart — Step 31). Raises
    RuntimeError when sqlite is asked for without the ``[durable]`` extra, and
    ValueError for an unknown kind."""
    if kind == "memory":
        return InMemorySaver()
    if kind == "sqlite":
        try:
            from agentforge.sqlite_saver import ThreadedSqliteSaver
        except ImportError as exc:
            raise RuntimeError("AGENT_CHECKPOINTER=sqlite needs the [durable] extra: "
                               'pip install -e ".[durable]"') from exc
        return ThreadedSqliteSaver.open(path)
    raise ValueError(f"Unknown AGENT_CHECKPOINTER {kind!r}; use 'memory' or 'sqlite'")


# One checkpointer for every graph. Thread IDs are random per turn, so turns
# never share state.
CHECKPOINTER = build_checkpointer(AGENT_CHECKPOINTER, AGENT_CHECKPOINT_DB)


class ApprovalExpired(ValueError):
    """A paused turn waited longer than ``AGENT_APPROVAL_TTL_SECONDS``. A
    ValueError, so callers that treat "no longer pending" as ValueError keep
    working unchanged."""


def is_durable() -> bool:
    """True when paused turns are saved outside the process (survive a restart)."""
    return not isinstance(CHECKPOINTER, InMemorySaver)


# One journal per checkpointer, created on first use. Keyed weakly by the
# checkpointer object, so a swapped CHECKPOINTER (tests) gets its own.
_JOURNALS: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def journal():
    """The tool-call journal that goes with ``CHECKPOINTER``: in the same
    SQLite file when durable, in the process otherwise."""
    found = _JOURNALS.get(CHECKPOINTER)
    if found is None:
        make = getattr(CHECKPOINTER, "make_journal", None)
        found = _JOURNALS[CHECKPOINTER] = make() if make else MemoryJournal()
    return found


def journal_slot() -> Optional[JournalSlot]:
    """The journal place for the tool call the CURRENT graph node is making —
    call it from inside a node and pass it to ``gw.call``. The key is the
    node run's namespace, so it is the same when the node runs again after a
    resume or a crash. None outside a graph (no journaling)."""
    try:
        conf = get_config()["configurable"]
    except RuntimeError:
        return None
    return JournalSlot(journal(), conf["thread_id"], call_key(conf["checkpoint_ns"]))


def _delete_run(thread_id: str) -> None:
    """Delete a run's checkpoints AND its journal rows and claims."""
    CHECKPOINTER.delete_thread(thread_id)
    journal().delete_thread(thread_id)


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
    paused_at: Optional[datetime] = None   # last saved checkpoint (Step 31)
    interrupt_id: Optional[str] = None     # which pause this is; the claim key
    already_ran: bool = False              # the call ran before a crash (journaled)


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


def _thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _saved_trace_id(saved) -> Optional[str]:
    """The turn's trace ID from a raw checkpoint tuple (None if gone)."""
    if saved is None:
        return None
    return saved.checkpoint.get("channel_values", {}).get("trace_id")


def _is_paused(saved) -> bool:
    return saved is not None and any(
        write[1] == _INTERRUPT_CHANNEL for write in (saved.pending_writes or []))


def thread_ids() -> list:
    """Every turn the checkpointer holds: paused ones, and any a crash left."""
    if hasattr(CHECKPOINTER, "thread_ids"):
        return CHECKPOINTER.thread_ids()
    # InMemorySaver keys storage by thread ID. It is a defaultdict, so a read
    # of a deleted thread leaves an empty entry behind — skip those.
    return [tid for tid, by_ns in CHECKPOINTER.storage.items() if any(by_ns.values())]


def is_paused(thread_id: str) -> bool:
    """Whether the turn is waiting at an approval. Reads the raw checkpoint,
    so it needs no graph — the startup sweep runs before any graph is built."""
    return _is_paused(CHECKPOINTER.get_tuple(_thread_config(thread_id)))


def last_activity(thread_id: str) -> Optional[datetime]:
    """When the turn last saved a checkpoint, in any namespace — a nested
    graph (the supervisor's REACT worker) saves its own, later than its
    parent's. The approval TTL counts from here. None if the turn is gone."""
    stamps = [saved.checkpoint["ts"] for saved in CHECKPOINTER.list(_thread_config(thread_id))]
    return max(datetime.fromisoformat(ts) for ts in stamps) if stamps else None


def _age_seconds(paused_at: datetime, now: Optional[datetime] = None) -> float:
    return ((now or datetime.now(timezone.utc)) - paused_at).total_seconds()


def _expired(paused_at: Optional[datetime], now: Optional[datetime] = None) -> bool:
    return (AGENT_APPROVAL_TTL_SECONDS > 0 and paused_at is not None
            and _age_seconds(paused_at, now) > AGENT_APPROVAL_TTL_SECONDS)


def _expire(thread_id: str, paused_at: datetime, trace_id: Optional[str],
            now: Optional[datetime] = None) -> None:
    _delete_run(thread_id)
    log_event("approval_expired",
              {"thread_id": thread_id, "age_seconds": round(_age_seconds(paused_at, now)),
               "ttl_seconds": AGENT_APPROVAL_TTL_SECONDS},
              trace_id=trace_id)


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
    interrupt_id = waiting[0].id
    values = dict(snapshot.values)
    paused_at = last_activity(thread_id)
    if _expired(paused_at):
        # Checked on every read, so an approval that expires while its card is
        # on screen is refused at the click, not only at the next restart.
        _expire(thread_id, paused_at, values.get("trace_id"))
        raise ApprovalExpired("This approval has expired and cannot be resumed.")
    return PendingApproval(
        thread_id=thread_id, pipeline=pipeline,
        request=ApprovalRequest(tool=payload["tool"], server=payload["server"],
                                arguments=payload["arguments"]),
        user_id=values.get("user_id", ""), trace_id=values.get("trace_id"),
        values=values, paused_at=paused_at, interrupt_id=interrupt_id,
        # The paused call's journal key IS its interrupt ID (tool_journal).
        already_ran=journal().lookup(thread_id, interrupt_id) is not None)


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
    card is pending), so it does not stay saved (in memory, or on disk with
    AGENT_CHECKPOINTER=sqlite) until it expires.
    Safe to call for a thread that no longer exists. The event is logged on
    the turn's own trace (read from its saved state before deletion), so it
    is not orphaned from the turn it ends (issue #7)."""
    trace_id = _saved_trace_id(CHECKPOINTER.get_tuple(_thread_config(thread_id)))
    _delete_run(thread_id)
    log_event("approval_discarded", {"thread_id": thread_id, "reason": reason},
              trace_id=trace_id)


def sweep(now: Optional[datetime] = None) -> dict:
    """Clean up the saved turns a restart finds (Step 31). Call once at
    startup, before any turn runs, and only with one process per database.

    A turn paused at an approval is kept, unless its approval expired. Any
    other saved turn stopped in the middle of a node: the process died during
    it (a finished or failed turn deletes its own thread in ``run_graph``). It
    is deleted, NOT re-driven: the browser or terminal that asked is gone, so
    finishing it would spend tokens — and maybe call tools — for nobody.
    """
    counts = {"kept": 0, "expired": 0, "abandoned": 0}
    # Claims belong to resumes in flight; at startup none are, so any left
    # are from a process that died mid-resume and would block the card.
    journal().clear_claims()
    for thread_id in thread_ids():
        saved = CHECKPOINTER.get_tuple(_thread_config(thread_id))
        trace_id = _saved_trace_id(saved)
        if _is_paused(saved):
            paused_at = last_activity(thread_id)
            if _expired(paused_at, now):
                _expire(thread_id, paused_at, trace_id, now)
                counts["expired"] += 1
            else:
                counts["kept"] += 1
            continue
        _delete_run(thread_id)
        log_event("run_abandoned", {"thread_id": thread_id}, trace_id=trace_id)
        counts["abandoned"] += 1
    # Journal rows whose run is gone: a crash between deleting the checkpoints
    # and the journal rows of a finished turn.
    orphans = journal().thread_ids() - set(thread_ids())
    for thread_id in orphans:
        journal().delete_thread(thread_id)
    counts["journal_orphans"] = len(orphans)
    log_event("checkpoint_sweep", {**counts, "durable": is_durable()})
    return counts


@contextlib.contextmanager
def claimed(pending: PendingApproval):
    """Hold the resume claim on ``pending`` for the length of one resume.
    A second resume of the same approval while this one runs (two tabs, one
    card) is refused with ValueError, so it cannot run the tool again.
    A ``PendingApproval`` built without an interrupt ID (one not read from a
    checkpoint) has nothing to claim, and resumes as before."""
    if pending.interrupt_id is None:
        yield
        return
    if not journal().claim(pending.thread_id, pending.interrupt_id):
        log_event(f"{pending.pipeline}_resume_rejected",
                  {"thread_id": pending.thread_id, "reason": "already being resumed"},
                  trace_id=pending.trace_id)
        raise ValueError("This approval is already being resumed.")
    try:
        yield
    finally:
        journal().release(pending.thread_id, pending.interrupt_id)


async def run_graph(graph, graph_input, *, thread_id: str, recursion_limit: int,
                    context: Any, pipeline: str,
                    trace_id: Optional[str], output_key: str,
                    progress_key: str) -> Any:
    """Invoke (or resume) ``graph``; translate a pause into ApprovalRequired.

    A paused graph surfaces as ApprovalRequired carrying ``thread_id`` — the
    only handle a resume needs; the turn state lives in the checkpointer. A
    finished turn's thread is deleted, so the checkpointer does not grow with
    every turn. ``progress_key`` names the state field logged at a pause
    (REACT's step counter, ACT's tool-call index).

    ``durability="sync"`` (Step 31): each checkpoint is written before the next
    node starts, so a crash cannot lose the last completed step. Nested graphs
    (the supervisor's REACT worker) inherit it through the config.
    """
    paused = False
    try:
        result = await graph.ainvoke(graph_input, _config(thread_id, recursion_limit),
                                     context=context, durability="sync")
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
            await asyncio.to_thread(journal().delete_thread, thread_id)
