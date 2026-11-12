"""Tool-call journal and resume claims (Step 31, part 2 — issue #80).

The problem the journal solves. LangGraph saves state only BETWEEN nodes. If
the process dies after a tool call returns but before its node's checkpoint is
saved, the saved run is still "paused at the approval". The card comes back
after the restart, and approving it runs the tool a second time.

So the gateway records each tool result here the moment the call returns,
outside the graph state (the crash happens exactly before that state is
saved). When the node runs again, the gateway finds the record and uses it
instead of calling the tool. A window remains — the tool returned, the record
is not written yet — so the guarantee is AT-LEAST-ONCE, not exactly-once.
Exactly-once would need the tool itself to accept an idempotency key.

The key. LangGraph gives every node run a namespace (``checkpoint_ns``) that
stays the same when that node runs again from the same checkpoint — at a
resume, and after a crash. Its pending interrupt's ID is a hash of that
namespace, and ``call_key`` uses LangGraph's own hash. So one value is the
journal key, the claim key, and how the approval card knows a call already
ran. Each tool node makes one call per run, so one key is one call.

Claims stop two resumes of one approval (two tabs, one card) from both
running the tool: the first resume claims the interrupt, a second is refused.

``MemoryJournal`` lives in the process, like the in-memory checkpointer; the
SQLite one (``sqlite_saver.SqliteJournal``) shares the checkpoint file.
"""
import threading
from dataclasses import dataclass
from typing import Optional

from langgraph.types import Interrupt


@dataclass(frozen=True)
class JournalEntry:
    """A tool's raw result, before sanitize / guardrail / nonce wrap — those
    run again on every replay, with the current turn's nonce."""
    text: str
    is_error: bool


def call_key(checkpoint_ns: str) -> str:
    """The journal key for a node run: LangGraph's interrupt ID for the same
    namespace, so a paused call's key is its pending interrupt's ID."""
    return Interrupt.from_ns(None, checkpoint_ns).id


class MemoryJournal:
    """In-process journal and claims. Lost with the process — and so is every
    run the in-memory checkpointer held, so nothing is left to replay."""

    def __init__(self):
        self._lock = threading.Lock()
        self._calls: dict = {}     # (thread_id, key) -> JournalEntry
        self._claims: set = set()  # (thread_id, interrupt_id)

    def lookup(self, thread_id: str, key: str) -> Optional[JournalEntry]:
        with self._lock:
            return self._calls.get((thread_id, key))

    def record(self, thread_id: str, key: str, entry: JournalEntry) -> None:
        with self._lock:
            self._calls[(thread_id, key)] = entry

    def claim(self, thread_id: str, interrupt_id: str) -> bool:
        """True if this caller now holds the claim; False if someone else does."""
        with self._lock:
            if (thread_id, interrupt_id) in self._claims:
                return False
            self._claims.add((thread_id, interrupt_id))
            return True

    def release(self, thread_id: str, interrupt_id: str) -> None:
        with self._lock:
            self._claims.discard((thread_id, interrupt_id))

    def delete_thread(self, thread_id: str) -> None:
        with self._lock:
            self._calls = {k: v for k, v in self._calls.items() if k[0] != thread_id}
            self._claims = {c for c in self._claims if c[0] != thread_id}

    def thread_ids(self) -> set:
        with self._lock:
            return {k[0] for k in self._calls}

    def clear_claims(self) -> None:
        with self._lock:
            self._claims.clear()


class JournalSlot:
    """One tool call's place in the journal: what the gateway is handed, so
    it never needs to know about threads or namespaces."""

    def __init__(self, journal, thread_id: str, key: str):
        self.journal = journal
        self.thread_id = thread_id
        self.key = key

    def lookup(self) -> Optional[JournalEntry]:
        return self.journal.lookup(self.thread_id, self.key)

    def record(self, text: str, is_error: bool) -> None:
        self.journal.record(self.thread_id, self.key, JournalEntry(text, is_error))
