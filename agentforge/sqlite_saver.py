"""SQLite checkpointer for durable runs (Step 31). Needs the ``[durable]`` extra.

Imported only when ``AGENT_CHECKPOINTER=sqlite`` (see ``hitl_graph``), so a
plain clone never needs ``langgraph-checkpoint-sqlite``.

Why a subclass of the SYNC ``SqliteSaver`` and not ``AsyncSqliteSaver``: every
turn runs under its own ``asyncio.run`` (``approval.run_interruptible``), so a
turn gets a new event loop. ``AsyncSqliteSaver`` binds its aiosqlite connection
to the loop it was opened on; on the next turn's loop it hangs instead of
failing (measured in the Step 31 spike). The sync saver has no loop binding and
guards its one connection with a lock, so its async methods here just run the
sync ones in a worker thread. The Streamlit card keeps using the sync methods
directly.

Single process per database file: the startup sweep treats any run that is
not paused as abandoned, so two processes sharing one file would sweep each
other's in-flight runs.

``SqliteJournal`` keeps the tool-call journal and resume claims
(``agentforge.tool_journal``) in the same file, on the saver's connection and
lock, so a journal write never waits on a second connection's lock.
"""
import asyncio
import sqlite3
from datetime import datetime, timezone
from typing import Optional

from langgraph.checkpoint.sqlite import SqliteSaver

from agentforge.tool_journal import JournalEntry


class ThreadedSqliteSaver(SqliteSaver):
    """``SqliteSaver`` whose async methods run the sync ones in a thread."""

    @classmethod
    def open(cls, path: str) -> "ThreadedSqliteSaver":
        """A saver on ``path`` (created if missing), with its tables set up."""
        saver = cls(sqlite3.connect(path, check_same_thread=False))
        saver.setup()
        return saver

    def thread_ids(self) -> list:
        """Every thread with at least one saved checkpoint."""
        with self.cursor(transaction=False) as cur:
            cur.execute("SELECT DISTINCT thread_id FROM checkpoints")
            return [row[0] for row in cur.fetchall()]

    def make_journal(self) -> "SqliteJournal":
        """The tool-call journal for this file (``hitl_graph.journal``)."""
        return SqliteJournal(self)

    async def aget_tuple(self, config):
        return await asyncio.to_thread(self.get_tuple, config)

    async def alist(self, config, *, filter=None, before=None, limit=None):
        items = await asyncio.to_thread(
            lambda: list(self.list(config, filter=filter, before=before, limit=limit)))
        for item in items:
            yield item

    async def aput(self, config, checkpoint, metadata, new_versions):
        return await asyncio.to_thread(self.put, config, checkpoint, metadata, new_versions)

    async def aput_writes(self, config, writes, task_id, task_path=""):
        return await asyncio.to_thread(self.put_writes, config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id):
        return await asyncio.to_thread(self.delete_thread, thread_id)

    async def aget_delta_channel_history(self, *args, **kwargs):
        return await asyncio.to_thread(self.get_delta_channel_history, *args, **kwargs)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SqliteJournal:
    """Tool-call journal and resume claims in the checkpoint file. Same
    methods as ``tool_journal.MemoryJournal``. Every write commits at once:
    a record that waits for a later commit would be lost by the very crash it
    is there for."""

    def __init__(self, saver: ThreadedSqliteSaver):
        self.saver = saver
        with saver.cursor() as cur:
            cur.execute("""CREATE TABLE IF NOT EXISTS tool_journal (
                thread_id TEXT NOT NULL, call_key TEXT NOT NULL,
                is_error INTEGER NOT NULL, text TEXT NOT NULL, created_at TEXT NOT NULL,
                PRIMARY KEY (thread_id, call_key))""")
            cur.execute("""CREATE TABLE IF NOT EXISTS resume_claims (
                thread_id TEXT NOT NULL, interrupt_id TEXT NOT NULL, claimed_at TEXT NOT NULL,
                PRIMARY KEY (thread_id, interrupt_id))""")

    def lookup(self, thread_id: str, key: str) -> Optional[JournalEntry]:
        with self.saver.cursor(transaction=False) as cur:
            cur.execute("SELECT text, is_error FROM tool_journal WHERE thread_id = ? AND call_key = ?",
                        (thread_id, key))
            row = cur.fetchone()
        return JournalEntry(row[0], bool(row[1])) if row else None

    def record(self, thread_id: str, key: str, entry: JournalEntry) -> None:
        with self.saver.cursor() as cur:
            cur.execute("INSERT OR REPLACE INTO tool_journal VALUES (?, ?, ?, ?, ?)",
                        (thread_id, key, int(entry.is_error), entry.text, _now()))

    def claim(self, thread_id: str, interrupt_id: str) -> bool:
        # One atomic insert: of two racing resumes, exactly one adds the row.
        with self.saver.cursor() as cur:
            cur.execute("INSERT OR IGNORE INTO resume_claims VALUES (?, ?, ?)",
                        (thread_id, interrupt_id, _now()))
            return cur.rowcount == 1

    def release(self, thread_id: str, interrupt_id: str) -> None:
        with self.saver.cursor() as cur:
            cur.execute("DELETE FROM resume_claims WHERE thread_id = ? AND interrupt_id = ?",
                        (thread_id, interrupt_id))

    def delete_thread(self, thread_id: str) -> None:
        with self.saver.cursor() as cur:
            cur.execute("DELETE FROM tool_journal WHERE thread_id = ?", (thread_id,))
            cur.execute("DELETE FROM resume_claims WHERE thread_id = ?", (thread_id,))

    def thread_ids(self) -> set:
        with self.saver.cursor(transaction=False) as cur:
            cur.execute("SELECT DISTINCT thread_id FROM tool_journal")
            return {row[0] for row in cur.fetchall()}

    def clear_claims(self) -> None:
        with self.saver.cursor() as cur:
            cur.execute("DELETE FROM resume_claims")
