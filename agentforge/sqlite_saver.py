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
"""
import asyncio
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver


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
