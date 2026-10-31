"""SQLite persistence layer: tasks, message idempotency records, decision cache.

All access is serialised through one re-entrant lock so that "check state, then
change state" sequences (receipt vs. cancel race) are atomic. Run ONE server
process; the lock is in-process.
"""
import json
import os
import sqlite3
import threading
from contextlib import contextmanager


class Store:
    def __init__(self, path):
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.lock = threading.RLock()
        with self.lock:
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    id TEXT NOT NULL UNIQUE,
                    principal TEXT NOT NULL,
                    context_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    ts TEXT NOT NULL,
                    doc TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS tasks_principal ON tasks(principal);
                CREATE TABLE IF NOT EXISTS messages (
                    principal TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    hash TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    PRIMARY KEY (principal, message_id)
                );
                CREATE TABLE IF NOT EXISTS decision_cache (
                    key TEXT PRIMARY KEY,
                    doc TEXT NOT NULL
                );
                """
            )

    @contextmanager
    def tx(self):
        with self.lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                yield
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise
            else:
                self.conn.execute("COMMIT")

    # ---- tasks -------------------------------------------------------
    def get_task(self, principal, task_id):
        with self.lock:
            row = self.conn.execute(
                "SELECT doc FROM tasks WHERE id=? AND principal=?", (task_id, principal)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def put_task(self, principal, task):
        st = task["status"]
        doc = json.dumps(task, separators=(",", ":"), ensure_ascii=False)
        with self.lock:
            cur = self.conn.execute(
                "UPDATE tasks SET state=?, ts=?, doc=? WHERE id=? AND principal=?",
                (st["state"], st.get("timestamp", ""), doc, task["id"], principal),
            )
            if cur.rowcount == 0:
                self.conn.execute(
                    "INSERT INTO tasks (id, principal, context_id, state, ts, doc) VALUES (?,?,?,?,?,?)",
                    (task["id"], principal, task["contextId"], st["state"], st.get("timestamp", ""), doc),
                )

    def list_tasks(self, principal):
        """All of this principal's tasks, most recently updated first."""
        with self.lock:
            rows = self.conn.execute(
                "SELECT doc FROM tasks WHERE principal=? ORDER BY ts DESC, seq DESC", (principal,)
            ).fetchall()
        return [json.loads(r[0]) for r in rows]

    def submitted_tasks(self):
        with self.lock:
            rows = self.conn.execute(
                "SELECT principal, doc FROM tasks WHERE state IN ('TASK_STATE_SUBMITTED','TASK_STATE_WORKING')"
            ).fetchall()
        return [(r[0], json.loads(r[1])) for r in rows]

    # ---- message idempotency ------------------------------------------
    def get_message(self, principal, message_id):
        with self.lock:
            row = self.conn.execute(
                "SELECT hash, task_id FROM messages WHERE principal=? AND message_id=?",
                (principal, message_id),
            ).fetchone()
        return {"hash": row[0], "task_id": row[1]} if row else None

    def put_message(self, principal, message_id, msg_hash, task_id):
        with self.lock:
            self.conn.execute(
                "INSERT OR IGNORE INTO messages (principal, message_id, hash, task_id) VALUES (?,?,?,?)",
                (principal, message_id, msg_hash, task_id),
            )

    # ---- decision cache --------------------------------------------------
    def cache_get(self, key):
        with self.lock:
            row = self.conn.execute("SELECT doc FROM decision_cache WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def cache_put(self, key, doc):
        with self.lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO decision_cache (key, doc) VALUES (?,?)",
                (key, json.dumps(doc, separators=(",", ":"), ensure_ascii=False)),
            )
