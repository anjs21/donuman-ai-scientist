"""Small durable store. Every write commits; model messages are not the database."""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from .common import dumps, digest


class Store:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = str(Path(path).resolve())
        self.db = sqlite3.connect(self.path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS records
        (kind TEXT NOT NULL, id TEXT NOT NULL, body TEXT NOT NULL,
         PRIMARY KEY(kind,id));
        CREATE TABLE IF NOT EXISTS events
        (seq INTEGER PRIMARY KEY AUTOINCREMENT, time TEXT NOT NULL,
         kind TEXT NOT NULL, body TEXT NOT NULL);
        """)
        self.db.commit()

    def put(self, kind, key, body):
        self.db.execute("INSERT OR REPLACE INTO records VALUES (?,?,?)",
                        (kind, key, dumps(body)))
        self.db.commit()

    def get(self, kind, key, default=None):
        row = self.db.execute("SELECT body FROM records WHERE kind=? AND id=?",
                              (kind, key)).fetchone()
        return json.loads(row[0]) if row else default

    def all(self, kind):
        return [json.loads(r[0]) for r in self.db.execute(
            "SELECT body FROM records WHERE kind=? ORDER BY id", (kind,))]

    def event(self, kind, body):
        self.db.execute("INSERT INTO events(time,kind,body) VALUES (?,?,?)",
                        (datetime.now(timezone.utc).isoformat(), kind, dumps(body)))
        self.db.commit()

    def recent(self, kind, n):
        """The n records of one kind written last, oldest first (rewrites move a record to the end)."""
        rows = self.db.execute("SELECT body FROM records WHERE kind=? ORDER BY rowid DESC LIMIT ?",
                               (kind, n)).fetchall()
        return [json.loads(r[0]) for r in reversed(rows)]

    def events(self, kind):
        """Event bodies of one kind, oldest first."""
        return [json.loads(r[0]) for r in self.db.execute(
            "SELECT body FROM events WHERE kind=? ORDER BY seq", (kind,))]

    def evidence(self, text, source, kind="literature"):
        key = "ev_" + digest({"text": text, "source": source, "kind": kind})[:16]
        self.put("evidence", key, {"id": key, "text": text, "source": source,
                                  "kind": kind})
        return key

    def close(self):
        self.db.close()
