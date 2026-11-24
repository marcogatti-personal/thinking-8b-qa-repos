"""Session-local cache of validated exact requests, independent of checkpoints."""
import hashlib
import json
import sqlite3

from .common import dump


class ResponseCache:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS responses(key TEXT PRIMARY KEY, value TEXT NOT NULL)")

    def key(self, identity, snapshot, task, system, payload, schema):
        return hashlib.sha256(dump([identity, snapshot, task, system, payload, schema]).encode()).hexdigest()

    def get(self, key):
        row = self.db.execute("SELECT value FROM responses WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key, value):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO responses VALUES(?,?)", (key, dump(value)))

    def close(self):
        self.db.close()

    def delete(self, key):
        with self.db:
            self.db.execute("DELETE FROM responses WHERE key=?", (key,))
