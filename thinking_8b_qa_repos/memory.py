from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

from .common import atomic_write, byte_len, clip, match_query, terms
from .control import MemoryDecision


class Memory:
    """Append-only evidence archive plus a fixed-byte working-memory projection."""
    def __init__(self, directory: Path, capacity: int = 4096):
        if capacity < 1024:
            raise ValueError("memory_bytes must be at least 1024")
        self.directory = directory
        self.capacity = capacity
        self.render_stats = {}
        directory.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(directory / "archive.sqlite")
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
        PRAGMA journal_mode=WAL;
        PRAGMA cache_size=-4096;
        CREATE TABLE IF NOT EXISTS facts(
          id INTEGER PRIMARY KEY, fingerprint TEXT UNIQUE, text TEXT, chunk_id INTEGER,
          quote TEXT, step INTEGER, path TEXT);
        CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(text, content='facts', content_rowid='id');
        CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
          INSERT INTO facts_fts(rowid,text) VALUES(new.id,new.text); END;
        CREATE TABLE IF NOT EXISTS seen(chunk_id INTEGER PRIMARY KEY, step INTEGER);
        CREATE TABLE IF NOT EXISTS queries(
          id INTEGER PRIMARY KEY, text TEXT UNIQUE, offset INTEGER DEFAULT 0,
          priority INTEGER DEFAULT 0, exhausted INTEGER DEFAULT 0);
        CREATE TABLE IF NOT EXISTS kv(key TEXT PRIMARY KEY, value TEXT);
        CREATE INDEX IF NOT EXISTS facts_recent ON facts(step DESC,id DESC);
        CREATE INDEX IF NOT EXISTS query_queue ON queries(exhausted,priority DESC,id);
        """)

    def close(self):
        self.db.close()

    def add_query(self, query, priority=0):
        if isinstance(query, str) and query.strip():
            query = clip(query.strip().lower(), 180)
            cursor = self.db.execute("INSERT OR IGNORE INTO queries(text,priority) VALUES(?,?)", (query, priority))
            return cursor.rowcount > 0
        return False

    def has_seen(self, chunk_id):
        return self.db.execute("SELECT 1 FROM seen WHERE chunk_id=?", (chunk_id,)).fetchone() is not None

    def accept(self, finding, sources, step):
        """Check provenance mechanically; does NOT prove semantic entailment."""
        if not isinstance(finding, dict):
            return False
        text, ref, quote = (finding.get(k) for k in ("text", "source", "quote"))
        if not all(isinstance(v, str) for v in (text, ref, quote)):
            return False
        if not 8 <= byte_len(text) <= 600 or not 12 <= byte_len(quote) <= 500:
            return False
        source = sources.get(ref)
        if not source or quote not in source["text"]:
            return False
        fingerprint = hashlib.sha256((text.strip().casefold() + "\0" + ref).encode()).hexdigest()
        cursor = self.db.execute("INSERT OR IGNORE INTO facts(fingerprint,text,chunk_id,quote,step,path) VALUES(?,?,?,?,?,?)",
                                 (fingerprint, text.strip(), int(ref[1:]), quote, step, source["path"]))
        return cursor.rowcount > 0

    def candidates(self, query, limit=40):
        found = {}
        match = match_query(query)
        if match:
            for row in self.db.execute("""SELECT f.* FROM facts_fts JOIN facts f ON f.id=facts_fts.rowid
               WHERE facts_fts MATCH ? ORDER BY bm25(facts_fts), f.id DESC LIMIT ?""", (match, limit)):
                found[row["id"]] = row
        for row in self.db.execute("SELECT * FROM facts ORDER BY step DESC,id DESC LIMIT ?", (limit,)):
            found.setdefault(row["id"], row)
        return list(found.values())

    def render(self, question, focus, unknowns, decision=None):
        """Deterministic compaction; old findings remain retrievable from SQLite."""
        decision = decision or MemoryDecision("static", self.capacity, 40, None)
        budget = min(self.capacity, max(1024, decision.byte_budget))
        header = "REPOSITORY WORKING MEMORY\n" + "Question: " + clip(question, min(500, budget // 5)) + "\n"
        suffix = "\nOpen questions:\n" + "\n".join("- " + clip(q, 160) for q in unknowns[:3] if isinstance(q, str)) + "\n"
        suffix = clip(suffix, budget // 5)
        remaining = budget - byte_len(header) - byte_len(suffix) - 1
        query = question + " " + focus
        if decision.focused:
            query += " " + " ".join(clip(q, 180) for q in unknowns[:4])
        qterms = set(terms(query, 64))
        candidates = self.candidates(query, decision.candidate_limit) if decision.candidate_limit else []
        if decision.focused and candidates:
            # Keep one recent continuity anchor even without lexical overlap.
            anchor = max(candidates, key=lambda row: (row["step"], row["id"]))["id"]
            candidates = [r for r in candidates if r["id"] == anchor or
                          qterms.intersection(terms(r["text"] + " " + r["path"], 80))]
        latest = max((row["step"] for row in candidates), default=1) or 1
        # Tokenize each candidate once, not on every greedy ranking iteration.
        base_scores = {row["id"]: len(qterms.intersection(terms(row["text"] + " " + row["path"], 80))) * 3
                       + 2 * row["step"] / latest for row in candidates}
        selected, used_paths = [], {}
        while candidates:
            if decision.max_facts is not None and len(selected) >= decision.max_facts:
                break
            def score(row):
                return base_scores[row["id"]] - used_paths.get(row["path"], 0) * 4
            winner = max(candidates, key=score)
            candidates.remove(winner)
            line = f"[F{winner['id']}] {winner['text']} [C{winner['chunk_id']}]\n"
            if byte_len(line) <= remaining:
                selected.append(line)
                remaining -= byte_len(line)
                used_paths[winner["path"]] = used_paths.get(winner["path"], 0) + 1
        content = header + "".join(selected) + suffix
        self.render_stats = {"memory_active_bytes": byte_len(content.rstrip()),
                             "memory_facts": len(selected), "memory_mode": decision.mode}
        padding = self.capacity - byte_len(content) - 1
        assert padding >= 0
        atomic_write(self.directory / "memory.txt", content + " " * padding + "\n")
        return content.rstrip()

    def stats(self):
        return {"facts": self.db.execute("SELECT count(*) FROM facts").fetchone()[0],
                "seen_chunks": self.db.execute("SELECT count(*) FROM seen").fetchone()[0],
                "memory_bytes": (self.directory / "memory.txt").stat().st_size, **self.render_stats}
