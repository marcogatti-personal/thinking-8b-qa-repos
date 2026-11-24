from __future__ import annotations

import fnmatch
import hashlib
import os
import sqlite3
import subprocess
import uuid
from pathlib import Path

from .common import match_query, search_text


SKIP_DIRS = {".git", ".hg", ".svn", ".venv", "venv", "node_modules", "__pycache__",
             ".thinking-8b-qa-repos", ".mypy_cache", ".pytest_cache", "dist", "build", "vendor"}
SECRET_PATTERNS = (".env", ".env.*", "*.pem", "*.key", "*.p12", "*.pfx", "id_rsa*", "id_ed25519*",
                   "credentials", "credentials.json", ".netrc", ".npmrc", ".pypirc")


def excluded_directory(name: str) -> bool:
    """Exclude dot-directories at every depth below the selected repository root."""
    return name.startswith(".") or name in SKIP_DIRS


SCHEMA = """
PRAGMA foreign_keys=ON;
PRAGMA journal_mode=WAL;
PRAGMA cache_size=-8192;
PRAGMA temp_store=FILE;
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS files(
 path TEXT PRIMARY KEY, digest TEXT NOT NULL, size INTEGER NOT NULL, seen TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS chunks(
 id INTEGER PRIMARY KEY AUTOINCREMENT, path TEXT NOT NULL REFERENCES files(path) ON DELETE CASCADE,
 start INTEGER NOT NULL, end INTEGER NOT NULL, body TEXT NOT NULL, search TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS chunks_path ON chunks(path,start);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(path,search,content='chunks',content_rowid='id');
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
 INSERT INTO chunks_fts(rowid,path,search) VALUES(new.id,new.path,new.search); END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
 INSERT INTO chunks_fts(chunks_fts,rowid,path,search) VALUES('delete',old.id,old.path,old.search); END;
"""


def nul_paths(stream):
    pending = b""
    while True:
        block = stream.read(65536)
        if not block:
            break
        parts = (pending + block).split(b"\0")
        pending = parts.pop()
        for part in parts:
            yield os.fsdecode(part)
    if pending:
        yield os.fsdecode(pending)


def paths(root: Path, mode: str = "auto"):
    """Stream Git's file list, including untracked nonignored files; fallback walk."""
    try:
        probe = subprocess.run(["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"],
                               capture_output=True, timeout=10)
        use_git = probe.returncode == 0 and probe.stdout.strip() == b"true"
    except (OSError, subprocess.TimeoutExpired):
        use_git = False
    if mode == "git" and not use_git:
        raise ValueError("Git mode requires a Git repository")
    if use_git and mode != "walk":
        with subprocess.Popen(["git", "-C", str(root), "ls-files", "--cached", "--others",
                               "--exclude-standard", "-z"], stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL) as process:
            yield from nul_paths(process.stdout)
            if process.wait() != 0:
                raise RuntimeError("git ls-files failed")
        return
    # scandir streaming: memory depends on directory depth, not repository size.
    def walk(directory):
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_symlink():
                    continue
                if entry.is_dir(follow_symlinks=False):
                    if not excluded_directory(entry.name):
                        yield from walk(entry.path)
                elif entry.is_file(follow_symlinks=False):
                    yield Path(entry.path).relative_to(root).as_posix()
    yield from walk(root)


def chunks(stream, chunk_bytes=2200, overlap_lines=6):
    """UTF-8 source chunks; bounded readline handles generated single-line files."""
    pending = []
    used = 0
    line_number = 1
    while True:
        raw = stream.readline(chunk_bytes)
        if not raw:
            break
        if b"\0" in raw:
            raise ValueError("binary")
        # A UTF-8 sequence can straddle the bounded read. Complete it without
        # dropping bytes or breaking line provenance.
        for _ in range(3):
            try:
                text = raw.decode("utf-8")
                break
            except UnicodeDecodeError as exc:
                if exc.reason != "unexpected end of data":
                    raise ValueError("non-utf8") from exc
                extra = stream.read(1)
                if not extra:
                    raise ValueError("non-utf8") from exc
                raw += extra
        else:
            text = raw.decode("utf-8")
        if pending and used + len(raw) > chunk_bytes:
            yield pending[0][0], pending[-1][0], "".join(t for _, t in pending)
            pending = pending[-overlap_lines:] if overlap_lines else []
            used = sum(len(t.encode("utf-8")) for _, t in pending)
            while pending and used + len(raw) > chunk_bytes:
                used -= len(pending.pop(0)[1].encode("utf-8"))
        pending.append((line_number, text))
        used += len(raw)
        line_number += raw.count(b"\n")
    if pending:
        yield pending[0][0], pending[-1][0], "".join(t for _, t in pending)


class RepoIndex:
    def __init__(self, database: Path, root: Path):
        self.root = root.resolve()
        if not self.root.is_dir():
            raise ValueError("Repository not found")
        database.parent.mkdir(parents=True, exist_ok=True)
        self.database = database.resolve()
        self.db = sqlite3.connect(database, timeout=60)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        old = self.meta("root")
        if old and old != str(self.root):
            self.db.close()
            raise ValueError("This index belongs to another repository")
        self.set_meta("root", str(self.root))
        self.db.commit()

    def close(self):
        self.db.close()

    def meta(self, key):
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key, value):
        self.db.execute("INSERT OR REPLACE INTO meta VALUES(?,?)", (key, str(value)))

    @property
    def revision(self):
        return self.meta("revision") or "empty"

    def build(self, *, mode="auto", exclude=(), max_file_bytes=64 * 1024 * 1024,
              chunk_bytes=2200, overlap_lines=6):
        if chunk_bytes < 128 or overlap_lines < 0 or max_file_bytes < 0:
            raise ValueError("Invalid indexing limits")
        stats = dict(indexed=0, unchanged=0, skipped=0, deleted=0, chunks=0, bytes=0)
        stamp = uuid.uuid4().hex
        settings = f"{chunk_bytes}:{overlap_lines}"
        rebuild = self.meta("chunk_settings") != settings
        # One SQLite transaction: a failed/interrupted build never exposes a
        # partial new snapshot. Content hashes detect edits with unchanged mtime.
        with self.db:
            for relative in paths(self.root, mode):
                rel = Path(relative)
                if (rel.is_absolute() or ".." in rel.parts or any(p in SKIP_DIRS for p in rel.parts)
                        or any(excluded_directory(p) for p in rel.parts[:-1])
                        or any(fnmatch.fnmatch(rel.name, p) for p in SECRET_PATTERNS)
                        or any(fnmatch.fnmatch(relative, p) for p in exclude)):
                    stats["skipped"] += 1
                    continue
                file = self.root / rel
                if (file.is_symlink() or not file.is_file() or file.resolve() != file.absolute()
                        or file.resolve() in {self.database, Path(str(self.database) + "-wal"),
                                              Path(str(self.database) + "-shm")}):
                    stats["skipped"] += 1
                    continue
                try:
                    size = file.stat().st_size
                    if max_file_bytes and size > max_file_bytes:
                        stats["skipped"] += 1
                        continue
                    digest = hashlib.sha256()
                    with file.open("rb") as stream:
                        for block in iter(lambda: stream.read(65536), b""):
                            digest.update(block)
                    hexdigest = digest.hexdigest()
                    old = self.db.execute("SELECT digest FROM files WHERE path=?", (relative,)).fetchone()
                    if old and old[0] == hexdigest and not rebuild:
                        self.db.execute("UPDATE files SET seen=? WHERE path=?", (stamp, relative))
                        stats["unchanged"] += 1
                        continue
                    self.db.execute("SAVEPOINT source_file")
                    try:
                        self.db.execute("DELETE FROM files WHERE path=?", (relative,))
                        self.db.execute("INSERT INTO files VALUES(?,?,?,?)", (relative, hexdigest, size, stamp))
                        count = 0
                        check = hashlib.sha256()
                        with file.open("rb") as stream:
                            # Hash again after parsing to reject files edited mid-index.
                            for start, end, body in chunks(stream, chunk_bytes, overlap_lines):
                                self.db.execute("INSERT INTO chunks(path,start,end,body,search) VALUES(?,?,?,?,?)",
                                                (relative, start, end, body, search_text(body)))
                                count += 1
                            stream.seek(0)
                            for block in iter(lambda: stream.read(65536), b""):
                                check.update(block)
                        if check.hexdigest() != hexdigest:
                            raise RuntimeError(f"File changed during indexing: {relative}; retry")
                    except (ValueError, OSError):
                        self.db.execute("ROLLBACK TO source_file")
                        stats["skipped"] += 1
                    else:
                        stats["indexed"] += 1
                        stats["chunks"] += count
                        stats["bytes"] += size
                    finally:
                        self.db.execute("RELEASE source_file")
                except OSError:
                    stats["skipped"] += 1
            stats["deleted"] = self.db.execute("SELECT count(*) FROM files WHERE seen<>?", (stamp,)).fetchone()[0]
            self.db.execute("DELETE FROM files WHERE seen<>?", (stamp,))
            if stats["indexed"] or stats["deleted"] or not self.meta("revision"):
                self.set_meta("revision", stamp)
            self.set_meta("chunk_settings", settings)
        stats["total_files"] = self.db.execute("SELECT count(*) FROM files").fetchone()[0]
        stats["total_chunks"] = self.db.execute("SELECT count(*) FROM chunks").fetchone()[0]
        stats["revision"] = self.revision
        return stats

    def get(self, chunk_id):
        return self.db.execute("SELECT * FROM chunks WHERE id=?", (chunk_id,)).fetchone()

    def search(self, query, limit=24, after=0):
        match = match_query(query)
        if not match:
            return []
        return self.db.execute("""SELECT c.*,bm25(chunks_fts,4.0,1.0) AS rank
          FROM chunks_fts JOIN chunks c ON c.id=chunks_fts.rowid
          WHERE chunks_fts MATCH ? ORDER BY rank,c.id LIMIT ? OFFSET ?""",
                               (match, limit, after)).fetchall()

    def scan(self, after, limit=24):
        return self.db.execute("SELECT * FROM chunks WHERE id>? ORDER BY id LIMIT ?", (after, limit)).fetchall()

    def stats(self):
        return {"root": str(self.root), "revision": self.revision,
                "files": self.db.execute("SELECT count(*) FROM files").fetchone()[0],
                "chunks": self.db.execute("SELECT count(*) FROM chunks").fetchone()[0]}


def evidence(row):
    return {"id": f"C{row['id']}", "path": row["path"], "start": row["start"],
            "end": row["end"], "text": row["body"]}
