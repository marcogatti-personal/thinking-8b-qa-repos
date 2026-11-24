from __future__ import annotations

import json
import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path


def dump(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def byte_len(text: str) -> int:
    return len(text.encode("utf-8"))


def clip(text: str, budget: int) -> str:
    return text.encode("utf-8")[:max(0, budget)].decode("utf-8", errors="ignore")


STOP = set("the a an and or to of in is are how what where why does do this that with for from "
           "il lo la le gli un una di del della nel come cosa dove quale quali che e è per con "
           "viene vengono repository code codice please explain spiega".split())


def terms(text: str, limit: int = 32) -> list[str]:
    expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    tokens = re.findall(r"[^\W_]+", expanded.lower(), flags=re.UNICODE)
    return list(dict.fromkeys(t for t in tokens if len(t) > 1 and t not in STOP))[:limit]


def search_text(text: str) -> str:
    # Keep whole identifiers AND camel/snake components; language-independent.
    expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return text + "\n" + expanded.replace("_", " ")


def match_query(text: str) -> str:
    return " OR ".join('"' + t.replace('"', '""') + '"' for t in terms(text))


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


@contextmanager
def session_lock(directory: Path):
    """OS lock survives crashes without stale lock recovery; not a PID file."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a+b") as lock:
        try:
            if os.name == "nt":
                import msvcrt
                lock.seek(0)
                lock.write(b"0")
                lock.flush()
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Session is already in use by another process") from exc
        try:
            yield
        finally:
            if os.name == "nt":
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
