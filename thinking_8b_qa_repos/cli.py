from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from .demo import DemoModel, make_demo
from .engine import Harness, Settings
from .index import RepoIndex
from .model import LocalModel


def progress_arguments(command):
    options = command.add_mutually_exclusive_group()
    options.add_argument("--progress", choices=["concise", "detailed", "quiet"],
                         help="Live investigation summaries (default: concise); detailed shows evidence and technical counters")
    options.add_argument("--explain", action="store_true", help="Alias for --progress detailed; brief live summaries are shown by default")


def parser():
    p = argparse.ArgumentParser(prog="thinking-8b-qa-repos", description="Repository Q&A with bounded working memory, disk-backed evidence and a local LLM.")
    sub = p.add_subparsers(dest="command", required=True)
    index = sub.add_parser("index", help="Index or update the snapshot without model calls")
    index.add_argument("repo", type=Path)
    index.add_argument("--db", type=Path)
    index.add_argument("--mode", choices=["auto", "git", "walk"], default="auto")
    index.add_argument("--exclude", action="append", default=[], help="Glob on relative paths; may be repeated")
    index.add_argument("--max-file-mb", type=float, default=64, help="0: unlimited; default 64 MiB")
    index.add_argument("--chunk-bytes", type=int, default=2200)
    index.add_argument("--overlap-lines", type=int, default=6)
    ask = sub.add_parser("ask", help="Ask a question about an indexed snapshot")
    ask.add_argument("repo", type=Path)
    ask.add_argument("question")
    ask.add_argument("--db", type=Path)
    ask.add_argument("--session", type=Path)
    ask.add_argument("--resume", action="store_true")
    ask.add_argument("--synthesize-only", action="store_true", help="With --resume, synthesize saved findings without planning or reading again")
    progress_arguments(ask)
    ask.add_argument("--model", required=True, help="ID of an already loaded or installed model; no automatic download")
    ask.add_argument("--provider", choices=["openai", "ollama"], default="openai")
    ask.add_argument("--base-url", help="Default: LM Studio 127.0.0.1:1234/v1, Ollama 127.0.0.1:11434")
    ask.add_argument("--allow-remote", action="store_true")
    ask.add_argument("--no-structured", action="store_true", help="Request JSON through the prompt if the server does not support JSON Schema")
    ask.add_argument("--context-window", type=int, default=16384)
    ask.add_argument("--output-tokens", type=int, default=1536)
    ask.add_argument("--prompt-bytes", type=int, default=12000)
    ask.add_argument("--memory-bytes", type=int, default=4096)
    ask.add_argument("--memory-policy", choices=["adaptive", "static"], default="adaptive",
                     help="Adaptive memory budgets and bounded search recovery; static reproduces the fixed policy")
    ask.add_argument("--evidence-bytes", type=int, default=6000)
    ask.add_argument("--max-steps", type=int, default=16, help="Total number of reading steps, including previous steps on resume")
    ask.add_argument("--patience", type=int, default=4)
    ask.add_argument("--scan", action="store_true", help="Scan the entire index sequentially within max-steps")
    ask.add_argument("--timeout", type=float, default=120)
    ask.add_argument("--retries", type=int, default=2)
    ask.add_argument("--no-cache", action="store_true", help="Disable reuse of identical requests in this session")
    search = sub.add_parser("search", help="Test chunk retrieval without an LLM")
    search.add_argument("repo", type=Path)
    search.add_argument("query")
    search.add_argument("--db", type=Path)
    search.add_argument("--limit", type=int, default=8)
    demo = sub.add_parser("demo", help="Run a simulated-model demo without model weights or a GPU")
    demo.add_argument("--output", type=Path, default=Path("demo-output"))
    progress_arguments(demo)
    sub.add_parser("doctor", help="Check Python and SQLite FTS5 without starting or downloading models")
    return p


def default_db(repo):
    preferred = repo.resolve() / ".thinking-8b-qa-repos" / "index.sqlite"
    legacy = repo.resolve() / ".repo-memory8b" / "index.sqlite"
    return legacy if legacy.exists() and not preferred.exists() else preferred


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "doctor":
            connection = sqlite3.connect(":memory:")
            connection.execute("CREATE VIRTUAL TABLE check_fts USING fts5(text)")
            connection.close()
            print(json.dumps({"python": sys.version.split()[0], "sqlite": sqlite3.sqlite_version,
                              "fts5": True, "venv": sys.prefix != sys.base_prefix}, indent=2))
            return 0
        if args.command == "demo":
            destination = args.output.resolve()
            if destination.exists() and any(destination.iterdir()):
                raise ValueError("The demo directory must be empty; choose another --output")
            repo = destination / "repository"
            make_demo(repo)
            index = RepoIndex(destination / "index.sqlite", repo)
            try:
                index.build()
                engine = Harness(index, DemoModel(), destination / "session",
                                 Settings(memory_bytes=2048, evidence_bytes=800),
                                 progress=lambda s: print(s, file=sys.stderr, flush=True),
                                 explain=args.explain, progress_mode=args.progress)
                result = engine.run("How does checkout work and how is the idempotency key reused?")
                print("SIMULATED DEMO: this does not measure LLM quality.\n")
                print(engine.markdown(result))
                print(f"File: {destination / 'session' / 'answer.md'}")
            finally:
                index.close()
            return 0
        database = args.db or default_db(args.repo)
        if args.command in {"ask", "search"} and not database.exists():
            raise ValueError("Index missing: run the index command first")
        index = RepoIndex(database, args.repo)
        try:
            if args.command == "index":
                excluded = list(args.exclude)
                # Exclude the custom database itself; never index generated WAL/SHM.
                result = index.build(mode=args.mode, exclude=excluded,
                                     max_file_bytes=int(args.max_file_mb * 1024 * 1024),
                                     chunk_bytes=args.chunk_bytes, overlap_lines=args.overlap_lines)
                print(json.dumps(result, indent=2))
            elif args.command == "search":
                if not 1 <= args.limit <= 100:
                    raise ValueError("limit must be between 1 and 100")
                for row in index.search(args.query, args.limit):
                    print(f"[C{row['id']}] {row['path']}:{row['start']}-{row['end']}\n{row['body']}\n")
            else:
                if args.resume and args.session is None:
                    raise ValueError("--resume requires --session")
                if args.synthesize_only and not args.resume:
                    raise ValueError("--synthesize-only requires --resume and an existing --session")
                if not index.meta("revision"):
                    raise ValueError("Index is not ready: run index first")
                session = args.session or database.parent / "sessions" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                base_url = args.base_url or ("http://127.0.0.1:11434" if args.provider == "ollama" else "http://127.0.0.1:1234/v1")
                model = LocalModel(provider=args.provider, base_url=base_url, model=args.model,
                                   prompt_bytes=args.prompt_bytes, output_tokens=args.output_tokens,
                                   context_window=args.context_window, timeout=args.timeout, retries=args.retries,
                                   structured=not args.no_structured, allow_remote=args.allow_remote)
                settings = Settings(memory_bytes=args.memory_bytes, evidence_bytes=args.evidence_bytes,
                                    prompt_bytes=args.prompt_bytes, max_steps=args.max_steps,
                                    scan=args.scan, patience=args.patience, cache=not args.no_cache,
                                    memory_policy=args.memory_policy)
                engine = Harness(index, model, session, settings, progress=lambda s: print(s, file=sys.stderr, flush=True),
                                 explain=args.explain, progress_mode=args.progress)
                if args.progress != "quiet":
                    print(f"Session: {session.resolve()}", file=sys.stderr, flush=True)
                result = engine.run(args.question, args.resume, args.synthesize_only)
                print(engine.markdown(result))
                if args.progress != "quiet":
                    print(f"File: {(session / 'answer.md').resolve()}", file=sys.stderr, flush=True)
        finally:
            index.close()
        return 0
    except (ValueError, RuntimeError, OSError, sqlite3.Error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted. The last completed step is checkpointed; use --resume.", file=sys.stderr)
        return 130
