"""Deterministic memory access decisions. No model inference or learned rewards."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .common import clip, terms


@dataclass(frozen=True)
class MemoryDecision:
    mode: str
    byte_budget: int
    candidate_limit: int
    max_facts: int | None
    focused: bool = False


def memory_decision(policy: str, capacity: int, population: int, stale: int) -> MemoryDecision:
    """Population can saturate at five: only empty/sparse/dense bins matter."""
    if policy == "static":
        return MemoryDecision("static", capacity, 40, None)
    if policy != "adaptive":
        raise ValueError("memory_policy must be adaptive or static")
    if stale:
        return MemoryDecision("expanded", capacity, 40, None, True)
    if not population:
        return MemoryDecision("empty", min(capacity, 1024), 0, 0, True)
    if population <= 4:
        return MemoryDecision("compact", min(capacity, 1024), 8, 4, True)
    return MemoryDecision("focused", min(capacity, 2048), 24, 8, True)


GENERIC = set("defined definition implementation implemented function method class value variable "
              "return returns caller callers calls called behavior handled missing unknown unresolved "
              "find locate check used using which when whether exists its can source file".split())
IDENTIFIER = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+\b|\b[a-z]+[A-Z][A-Za-z0-9]*\b")


def recovery_queries(unknowns, quotes, limit=16):
    """Bounded alternative lexical queries, prioritizing explicit code identifiers.

    No appended 'try again' instructions: each query changes the search terms.
    Values remain search data. Quotes supply hints, never new facts.
    """
    questions = [clip(q, 180) for q in unknowns[:4]]
    snippets = [clip(q, 500) for q in quotes[:6]]
    candidates = []
    for text in questions:
        candidates.extend(re.findall(r"`([A-Za-z_][A-Za-z0-9_.]*)`", text))
        candidates.extend(IDENTIFIER.findall(text))
    # Split unresolved questions into selective searches before trying source hints.
    for text in questions:
        candidates.extend(t for t in terms(text, 12) if t not in GENERIC)
    for text in snippets:
        candidates.extend(IDENTIFIER.findall(text))
    output, keys = [], set()
    for query in candidates:
        query = clip(query.strip().lower(), 180)
        key = tuple(sorted(terms(query)))
        if key and key not in keys:
            output.append(query)
            keys.add(key)
        if len(output) >= limit:
            break
    return output
