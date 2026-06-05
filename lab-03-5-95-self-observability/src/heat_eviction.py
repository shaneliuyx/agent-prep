"""W3.5.95 — Heat-scored eviction for the LEARNING store (leverages BAI-LAB/MemoryOS).

`metacog_recall` RANKS facts at read time but never PRUNES them — the `learning`
table grows unbounded, so recall slows and stale patterns accumulate. BAI-LAB's
MemoryOS keeps memory bounded with a HEAT score (visits + recency + importance)
and evicts the coldest, guarded by two rules:

  - importance-exempt: high-confidence facts are NEVER evicted (a rarely-recalled
    but high-confidence "I deadlock on nested locks" must survive a quiet month);
  - dedup: near-duplicate patterns collapse to the single strongest copy.

Pure code, no LLM — matches `metacog_recall`'s ethos. Dedup uses lexical TF-cosine
as a cheap LOCAL proxy for the semantic embedding-cosine BAI-LAB uses; note the
substitution honestly rather than pulling in an embedding dependency here.

Provenance: heat/eviction is BAI-LAB/MemoryOS. (The Ground-Truth Hierarchy in
W3.5.96 is the *other* repo, ClaudioDrews/memory-os — don't conflate them.)
"""
from __future__ import annotations

import math
import sqlite3
import time
from collections import Counter

from metacog_recall import RECENCY_HALFLIFE_S, _tok

# Heat weights — visits, recency, importance. Tunable; equal-weighted by default.
ALPHA_VISITS = 1.0
BETA_RECENCY = 1.0
GAMMA_IMPORTANCE = 1.0
IMPORTANCE_FLOOR = 0.85   # confidence ≥ floor ⇒ eviction-exempt
DEDUP_THRESHOLD = 0.92    # TF-cosine ≥ threshold ⇒ near-duplicate


def ensure_heat_columns(conn: sqlite3.Connection) -> None:
    """Add visit-tracking columns if absent (idempotent migration)."""
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(learning)")}
    if "recall_count" not in cols:
        conn.execute("ALTER TABLE learning ADD COLUMN recall_count INTEGER NOT NULL DEFAULT 0")
    if "last_recalled_ts" not in cols:
        conn.execute("ALTER TABLE learning ADD COLUMN last_recalled_ts REAL")
    conn.commit()


def touch(conn: sqlite3.Connection, ids: list[int], now: float | None = None) -> None:
    """Record a recall 'visit' — the heat signal. `recall()` calls this for the
    facts it injects (one-line wire-in); the demo calls it to simulate history."""
    now = now or time.time()
    conn.executemany(
        "UPDATE learning SET recall_count = recall_count + 1, last_recalled_ts = ? WHERE id = ?",
        [(now, i) for i in ids],
    )
    conn.commit()


def heat(row: sqlite3.Row, now: float | None = None) -> float:
    """visits (log-damped) + recency (decay since last touch/write) + importance."""
    now = now or time.time()
    last_seen = row["last_recalled_ts"] or row["ts"]
    recency = 0.5 ** ((now - last_seen) / RECENCY_HALFLIFE_S)
    return (
        ALPHA_VISITS * math.log1p(row["recall_count"])
        + BETA_RECENCY * recency
        + GAMMA_IMPORTANCE * row["confidence"]
    )


def _cosine(a: Counter, b: Counter) -> float:
    common = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def dedup(conn: sqlite3.Connection, threshold: float = DEDUP_THRESHOLD,
          now: float | None = None) -> int:
    """Collapse near-duplicate patterns to the hottest copy. Returns rows removed."""
    now = now or time.time()
    rows = conn.execute("SELECT * FROM learning").fetchall()
    tfs = {r["id"]: Counter(_tok(r["pattern_text"])) for r in rows}
    removed: set[int] = set()
    for i, a in enumerate(rows):
        if a["id"] in removed:
            continue
        for b in rows[i + 1:]:
            if b["id"] in removed:
                continue
            if _cosine(tfs[a["id"]], tfs[b["id"]]) >= threshold:
                loser = a if heat(a, now) < heat(b, now) else b
                removed.add(loser["id"])
    if removed:
        conn.executemany("DELETE FROM learning WHERE id = ?", [(i,) for i in removed])
        conn.commit()
    return len(removed)


def evict(conn: sqlite3.Connection, budget: int, now: float | None = None,
          importance_floor: float = IMPORTANCE_FLOOR) -> int:
    """Evict the coldest eviction-eligible facts until total ≤ budget.
    Facts with confidence ≥ importance_floor are exempt. Returns rows removed."""
    now = now or time.time()
    total = conn.execute("SELECT COUNT(*) AS n FROM learning").fetchone()["n"]
    if total <= budget:
        return 0
    eligible = [r for r in conn.execute("SELECT * FROM learning").fetchall()
                if r["confidence"] < importance_floor]
    eligible.sort(key=lambda r: heat(r, now))  # coldest first
    to_remove = eligible[: total - budget]
    if to_remove:
        conn.executemany("DELETE FROM learning WHERE id = ?", [(r["id"],) for r in to_remove])
        conn.commit()
    return len(to_remove)


def enforce(conn: sqlite3.Connection, budget: int, now: float | None = None) -> dict[str, int]:
    """Dedup, then evict to budget. The single entry point a retention job calls."""
    ensure_heat_columns(conn)
    deduped = dedup(conn, now=now)
    evicted = evict(conn, budget, now=now)
    final = conn.execute("SELECT COUNT(*) AS n FROM learning").fetchone()["n"]
    return {"deduped": deduped, "evicted": evicted, "final": final}
