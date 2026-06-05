"""W3.5.95 — Metacognitive recall: the pre-decision query layer (chapter §2.2
concept 4 + Phase 4).

At each agent step, BEFORE it picks an action, query LEARNING for self-pattern
facts relevant to the current (prompt, scratchpad), score by BM25 × recency-decay,
and inject the top-K into the context window under a clear header. This is the
READ side of self-facing memory — in-context, not fine-tuning (chapter §2.4):
cheap, immediate, debuggable, rollback-able.

Pure code, no LLM — recall is retrieval, not generation. (Generating new
self-knowledge in-the-moment is reflection, a W5.5 primitive; this is recall.)
"""
from __future__ import annotations

import math
import re
import sqlite3
import time

_TOKEN = re.compile(r"[a-z0-9]+")
RECALL_K = 3
RECENCY_HALFLIFE_S = 14 * 24 * 3600.0  # 2 weeks — a fact's weight halves per fortnight


def _tok(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _bm25_scores(query: str, docs: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """Lightweight BM25 over the (small) LEARNING corpus. Standard formula;
    avgdl/idf computed over just the stored patterns — fine at this scale."""
    q_terms = set(_tok(query))
    doc_toks = [_tok(d) for d in docs]
    n = len(docs)
    avgdl = (sum(len(d) for d in doc_toks) / n) if n else 0.0
    # document frequency per query term
    df = {t: sum(1 for dt in doc_toks if t in dt) for t in q_terms}
    scores = []
    for dt in doc_toks:
        dl = len(dt)
        s = 0.0
        for t in q_terms:
            if t not in dt:
                continue
            f = dt.count(t)
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            s += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / (avgdl or 1)))
        scores.append(s)
    return scores


def recall(conn: sqlite3.Connection, query: str, k: int = RECALL_K,
           now: float | None = None, track: bool = False) -> list[sqlite3.Row]:
    """Return the top-k LEARNING facts for the query, ranked by
    BM25 × recency-decay × confidence. Empty list if nothing relevant (BM25>0).

    track=True records a recall 'visit' on the returned facts (heat_eviction.touch),
    so the heat score reflects what actually gets used. Default False keeps recall
    pure-read for tests/ablation; the live loop (recall_block) opts in."""
    now = now or time.time()
    facts = conn.execute("SELECT * FROM learning").fetchall()
    if not facts:
        return []
    bm = _bm25_scores(query, [f["pattern_text"] for f in facts])
    ranked = []
    for f, score in zip(facts, bm):
        if score <= 0:
            continue  # not lexically relevant — don't inject noise
        recency = 0.5 ** ((now - f["ts"]) / RECENCY_HALFLIFE_S)
        ranked.append((score * recency * f["confidence"], f))
    ranked.sort(key=lambda x: x[0], reverse=True)
    top = [f for _, f in ranked[:k]]
    if track and top:
        # lazy import: heat_eviction imports this module, so importing at top
        # would cycle. By call time metacog_recall is fully loaded.
        from heat_eviction import ensure_heat_columns, touch
        ensure_heat_columns(conn)
        touch(conn, [f["id"] for f in top], now=now)
    return top


def format_injection(facts: list[sqlite3.Row]) -> str:
    """Render recalled facts as a context-window block. Hedged ('you observed in
    the past ... verify') so the agent treats them as priors, not ground truth —
    guards against stale-pattern over-confidence."""
    if not facts:
        return ""
    lines = ["## Self-Patterns You Have Observed",
             "(You noticed these about your own past behavior. Treat as priors to "
             "consider, not certainties — verify before relying on them.)"]
    for f in facts:
        lines.append(f"- [{f['type']}] {f['pattern_text']}")
    return "\n".join(lines)


def recall_block(conn: sqlite3.Connection, query: str, k: int = RECALL_K,
                 track: bool = True) -> str:
    """Convenience: recall + format in one call (what the agent loop uses).
    track defaults True here — the live loop's recalls SHOULD accrue heat."""
    return format_injection(recall(conn, query, k, track=track))
