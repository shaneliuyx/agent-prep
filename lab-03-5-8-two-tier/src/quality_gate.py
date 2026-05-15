"""Quality-score promotion gate — STM → LTM filter.

Reference: PraisonAI's memory subsystem uses a similar pattern at
src/praisonai-agents/praisonaiagents/memory/memory.py — each candidate
STM entry receives a quality_score in [0.0, 1.0]; only entries above
a configurable threshold promote to LTM.

Three signals combined:
  (a) length      — reward summaries in the 8-25 word band
  (b) specificity — concrete tokens (numbers, units, proper nouns)
  (c) novelty     — 1.0 - max(similarity to existing memories) via EverCore search

The novelty signal calls EverCore's /api/v1/memories/search; on any error
(connection failure, empty store, search timeout) it defaults to 1.0 so
the consolidation pipeline degrades gracefully instead of stalling.

Default weights and thresholds are tuned for the 20-scroll probe set
(10 high-value + 10 low-value) — see RESULTS.md for measured precision/recall
at threshold ∈ {0.3, 0.5, 0.7}. Tune per domain.
"""
from __future__ import annotations

import re
from typing import Any


SPECIFICITY_HINTS = re.compile(
    r"\b(\d+(\.\d+)?(%|ms|s|min|h|GB|MB|req/s)?|[A-Z][a-zA-Z]{2,})\b"
)

DEFAULT_WEIGHTS = {"length": 0.3, "specificity": 0.4, "novelty": 0.3}
DEFAULT_THRESHOLD = 0.5


def _length_score(summary: str) -> float:
    """Reward summaries in the 8-25 word band; penalise outside.

    Triangular peak at 15 words. Below 5 words → 0.0 (no content).
    Above 40 → 0.2 (verbose; clamped, not zeroed, because long facts
    can still be useful if specific).
    """
    n_words = len(summary.split())
    if n_words < 5:
        return 0.0
    if n_words > 40:
        return 0.2
    return max(0.0, 1.0 - abs(n_words - 15) / 15.0)


def _specificity_score(summary: str) -> float:
    """Reward summaries with concrete tokens (numbers, units, proper nouns).

    Saturates at 3 specific tokens — diminishing returns past that.
    """
    hits = len(SPECIFICITY_HINTS.findall(summary))
    return min(1.0, hits / 3.0)


def _novelty_score(
    summary: str,
    tm: "Any | None",
    top_k: int = 5,
) -> float:
    """Novelty = 1.0 - max(similarity) over top-k nearest existing memories.

    EverCore's hybrid search returns each episode with a `score` in [0.0, 1.0].
    High score = high similarity = low novelty. On any search failure
    (connection refused, empty store, timeout, schema drift) we default to 1.0
    so the consolidation pipeline keeps running. Production rule: a novelty
    signal that crashes the pipeline is worse than one that occasionally
    over-promotes.
    """
    if tm is None:
        return 1.0
    try:
        matches = tm.query_context(query=summary, k=top_k)
    except Exception:                                              # noqa: BLE001
        return 1.0
    if not matches:
        return 1.0
    scores = [float(m.get("score") or 0.0) for m in matches]
    return max(0.0, 1.0 - max(scores))


def quality_score(
    summary: str,
    tm: "Any | None" = None,
    weights: dict[str, float] | None = None,
) -> float:
    """Combined quality score in [0.0, 1.0]. Weighted average of three signals.

    Pass `tm` to enable the EverCore-backed novelty signal; omit (or pass None)
    for unit-testable offline scoring (novelty defaults to 1.0).

    `weights` overrides defaults; missing keys fall back to DEFAULT_WEIGHTS.
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    length = _length_score(summary)
    specificity = _specificity_score(summary)
    novelty = _novelty_score(summary, tm)
    return w["length"] * length + w["specificity"] * specificity + w["novelty"] * novelty


def should_promote(
    summary: str,
    threshold: float = DEFAULT_THRESHOLD,
    tm: "Any | None" = None,
    weights: dict[str, float] | None = None,
) -> bool:
    """Promotion gate. Default threshold tuned on the 20-scroll probe.

    Domain biases:
      - incident-response agents → threshold LOW (false-negative loses lessons)
      - high-precision research agents → threshold HIGH (false-positive pollutes LTM)
    """
    return quality_score(summary, tm=tm, weights=weights) >= threshold
