"""Rank-fusion strategies.

Currently exposes Reciprocal Rank Fusion (RRF) — the rank-domain fusion baseline
used in production hybrid-search systems. Designed for one extension point:
swap or add fusion methods (Distribution-Based Score Fusion, learned fusion)
without touching callsites.

Why rank-domain fusion (RRF) and not score normalization:
  Cormack et al. (SIGIR 2009) showed RRF outperforms BM25-score normalization
  on heterogeneous retriever combinations. k=60 is the empirical default —
  insensitive to corpus size, robust across retriever pairs.

Native vs manual:
  Qdrant 1.10+ provides server-side RRF via Prefetch + FusionQuery. Use that
  via the Retriever class. This module hosts the *manual* (Python-side) RRF
  for backends without native fusion support, or for cross-backend fusion.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Iterable, Protocol, Sequence

RRF_DEFAULT_K = 60  # Cormack et al. 2009 — robust default; corpus-size insensitive.


class _HasId(Protocol):
    id: Any


class FusionStrategy(str, Enum):
    """Selects fusion impl. NATIVE_RRF used by Retriever when backend supports it
    (Qdrant 1.10+); MANUAL_RRF works against any rank-list source."""
    MANUAL_RRF = "manual_rrf"
    NATIVE_RRF = "native_rrf"


def rrf_fuse(rankings: Sequence[Iterable[_HasId]], k: int = RRF_DEFAULT_K) -> list:
    """Reciprocal Rank Fusion (Cormack et al. 2009).

    Each ranking contributes ``1 / (k + rank + 1)`` to its docs' aggregate score
    (rank is 0-indexed; the +1 makes the formula match the published convention
    where rank starts at 1). Returns docs sorted by aggregate score, deduped by
    ``id``. First occurrence wins for the returned object (so payloads from the
    earlier ranking survive).

    Caller invariant: every input element has a hashable ``.id``. Designed for
    Qdrant ``ScoredPoint`` objects but works for anything with that shape.
    """
    scores: dict = {}
    pt_by_id: dict = {}
    for ranking in rankings:
        for rank, pt in enumerate(ranking):
            pid = pt.id
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
            pt_by_id.setdefault(pid, pt)

    # Stable sort: ties resolved by insertion order, so callers get reproducible
    # output for tied scores (which happen often when rankings share top docs).
    fused_ids = sorted(scores.keys(), key=lambda pid: -scores[pid])
    return [pt_by_id[pid] for pid in fused_ids]
