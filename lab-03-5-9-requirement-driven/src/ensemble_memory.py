# src/ensemble_memory.py — Pattern-3 ensemble backend (~90 LOC)
"""ENSEMBLE memory: fan out reads to multiple backends, RRF-merge their
retrieved facts, return the fused union.

WHY (the §2.3 stretch, motivated by measured data): a question-type ROUTER
(RouterMemory) PICKS ONE backend per question, so it is upper-bounded by
"best-single-backend-per-axis" — on the w358 slice it scored 50%, 5 pts BELOW
the best single backend (atomic_fact 55%). An ensemble has no such ceiling: it
COMBINES backends, so it can answer questions that every individual backend
misses. Each member misses a DIFFERENT needle subset (mem0's BM25 catches
lexical matches dense retrieval drops, and vice-versa), so the union recovers
needles no single store surfaces.

Fusion = Reciprocal Rank Fusion (RRF) — the same rank-based fusion Mem0 uses
internally to merge semantic + BM25 + entity signals. RRF is robust because it
needs only RANKS, not comparable scores across heterogeneous stores.

Writes go to every member; reads over-fetch per member, then RRF-merge.
"""
from __future__ import annotations

import os
from typing import Any

# Standard RRF constant. Larger K flattens the rank-weighting (later ranks
# matter more); 60 is the canonical default from the original RRF paper.
RRF_K = int(os.getenv("RRF_K", "60"))


class EnsembleMemory:
    """Combine multiple TieredMemory-compatible backends via RRF fusion.

    Default ensemble = the two strongest COMPLEMENTARY backends from the matrix:
    atomic_fact (dense cosine over user-turn atomic facts) + mem0 (dense + BM25
    hybrid). Members use their own isolated collections (prefixed `ens-`) so the
    ensemble never reads the standalone backends' stores."""

    def __init__(self, user_id: str, agent_id: str = "lme-eval",
                 members: list[tuple[str, Any]] | None = None) -> None:
        self.user_id = user_id
        self.agent_id = agent_id
        if members is None:
            from src.atomic_fact_memory import AtomicFactMemory
            from src.mem0_backend_adapter import Mem0Adapter
            members = [
                ("atomic_fact", AtomicFactMemory(user_id=f"ens-af-{user_id}", agent_id=agent_id)),
                ("mem0", Mem0Adapter(user_id=f"ens-m0-{user_id}", agent_id=agent_id)),
            ]
        self._members = members

    def imprint(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        """Write to EVERY member (each applies its own extraction)."""
        ids: list[str] = []
        for name, m in self._members:
            try:
                ids.append(f"{name}:{m.imprint(content, metadata)}")
            except Exception as exc:  # noqa: BLE001 — one member failing must not sink the ensemble
                ids.append(f"{name}:<err {repr(exc)[:40]}>")
        return " ".join(ids)

    def query_context(self, query: str, k: int = 5, **_kwargs: Any) -> list[dict[str, Any]]:
        """Query each member, RRF-merge their ranked fact lists, return top-k.

        RRF score for a fact = Σ over members of 1/(RRF_K + rank_in_member).
        A fact retrieved by MULTIPLE members accumulates score → rises. Dedup is
        by normalized fact text (lowercased), so the same fact from two stores
        is one fused entry whose score reflects cross-store agreement."""
        fetch = max(k, 40)  # over-fetch so fusion has depth to work with
        fused: dict[str, dict[str, Any]] = {}
        for name, m in self._members:
            try:
                hits = m.query_context(query, k=fetch)
            except Exception:  # noqa: BLE001 — degrade to the other members
                hits = []
            for rank, h in enumerate(hits):
                content = (h.get("content") or h.get("summary") or h.get("episode") or "").strip()
                if not content:
                    continue
                key = content.lower()
                rr = 1.0 / (RRF_K + rank + 1)
                if key in fused:
                    fused[key]["score"] += rr
                    fused[key]["sources"].append(name)
                else:
                    fused[key] = {"content": content, "score": rr,
                                  "sources": [name], "metadata": h.get("metadata", {})}
        ranked = sorted(fused.values(), key=lambda x: x["score"], reverse=True)
        return ranked[:k]
