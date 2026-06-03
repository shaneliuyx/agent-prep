# src/three_tier_memory.py — Phase 7 wrapper (~120 LOC)
"""Three-tier memory: L1 (guild) + L2 (EverCore or Qdrant) + L3 (HyperMem).

Extends W3.5.8's TieredMemory with query_relations() for multi-entity
intersection queries. Imprint path stays single (writes to L2 always;
typed-edge extraction to L3 happens in the consolidation pipeline,
Phase 8). Read path is split: short queries → L2; multi-entity
intersection queries → L3.
"""
from __future__ import annotations

import os
from typing import Any

import httpx

from src.tiered_memory_qdrant import TieredMemory, TieredMemoryConfig


class ThreeTierMemory(TieredMemory):
    """L1 (guild) + L2 (Qdrant) + L3 (HyperMem) wrapper.

    Phase 8's consolidation pipeline writes typed hyperedges to L3
    alongside the existing L2 imprints. Phase 9's benchmark queries
    L3 for the multi-entity-intersection subset of LongMemEval
    questions (temporal-reasoning + some knowledge-update axes).
    """

    def __init__(
        self,
        user_id: str,
        agent_id: str = "lme-eval",
        config: TieredMemoryConfig | None = None,
        hypermem_url: str = "http://localhost:1996",
    ) -> None:
        super().__init__(user_id=user_id, agent_id=agent_id, config=config)
        self._hypermem = httpx.Client(base_url=hypermem_url, timeout=30.0)
        # L2 = atomic-fact store (same engine as the `atomic_fact` backend), NOT
        # the inherited raw-scroll embed. TieredMemory.imprint embeds whole
        # content as ONE point; fed a session scroll it stores a ~4 KB blob, and
        # the reader truncates each memory to 400 chars — so only the session
        # opening survives (measured: three_tier returned just the blazer → 1).
        # Delegating L2 to AtomicFactMemory gives per-fact, user-turn-filtered
        # memories in a per-user `af_{user_id}` collection (no shared-namespace
        # residue). L3 (HyperMem, below) keeps its real job: multi-entity
        # relation intersection via query_relations().
        from src.atomic_fact_memory import AtomicFactMemory
        self._l2 = AtomicFactMemory(user_id=user_id, agent_id=agent_id)

    def imprint(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        """L2 write — atomic facts (user-turn extraction), not a raw-scroll blob."""
        return self._l2.imprint(content, metadata)

    def query_context(self, query: str, k: int = 5, **_kwargs: Any) -> list[dict[str, Any]]:
        """L2 read — cosine top-k over atomic facts. Multi-entity relation
        queries go through query_relations() (L3), not this path."""
        return self._l2.query_context(query, k=k)

    def query_relations(
        self,
        intersection: list[dict[str, Any]],
        return_type: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Multi-entity intersection query against L3.

        intersection: list of node specifications, e.g.
            [{"node": {"type": "project", "id": "payments"}},
             {"node": {"type": "tech",    "id": "postgres"}}]
        return_type: the node-type to return (e.g., "user")
        """
        payload = {
            "intersection": intersection,
            "return_type": return_type,
            "limit": limit,
            "user_id": self.user_id,
        }
        r = self._hypermem.post("/api/v1/query/relations", json=payload)
        r.raise_for_status()
        return r.json().get("results", []) or []

    def close(self) -> None:
        """Clean up HTTP client alongside parent's cleanup."""
        self._hypermem.close()
        # parent's close() handles the rest (Qdrant, etc.)
        if hasattr(super(), "close"):
            super().close()
