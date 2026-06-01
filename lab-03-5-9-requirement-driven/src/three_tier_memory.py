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
