# src/router_memory.py — Phase 4 question-type router (~150 LOC)
"""Hybrid memory: routes write to 1-tier atomic-fact (always), routes read
to 1-tier OR 2-tier based on question_type.

Design choice: ALL writes go through atomic-fact path. The 2-tier backend
is queried READ-side for knowledge-update questions whose answer needs
dedup+supersede semantics that atomic-fact alone doesn't natively give.

This is a deliberate asymmetry: rather than maintain TWO write paths in sync
(harder + slower + reconciliation risk), the chapter chooses ONE canonical
write path and routes the read. If knowledge-update accuracy is weak,
the next experiment is dual-write (Phase 4.6 future work).
"""
from __future__ import annotations

import re
from typing import Any

from src.atomic_fact_memory import AtomicFactMemory
from src.tiered_memory_qdrant import TieredMemory, TieredMemoryConfig


class RouterMemory:
    """Hybrid backend with question-type-based read routing."""

    # LongMemEval question_type labels → backend tag
    # In production this comes from a classifier (rule-based regex + LLM
    # fallback). For the lab, we receive the label directly from the slice
    # data via the kwarg `question_type` on query_context().
    READ_ROUTE = {
        "single-session-user": "atomic_fact",
        "single-session-assistant": "atomic_fact",
        "single-session-preference": "atomic_fact",
        "multi-session": "atomic_fact",
        "knowledge-update": "tiered_2tier",     # 2-tier wins via dedup+supersede
        "temporal-reasoning": "atomic_fact",     # graph-tier would be ideal; deferred
    }
    DEFAULT_ROUTE = "atomic_fact"

    def __init__(self, user_id: str, agent_id: str = "lme-eval") -> None:
        self.user_id = user_id
        self.agent_id = agent_id
        self._af = AtomicFactMemory(user_id=user_id, agent_id=agent_id)
        # Lazy-init 2-tier — only spin up if a read actually routes to it.
        self._tt: TieredMemory | None = None

    def _get_2tier(self) -> TieredMemory:
        if self._tt is None:
            self._tt = TieredMemory(
                user_id=self.user_id,
                agent_id=self.agent_id,
                config=TieredMemoryConfig(),
            )
        return self._tt

    def _classify(self, question_type: str | None, question: str) -> str:
        """Resolve which backend should serve this question.
        Prefer the explicit label (lab fast-path); fall back to a regex
        heuristic on the question text (production realism).
        """
        if question_type:
            # Strip _abs suffix (abstention overlay) — route by base type
            base_type = question_type.rsplit("_abs", 1)[0]
            return self.READ_ROUTE.get(base_type, self.DEFAULT_ROUTE)

        # Rule-based fallback when no label is provided
        q = question.lower()
        if re.search(r"\b(current|now|today|latest|most recent)\b", q):
            return "tiered_2tier"  # knowledge-update-shape heuristic
        if re.search(r"\b(when|how long ago|how many days|months|years)\b", q):
            return "atomic_fact"   # temporal-reasoning — atomic-fact + reader arithmetic
        return self.DEFAULT_ROUTE

    def imprint(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        """Write-side: atomic-fact path always.
        Decision rationale: dual-write doubles latency + introduces
        reconciliation risk (which backend is authoritative if they disagree?).
        Single write path keeps the architecture honest.
        """
        return self._af.imprint(content, metadata)

    def query_context(
        self,
        query: str,
        k: int = 5,
        question_type: str | None = None,
        **_kwargs: Any,
    ) -> list[dict[str, Any]]:
        route = self._classify(question_type, query)
        if route == "tiered_2tier":
            return self._get_2tier().query_context(query, k=k)
        return self._af.query_context(query, k=k)
