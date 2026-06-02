# src/mem0_backend_adapter.py — Phase 3 Mem0 adapter (~80 LOC)
"""Thin adapter wrapping mem0ai's SDK in the TieredMemory-like interface
used by the lab's existing eval driver. Single-tier ADD-only semantics —
Mem0 owns the fact-extraction + retrieval pipeline; this adapter just
translates call shapes.

Why ADAPTER instead of inheritance: Mem0's SDK isn't a drop-in subclass
of TieredMemory (different method names, different argument types).
A protocol-based adapter keeps the eval driver agnostic without forcing
Mem0 into our class hierarchy.
"""
from __future__ import annotations

import os
from typing import Any

from mem0 import Memory


class Mem0Adapter:
    """TieredMemory-compatible facade over mem0ai's Memory client."""

    def __init__(self, user_id: str, agent_id: str = "lme-eval") -> None:
        self.user_id = user_id
        self.agent_id = agent_id
        # Mem0's default config uses OpenAI for extraction + Qdrant for storage.
        # Override to use local oMLX endpoint for LLM, point at lab's Qdrant.
        config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": os.getenv("MODEL_HAIKU", "gemma-4-26B-A4B-it-heretic-4bit"),
                    "openai_base_url": os.getenv("OMLX_BASE_URL"),
                    "api_key": os.getenv("OMLX_API_KEY", "dummy"),
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": os.getenv("MODEL_EMBED", "bge-m3-mlx-fp16"),
                    "openai_base_url": os.getenv("OMLX_BASE_URL"),
                    "api_key": os.getenv("OMLX_API_KEY", "dummy"),
                    # bge-m3 = 1024-dim. mem0 derives the Qdrant collection dim
                    # from the embedder; without this it defaults to OpenAI's
                    # 1536 and Qdrant rejects the 1024 vectors on add().
                    "embedding_dims": 1024,
                },
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "host": "localhost",
                    "port": 6333,
                    "collection_name": f"mem0_{user_id}",
                    # bge-m3 emits 1024-dim vectors; mem0 defaults to 1536
                    # (OpenAI text-embedding-3) and creates the collection at
                    # that dim, causing "expected dim 1536, got 1024" on add().
                    "embedding_model_dims": 1024,
                },
            },
        }
        self._client = Memory.from_config(config)

    def imprint(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        """Write content as atomic facts. Mem0's add() extracts facts
        from a messages list — we synthesize a 1-turn user message
        carrying the content, since lab's per-session imprint pattern
        passes pre-summarized strings, not multi-turn dialogues.
        """
        messages = [{"role": "user", "content": content}]
        result = self._client.add(messages, user_id=self.user_id, metadata=metadata or {})
        # Mem0 returns a dict with 'results' = list of {memory, event, ...}
        # Return the first memory's id (or a synthetic one if Mem0's response shape varies)
        results = result.get("results", []) if isinstance(result, dict) else []
        return str(results[0].get("id", "")) if results else ""

    def query_context(
        self, query: str, k: int = 5, **_kwargs: Any
    ) -> list[dict[str, Any]]:
        """Retrieve top-k facts via Mem0's multi-signal search.
        Translates Mem0's response shape to the lab's expected shape
        (each result has at minimum a `content` field readable by the
        eval driver's reader-prompt builder).
        """
        # mem0 >= 2.x: user_id must go in filters=, not as a top-level kwarg.
        hits = self._client.search(query=query, filters={"user_id": self.user_id}, limit=k)
        # Mem0 may return list-of-dicts OR {'results': [...]} depending on version
        if isinstance(hits, dict):
            hits = hits.get("results", []) or []
        out: list[dict[str, Any]] = []
        for h in hits:
            content = h.get("memory") or h.get("content") or h.get("text", "")
            out.append({
                "content": content,
                "score": h.get("score", 0.0),
                "metadata": h.get("metadata", {}),
            })
        return out
