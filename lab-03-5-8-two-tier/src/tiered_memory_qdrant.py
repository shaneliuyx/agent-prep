"""TieredMemory — Qdrant variant. Drop-in replacement for the EverCore version.

Operational tier (guild) is identical to src/tiered_memory.py — same
post_task / claim_task / complete_task / list_closed_quests / get_scroll
methods delegating to GuildClient.

Semantic tier swaps EverCore's HTTP API for raw Qdrant + bge-m3 embeddings
served by the local oMLX endpoint. The contract on TieredMemory.imprint()
and TieredMemory.query_context() stays identical so a one-line import
change in the demo + tests switches backends.

Trade-off vs EverCore variant (measured 2026-05-14, see W3.5.8 Production
Considerations):
  + 20-30x faster imprints (no LLM extraction pipeline; just embed + upsert)
  + 3-5x faster searches (pure HNSW vs hybrid Mongo+ES+Milvus)
  + 7x lighter infrastructure (1 container vs 7)
  - Lose atomic_fact decomposition (you store the consolidated fact as-is)
  - Lose profile aggregation (maintain a per-user profile table separately)
  - Lose hybrid retrieval (Qdrant native filtering covers ~95% of cases)

Identity model is identical: agent_id is per-instance Python label;
user_id (defaults to LAB358_USER_ID env or "shared") is the SHARED
tenant identity gating cross-agent recall. See W3.5.8 BCJ Entry 12.
"""
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Any

import httpx
from openai import OpenAI

from src.guild_client import GuildClient, is_accept_winner


COLLECTION = "lab358_memories"
EMBED_DIMS = 1024  # bge-m3-mlx-fp16


@dataclass
class TieredMemoryConfig:
    qdrant_base_url: str = "http://localhost:6333"
    qdrant_timeout_s: float = 10.0


class TieredMemory:
    """Drop-in replacement for src/tiered_memory.TieredMemory.

    Same class name + method signatures. Underlying semantic backend is
    Qdrant + bge-m3 (no extraction pipeline, no conversation-shape
    gymnastics, no boundary detection).
    """

    def __init__(
        self,
        agent_id: str,
        user_id: str | None = None,
        config: TieredMemoryConfig | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.user_id = user_id or os.getenv("LAB358_USER_ID", "shared")
        self.config = config or TieredMemoryConfig()
        self._guild = GuildClient(agent_id=agent_id)
        self._http = httpx.Client(
            base_url=self.config.qdrant_base_url,
            timeout=self.config.qdrant_timeout_s,
        )
        self._llm = OpenAI(
            base_url=os.getenv("OMLX_BASE_URL"),
            api_key=os.getenv("OMLX_API_KEY"),
        )
        self._embed_model = os.getenv("MODEL_EMBED", "bge-m3-mlx-fp16")
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Idempotent collection bootstrap. PUT to /collections/<name> returns
        200 on create, error on already-exists. Either is fine — we swallow
        409-class errors and let real failures bubble up at first imprint.
        """
        try:
            r = self._http.put(
                f"/collections/{COLLECTION}",
                json={"vectors": {"size": EMBED_DIMS, "distance": "Cosine"}},
            )
            # 200 = created; 4xx with "already exists" body = OK; other 4xx/5xx = real failure
            if r.status_code >= 400 and "already exists" not in r.text:
                r.raise_for_status()
        except httpx.HTTPError:
            # Collection probably exists — let the next operation fail
            # loudly if Qdrant is actually down.
            pass

    def _embed(self, text: str) -> list[float]:
        """Embed via the local oMLX bge-m3 endpoint. Returns 1024-d vector."""
        resp = self._llm.embeddings.create(model=self._embed_model, input=text)
        return resp.data[0].embedding

    async def __aenter__(self) -> "TieredMemory":
        await self._guild.__aenter__()
        return self

    async def __aexit__(self, *exc) -> None:
        await self._guild.__aexit__(*exc)
        self._http.close()

    # ── Operational tier (guild) — identical to EverCore variant ─────

    async def post_task(
        self,
        subject: str,
        spec: str | None = None,
        campaign: str | None = None,
    ) -> str:
        return await self._guild.quest_post(subject=subject, spec=spec, campaign=campaign)

    async def claim_task(self, quest_id: str) -> dict[str, Any]:
        text = await self._guild.quest_accept(quest_id=quest_id)
        return {"won": is_accept_winner(text), "response": text}

    async def complete_task(self, quest_id: str, report: str) -> str:
        return await self._guild.quest_fulfill(quest_id=quest_id, report=report)

    async def list_closed_quests(self, campaign: str | None = None) -> str:
        return await self._guild.quest_list(status="done", campaign=campaign)

    async def get_scroll(self, quest_id: str) -> str:
        return await self._guild.quest_scroll(quest_id=quest_id)

    # ── Semantic tier (Qdrant) ────────────────────────────────────────

    def imprint(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        """Embed + upsert one consolidated fact. ~150ms wall-clock.

        No conversation shape, no boundary detection, no flush dance.
        Returns the Qdrant point ID for audit/dedup.
        """
        point_id = str(uuid.uuid4())
        vector = self._embed(content)
        payload: dict[str, Any] = {
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "content": content,
        }
        if metadata:
            payload.update(metadata)
        r = self._http.put(
            f"/collections/{COLLECTION}/points",
            json={"points": [{"id": point_id, "vector": vector, "payload": payload}]},
        )
        r.raise_for_status()
        return point_id

    def query_context(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Cosine-nearest top-k filtered by SHARED user_id.

        Returns list of dicts with keys `content`, `score`, plus all
        payload fields (quest_id, agent_id, subject, quality_score, ...).
        """
        vector = self._embed(query)
        r = self._http.post(
            f"/collections/{COLLECTION}/points/search",
            json={
                "vector": vector,
                "limit": k,
                "filter": {
                    "must": [{"key": "user_id", "match": {"value": self.user_id}}]
                },
                "with_payload": True,
            },
        )
        r.raise_for_status()
        hits = r.json().get("result", []) or []
        return [
            {
                "content": hit["payload"]["content"],
                "score": hit["score"],
                **hit["payload"],
            }
            for hit in hits
        ]
