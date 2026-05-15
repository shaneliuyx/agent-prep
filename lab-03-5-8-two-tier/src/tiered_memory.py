"""TieredMemory — single facade over guild (operational) + EverCore (semantic).

Agents call this class; they never talk to either backend directly.
This is the seam that makes swapping backends cheap — change the
backend client, keep the orchestrator API stable.

Identity model — two-layer (load-bearing for the chapter's thesis):
  - `agent_id`  — Python-side persona label, per-instance. Lives in
    guild's session-scoped (anonymous) connection AND in EverCore
    imprint metadata. Used for attribution + audit, NOT for isolation.
  - `user_id`   — EverCore tenant identity, shared across all agents
    in the same project. Defaults to env `LAB358_USER_ID` or "shared".
    All agents on the same project MUST share the same user_id so
    EverCore's per-user index makes their consolidated knowledge
    visible across agent boundaries — exactly the cross-agent recall
    behavior this lab is built to demonstrate.

The pedagogical lesson: in EverCore's model, "user" = tenant, not
persona. If each agent gets its own user_id, the two-tier architecture
silently degrades into N isolated per-agent stores — and the
Phase 4 cross-agent recall demo returns 0 memories.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx

from src.guild_client import GuildClient, is_accept_winner


@dataclass
class TieredMemoryConfig:
    evercore_base_url: str = "http://localhost:1995"
    evercore_timeout_s: float = 30.0


class TieredMemory:
    """Operational + semantic memory facade.

    Operational queries (post_task / claim_task / complete_task) route to
    guild via the W3.5.5 GuildClient wrapper.
    Semantic queries (query_context / imprint) route to EverCore HTTP.
    Cross-tier consolidation is a separate batch job — not on the hot path.
    """

    def __init__(
        self,
        agent_id: str,
        user_id: str | None = None,
        config: TieredMemoryConfig | None = None,
    ) -> None:
        self.agent_id = agent_id
        # Tenant-scoped user_id is intentionally SHARED across agents in the
        # same lab so cross-agent recall via EverCore works. See module docstring.
        self.user_id = user_id or os.getenv("LAB358_USER_ID", "shared")
        self.config = config or TieredMemoryConfig()
        self._guild = GuildClient(agent_id=agent_id)
        self._http = httpx.Client(
            base_url=self.config.evercore_base_url,
            timeout=self.config.evercore_timeout_s,
        )

    async def __aenter__(self) -> "TieredMemory":
        await self._guild.__aenter__()  # auto-calls guild_session_start
        return self

    async def __aexit__(self, *exc) -> None:
        await self._guild.__aexit__(*exc)
        self._http.close()

    # ── Operational tier (guild) ──────────────────────────────────────

    async def post_task(
        self,
        subject: str,
        spec: str | None = None,
        campaign: str | None = None,
    ) -> str:
        """Create a quest. Returns server-assigned QUEST-ID (e.g. 'QUEST-42')."""
        return await self._guild.quest_post(subject=subject, spec=spec, campaign=campaign)

    async def claim_task(self, quest_id: str) -> dict[str, Any]:
        """Atomically accept a quest. Returns {won: bool, response: str}.

        guild's quest_accept uses an atomic SQLite UPDATE WHERE owner IS NULL
        primitive; only one caller wins per QUEST-ID. Losers receive an
        'already claimed' text response — classify via is_accept_winner().
        """
        text = await self._guild.quest_accept(quest_id=quest_id)
        return {"won": is_accept_winner(text), "response": text}

    async def complete_task(self, quest_id: str, report: str) -> str:
        """Mark quest fulfilled. `report` is REQUIRED by guild's schema."""
        return await self._guild.quest_fulfill(quest_id=quest_id, report=report)

    async def list_closed_quests(self, campaign: str | None = None) -> str:
        """Raw text listing of done-status quests (parse caller-side).

        guild has NO scroll_list_closed primitive (W3.5.5 §1.3 BCJ). Closed
        quests are queried via quest_list(status='done'); per-quest scroll
        text is then fetched via quest_scroll(quest_id).
        """
        return await self._guild.quest_list(status="done", campaign=campaign)

    async def get_scroll(self, quest_id: str) -> str:
        """Fetch the journal + report scroll for a completed quest."""
        return await self._guild.quest_scroll(quest_id=quest_id)

    # ── Semantic tier (EverCore) ──────────────────────────────────────
    #
    # EverCore exposes a CONVERSATION-shaped API (POST /api/v1/memories with
    # role/timestamp/content messages), NOT an arbitrary key-value imprint
    # API. We adapt by storing each consolidated fact as a single
    # assistant-role message under the agent's user_id, and parse search
    # responses out of the `data.episodes` array. See W3.5.8 §2.1 walkthrough
    # for the why-this-shape discussion.

    def _now_ms(self) -> int:
        import time
        return int(time.time() * 1000)

    def query_context(
        self,
        query: str,
        k: int = 5,
        min_confidence: float = 0.0,
        type_filter: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic recall — what do we know about <query>?

        Read-time application of write-time confidence + type signals
        (Batchelor-Manning 2026 forms #5 + #6). `min_confidence` filters
        out memories whose `quality_score` is below threshold;
        `type_filter` restricts to a subset of `type` values in the
        episode/memcell metadata. Over-fetches 3× k before filtering so
        results still return up to k hits when low-quality entries are
        scattered.

        Filter is `user_id=self.user_id` (SHARED tenant identity) so this
        agent sees memories imprinted by ANY agent on the same lab.
        Returns episode dicts; each carries `summary` / `episode` / `score`
        plus normalised `content` per OpenAPI schema.
        """
        r = self._http.post(
            "/api/v1/memories/search",
            json={
                "query": query,
                "top_k": k * 3 if min_confidence > 0 or type_filter else k,
                "filters": {"user_id": self.user_id},
            },
        )
        r.raise_for_status()
        data = r.json().get("data", {})
        episodes = data.get("episodes", []) or []
        for e in episodes:
            e.setdefault("content", e.get("summary") or e.get("episode") or "")
        # Read-time filters on write-time signals
        out = []
        for e in episodes:
            score = float(e.get("quality_score") or 1.0)
            if score < min_confidence:
                continue
            etype = e.get("type")
            if type_filter and etype and etype not in type_filter:
                continue
            out.append(e)
            if len(out) >= k:
                break
        return out

    def imprint(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        """Write a consolidated fact into long-term memory.

        EverCore's POST /api/v1/memories pipeline is conversation-shaped:
        accumulates messages, runs LLM boundary detection, only extracts a
        memcell when the LLM judges an episode boundary has occurred. Single
        isolated messages are stored as `accumulated` and never become
        searchable. To make consolidated facts visible to search:

          1. Wrap each fact as a 2-turn synthetic conversation
             (user "what about <subject>?" + assistant "<fact>") so the
             session has both a query side and an answer side.
          2. POST to /api/v1/memories with a unique session_id per fact.
          3. Immediately POST /api/v1/memories/flush with the SAME session_id
             — flush bypasses LLM boundary detection and forces memcell
             creation (`flush=True` short-circuits `_detect_boundaries`
             in EverCore's conv_memcell_extractor).

        Returns the session_id used (becomes the memcell anchor; one
        memcell per imprint call).
        """
        # Session_id uniqueness drives 1-imprint = 1-episode; using the
        # quest_id keeps it auditable. Fallback for non-quest imprints.
        session_id = (metadata or {}).get("quest_id") or f"imp-{self._now_ms()}"
        subject = (metadata or {}).get("subject") or "this topic"
        now_ms = self._now_ms()
        body = {
            "user_id": self.user_id,
            "session_id": session_id,
            "messages": [
                {
                    "role": "user",
                    "timestamp": now_ms,
                    "content": f"What do we know about {subject}?",
                },
                {
                    "role": "assistant",
                    "timestamp": now_ms + 1,
                    "content": content,
                },
            ],
        }
        r = self._http.post("/api/v1/memories", json=body)
        r.raise_for_status()
        # Force boundary close so the memcell extracts immediately
        # rather than waiting for LLM-judged conversational boundary
        # (which may never fire for single-fact imprints).
        rf = self._http.post(
            "/api/v1/memories/flush",
            json={"user_id": self.user_id, "session_id": session_id},
        )
        rf.raise_for_status()
        return session_id