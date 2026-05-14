"""TieredMemory — single facade over guild (operational) + EverCore (semantic).

Agents call this class; they never talk to either backend directly.
This is the seam that makes swapping backends cheap — change the
backend client, keep the orchestrator API stable.

Identity model: one TieredMemory instance = one agent stream
(guild's MCP session is anonymous; the agent_id is a Python-side
label propagated into EverCore imprint metadata only).
"""
from __future__ import annotations

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
        config: TieredMemoryConfig | None = None,
    ) -> None:
        self.agent_id = agent_id
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

    def query_context(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Semantic recall — what do we know about <query>?

        Returns episode dicts from EverCore's hybrid search; each dict has
        at minimum `summary` / `episode` / `score` (per OpenAPI schema).
        Caller can read `m['summary']` or `m['episode']` for content.
        """
        r = self._http.post(
            "/api/v1/memories/search",
            json={
                "query": query,
                "top_k": k,
                "filters": {"user_id": self.agent_id},
            },
        )
        r.raise_for_status()
        data = r.json().get("data", {})
        episodes = data.get("episodes", []) or []
        # Normalize: expose `content` field for chapter-level call sites.
        for e in episodes:
            e.setdefault("content", e.get("summary") or e.get("episode") or "")
        return episodes

    def imprint(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        """Write a consolidated fact into long-term memory.

        EverCore's POST /api/v1/memories shape: store the fact as one
        assistant-role message. Returns the request-scoped session_id we
        used (EverCore's own memory_id is assigned server-side and surfaced
        on subsequent search; we don't see it directly in the add response).
        """
        session_id = (metadata or {}).get("quest_id") or "consolidation"
        body = {
            "user_id": self.agent_id,
            "session_id": session_id,
            "messages": [
                {
                    "role": "assistant",
                    "timestamp": self._now_ms(),
                    "content": content,
                }
            ],
        }
        r = self._http.post("/api/v1/memories", json=body)
        r.raise_for_status()
        return session_id