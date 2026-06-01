"""Python wrapper over guild's MCP stdio interface (v3, schema-verified).

All method signatures aligned with guild's authoritative MCP input schemas
(probed live via session.list_tools()[i].inputSchema). The server rejects
extra args with 'unexpected additional properties'.

Two non-obvious facts about guild's MCP interface (full architectural notes
in chapter §2.1 docstring + RESULTS.md BCJ Entry 5):

1. RESPONSES ARE TEXT-ONLY. CallToolResult.structuredContent is always None.
   Wrappers regex-parse identifiers + keyword-match status from the text.

2. AGENT IDENTITY IS SESSION-SCOPED, NOT CALL-SCOPED. MCP schema rejects
   per-call agent identity (no owner / agent / agent_id args). One MCP
   connection = one anonymous agent stream. The agent_id constructor arg
   is a Python-side label only.
"""
from __future__ import annotations

import re
from contextlib import AsyncExitStack
from enum import Enum
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters


QUEST_ID_RE = re.compile(r"QUEST-\d+")


class QuestStatus(str, Enum):
    """Valid status filter values for quest_list (per MCP inputSchema)."""

    NEXT = "next"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    DONE = "done"


def is_accept_winner(resp: str) -> bool:
    """Classify a quest_accept response text as winner vs race-loser.

    guild's MCP returns human-readable text; winners contain wording like
    'accepted' or 'claimed', race-losers contain 'already claimed' /
    'already accepted'. Substring-match because there is no structured
    status field on the response.
    """
    low = resp.lower()
    return ("accept" in low or "claim" in low) and "already" not in low


class GuildClient:
    """One Python MCP-stdio session against guild."""

    def __init__(
        self,
        agent_id: str,
        command: str = "guild",
        args: tuple[str, ...] = ("mcp", "serve"),
    ) -> None:
        self.agent_id = agent_id
        self._params = StdioServerParameters(command=command, args=list(args))
        self._stack = AsyncExitStack()
        self._session: ClientSession | None = None

    async def __aenter__(self) -> "GuildClient":
        read, write = await self._stack.enter_async_context(
            stdio_client(self._params)
        )
        session = ClientSession(read, write)
        await self._stack.enter_async_context(session)
        await session.initialize()
        self._session = session
        # MANDATORY per guild's MCP contract.
        await session.call_tool("guild_session_start", arguments={})
        return self

    async def __aexit__(self, *exc) -> None:
        await self._stack.aclose()
        self._session = None

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _text(result) -> str:
        content = result.content or []
        return content[0].text if content else ""

    @staticmethod
    def _parse_quest_id(text: str) -> str:
        m = QUEST_ID_RE.search(text)
        return m.group(0) if m else ""

    async def _call(self, tool: str, **kwargs: Any) -> str:
        """Call an MCP tool with None-args filtered out; return response text."""
        if self._session is None:
            raise RuntimeError("GuildClient must be used as an async context manager")
        args = {k: v for k, v in kwargs.items() if v is not None}
        result = await self._session.call_tool(tool, arguments=args)
        return self._text(result)

    # ── Quest operations ──────────────────────────────────────────────

    async def quest_post(
        self,
        subject: str,
        spec: str | None = None,
        campaign: str | None = None,
        priority: str | None = None,
    ) -> str:
        """Create a quest. Returns server-assigned QUEST_ID parsed from text."""
        text = await self._call(
            "quest_post",
            subject=subject,
            spec=spec,
            campaign=campaign,
            priority=priority,
        )
        return self._parse_quest_id(text)

    async def quest_accept(self, quest_id: str) -> str:
        """Atomically claim. See is_accept_winner() to classify the response."""
        return await self._call("quest_accept", quest_id=quest_id)

    async def quest_journal(self, quest_id: str, text: str) -> str:
        return await self._call("quest_journal", quest_id=quest_id, text=text)

    async def quest_fulfill(self, quest_id: str, report: str) -> str:
        """Complete a quest. `report` is REQUIRED by guild's schema."""
        return await self._call("quest_fulfill", quest_id=quest_id, report=report)

    async def quest_scroll(self, quest_id: str) -> str:
        return await self._call("quest_scroll", quest_id=quest_id)

    async def quest_list(
        self,
        status: str | QuestStatus | None = None,
        campaign: str | None = None,
    ) -> str:
        """List quests with optional filters.

        Valid status values per MCP schema: next | in_progress | blocked | done.
        Use QuestStatus enum for IDE completion.
        """
        status_val = status.value if isinstance(status, QuestStatus) else status
        return await self._call("quest_list", status=status_val, campaign=campaign)

    # ── Session orchestration ────────────────────────────────────────

    async def session_start(self) -> str:
        """Manually re-issue guild_session_start (already called in __aenter__)."""
        return await self._call("guild_session_start")
