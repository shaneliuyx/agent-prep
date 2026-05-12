import pytest
from src.guild_client import GuildClient


@pytest.mark.asyncio
async def test_quest_lifecycle_round_trip() -> None:
    async with GuildClient(agent_id="alice") as gc:
        quest_id = await gc.quest_post(
            subject="smoke-test: round-trip the full quest lifecycle",
            spec="Verify Python wrapper covers post/accept/journal/fulfill/scroll.",
            campaign="lab-03-5-5",
        )
        assert quest_id.startswith("QUEST-"), f"unexpected: {quest_id!r}"

        accept_resp = await gc.quest_accept(quest_id)
        low = accept_resp.lower()
        assert ("accept" in low or "claim" in low), \
            f"unexpected accept response: {accept_resp!r}"

        await gc.quest_journal(quest_id, "smoke-test journal text")

        fulfill_resp = await gc.quest_fulfill(
            quest_id,
            report="commit smoke-test-stub; files: tests/test_guild_client.py; no remaining issues",
        )
        assert "fulfill" in fulfill_resp.lower()

        scroll = await gc.quest_scroll(quest_id)
        assert "smoke-test journal text" in scroll, \
            f"journal entry not in scroll: {scroll!r}"
