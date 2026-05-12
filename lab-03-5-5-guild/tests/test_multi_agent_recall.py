"""5 representative tests from the 15-Q multi-agent recall benchmark."""
import asyncio
import uuid

import pytest

from src.guild_client import GuildClient, is_accept_winner


def unique_subject(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest.mark.asyncio
async def test_01_same_agent_quest_appears_in_listing() -> None:
    subj = unique_subject("01-list")
    async with GuildClient(agent_id="alice") as gc:
        quest_id = await gc.quest_post(subject=subj, campaign="recall-bench")
        listing = await gc.quest_list(campaign="recall-bench")
        assert quest_id in listing, f"{quest_id} not in listing: {listing!r}"


@pytest.mark.asyncio
async def test_02_same_agent_journal_round_trip() -> None:
    subj = unique_subject("02-journal")
    async with GuildClient(agent_id="alice") as gc:
        quest_id = await gc.quest_post(subject=subj, campaign="recall-bench")
        await gc.quest_accept(quest_id)
        await gc.quest_journal(quest_id, "alice's notes on the test")
        scroll = await gc.quest_scroll(quest_id)
        assert "alice's notes" in scroll


@pytest.mark.asyncio
async def test_06_agent_b_reads_agent_a_journal() -> None:
    subj = unique_subject("06-handoff")
    async with GuildClient(agent_id="agent_a") as gc:
        quest_id = await gc.quest_post(subject=subj, campaign="recall-bench")
        await gc.quest_accept(quest_id)
        await gc.quest_journal(quest_id, "agent_a's handoff note")
        await gc.quest_fulfill(
            quest_id,
            report="commit handoff-stub; files: tests/test_06.py; complete",
        )

    async with GuildClient(agent_id="agent_b") as gc:
        scroll = await gc.quest_scroll(quest_id)
        assert "agent_a's handoff note" in scroll


@pytest.mark.asyncio
async def test_11_parallel_claim_exactly_one_winner() -> None:
    subj = unique_subject("11-race")
    async with GuildClient(agent_id="seed") as gc:
        quest_id = await gc.quest_post(subject=subj, campaign="recall-bench")

    async def try_claim(agent_id: str) -> str:
        async with GuildClient(agent_id=agent_id) as gc:
            return await gc.quest_accept(quest_id)

    a, b = await asyncio.gather(try_claim("alice"), try_claim("bob"))
    winners = sum(1 for r in (a, b) if is_accept_winner(r))
    assert winners == 1, f"expected 1 winner, got {winners}; a={a!r} b={b!r}"


@pytest.mark.asyncio
async def test_15_quest_scroll_contains_fulfill_report() -> None:
    subj = unique_subject("15-fulfill-report")
    async with GuildClient(agent_id="alice") as gc:
        quest_id = await gc.quest_post(subject=subj, campaign="recall-bench")
        await gc.quest_accept(quest_id)
        await gc.quest_fulfill(
            quest_id,
            report="commit abc123; files: src/main.py; no remaining issues",
        )
        scroll = await gc.quest_scroll(quest_id)
        assert "abc123" in scroll
