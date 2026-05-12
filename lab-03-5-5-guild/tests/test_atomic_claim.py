import asyncio
import pytest

from src.guild_client import GuildClient, is_accept_winner


@pytest.mark.asyncio
async def test_atomic_claim_exactly_one_winner() -> None:
    async with GuildClient(agent_id="seed") as gc:
        quest_id = await gc.quest_post(
            subject="atomic-claim test: only one winner",
            campaign="race-test",
        )

    async def try_claim(agent_id: str) -> str:
        async with GuildClient(agent_id=agent_id) as gc:
            return await gc.quest_accept(quest_id)

    results = await asyncio.gather(
        try_claim("agent_a"),
        try_claim("agent_b"),
    )

    winners = sum(1 for r in results if is_accept_winner(r))
    assert winners == 1, f"expected 1 winner, got {winners}; results={results}"
