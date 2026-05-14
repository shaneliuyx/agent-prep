import uuid

import pytest

from src.consolidation import consolidate
from src.tiered_memory import TieredMemory


# Each test gets its own campaign tag so guild's append-only quest store
# doesn't leak prior-run residue across tests. Without this, max_batch=10
# fills with old QUEST-Ns and the freshly-seeded quest is never reached.
def _fresh_campaign() -> str:
    return f"test-w358-{uuid.uuid4().hex[:8]}"


async def _seed_completed_quest(
    tm: TieredMemory, campaign: str, subject: str, report: str
) -> str:
    quest_id = await tm.post_task(subject=subject, campaign=campaign)
    claim = await tm.claim_task(quest_id)
    assert claim["won"], f"Could not claim {quest_id}: {claim['response']}"
    await tm.complete_task(quest_id, report=report)
    return quest_id


@pytest.mark.asyncio
async def test_consolidation_imprints_completed_scrolls():
    campaign = _fresh_campaign()
    async with TieredMemory(agent_id="test_agent") as tm:
        await _seed_completed_quest(
            tm,
            campaign=campaign,
            subject="deploy-via-terraform",
            report="deployed via terraform; ran apply; got 200; verified VPC peering",
        )
        result = await consolidate(tm, max_batch=10, campaign=campaign)
        assert result.scrolls_imprinted >= 1


@pytest.mark.asyncio
async def test_consolidation_idempotent_on_second_run():
    campaign = _fresh_campaign()
    async with TieredMemory(agent_id="test_agent") as tm:
        await _seed_completed_quest(
            tm,
            campaign=campaign,
            subject="check-auth-tokens",
            report="auth tokens expire after 30min; got 401 with stale token",
        )
        first = await consolidate(tm, max_batch=10, campaign=campaign)
        second = await consolidate(tm, max_batch=10, campaign=campaign)
        # First run imprints; second run should imprint zero (dedup table).
        assert first.scrolls_imprinted >= 1
        assert second.scrolls_imprinted == 0


@pytest.mark.asyncio
async def test_consolidation_skips_low_value_scrolls():
    campaign = _fresh_campaign()
    async with TieredMemory(agent_id="test_agent") as tm:
        await _seed_completed_quest(
            tm,
            campaign=campaign,
            subject="debug-session",
            report="trying things; not sure yet; logged some stuff",
        )
        result = await consolidate(tm, max_batch=10, campaign=campaign)
        # Low-value scroll should be SKIPped by summarizer.
        assert result.scrolls_skipped >= 1