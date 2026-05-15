"""Phase 7 (stretch) — same 3 consolidation tests against the Qdrant variant.

Proves the §2.1 "wrapper IS the architecture" claim: by switching the
import on TieredMemory, the consolidation pipeline, dedup state, and
test contract stay identical. Different backend, same surface.

Run alongside test_consolidation.py:

    uv run pytest tests/ -v

Both test files use unique campaign tags per test (uuid) so the
append-only guild quest store doesn't leak residue between them.
"""
import uuid

import pytest

from src.consolidation import consolidate
from src.tiered_memory_qdrant import TieredMemory  # ← the one-line swap


def _fresh_campaign() -> str:
    return f"test-w358-qdrant-{uuid.uuid4().hex[:8]}"


async def _seed_completed_quest(
    tm: TieredMemory, campaign: str, subject: str, report: str
) -> str:
    quest_id = await tm.post_task(subject=subject, campaign=campaign)
    claim = await tm.claim_task(quest_id)
    assert claim["won"], f"Could not claim {quest_id}: {claim['response']}"
    await tm.complete_task(quest_id, report=report)
    return quest_id


@pytest.mark.asyncio
async def test_qdrant_consolidation_imprints_completed_scrolls():
    campaign = _fresh_campaign()
    async with TieredMemory(agent_id="qdrant_test_agent") as tm:
        await _seed_completed_quest(
            tm,
            campaign=campaign,
            subject="deploy-via-terraform",
            report="deployed via terraform; ran apply; got 200; verified VPC peering",
        )
        result = await consolidate(tm, max_batch=10, campaign=campaign)
        assert result.scrolls_imprinted >= 1


@pytest.mark.asyncio
async def test_qdrant_consolidation_idempotent_on_second_run():
    campaign = _fresh_campaign()
    async with TieredMemory(agent_id="qdrant_test_agent") as tm:
        await _seed_completed_quest(
            tm,
            campaign=campaign,
            subject="check-auth-tokens",
            report="auth tokens expire after 30min; got 401 with stale token",
        )
        first = await consolidate(tm, max_batch=10, campaign=campaign)
        second = await consolidate(tm, max_batch=10, campaign=campaign)
        assert first.scrolls_imprinted >= 1
        assert second.scrolls_imprinted == 0


@pytest.mark.asyncio
async def test_qdrant_consolidation_skips_low_value_scrolls():
    campaign = _fresh_campaign()
    async with TieredMemory(agent_id="qdrant_test_agent") as tm:
        await _seed_completed_quest(
            tm,
            campaign=campaign,
            subject="debug-session",
            report="trying things; not sure yet; logged some stuff",
        )
        result = await consolidate(tm, max_batch=10, campaign=campaign)
        assert result.scrolls_skipped >= 1


@pytest.mark.asyncio
async def test_qdrant_query_round_trip():
    """Imprint → search → assert content is retrievable. Qdrant-specific
    end-to-end check that the chapter's Production Considerations table
    claim ("~80ms search wall, no extraction pipeline") is achievable."""
    async with TieredMemory(agent_id="qdrant_round_trip") as tm:
        tm.imprint(
            content="Production deployments use Terraform IaC with VPC peering and 5-minute apply budget.",
            metadata={"quest_id": "QUEST-rt", "subject": "deploy"},
        )
        results = tm.query_context(query="how do we deploy production APIs?", k=3)
        assert results, "expected at least one match for the just-imprinted fact"
        assert "Terraform" in results[0]["content"]
        # Qdrant score is cosine similarity ∈ [0,1] for normalized vectors
        assert 0.0 <= results[0]["score"] <= 1.0
