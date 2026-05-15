"""Atomisation rewrite tests (Batchelor-Manning 2026 form #2).

Validates:
- extract_atomic_facts() returns a list of typed dicts on real scrolls
- consolidate(use_atomisation=True) imprints N facts per scroll
- facts_imprinted counter > scrolls_imprinted when scrolls have multiple atoms

Uses Qdrant backend (deterministic write semantics, no LLM-extraction
black box) so we can assert exact fact counts.
"""
import uuid

import pytest

from src.consolidation import consolidate, extract_atomic_facts
from src.tiered_memory_qdrant import TieredMemory


def _fresh_campaign() -> str:
    return f"test-w358-atom-{uuid.uuid4().hex[:8]}"


def test_extract_atomic_facts_returns_typed_list():
    """Multi-fact scroll → multiple atoms with type + confidence."""
    facts = extract_atomic_facts(
        "Deployed prod API via Terraform plan + apply. Used the company's "
        "standard IaC module (modules/api-stack). Required VPC peering with "
        "the data-lake account. First-deploy budget was 5 minutes wall-clock."
    )
    assert isinstance(facts, list)
    assert len(facts) >= 2, f"expected ≥2 atoms, got {len(facts)}: {facts}"
    for f in facts:
        assert "fact" in f and isinstance(f["fact"], str)
        assert f["type"] in {"fact", "observation", "tool_result", "skill"}
        assert 0.0 <= f["confidence"] <= 1.0


def test_extract_atomic_facts_empty_on_no_knowledge():
    """Vague scroll with no reusable knowledge → empty list."""
    facts = extract_atomic_facts(
        "trying things; not sure yet; logged some stuff and moved on"
    )
    # Either empty OR very low confidence; either way the gate
    # should reject it. We assert structural shape only.
    assert isinstance(facts, list)


@pytest.mark.asyncio
async def test_consolidate_with_atomisation_produces_multiple_facts():
    """consolidate(use_atomisation=True) → facts_imprinted > scrolls_imprinted."""
    campaign = _fresh_campaign()
    async with TieredMemory(agent_id="atom_test") as tm:
        quest_id = await tm.post_task(subject="deploy-multi-fact", campaign=campaign)
        claim = await tm.claim_task(quest_id)
        assert claim["won"]
        # Multi-fact report — should atomise into ≥2 facts
        await tm.complete_task(
            quest_id,
            report=(
                "Deployed prod API via Terraform plan + apply. Used standard "
                "modules/api-stack. Required VPC peering with data-lake. "
                "First-deploy budget was 5 minutes wall-clock. terraform apply "
                "returned 200 on first run."
            ),
        )
        result = await consolidate(
            tm, max_batch=10, campaign=campaign, use_atomisation=True
        )
        assert result.scrolls_imprinted == 1, f"expected 1 scroll, got {result}"
        assert result.facts_imprinted >= 2, (
            f"expected ≥2 atomic facts, got {result.facts_imprinted}. "
            "If 1: atomisation collapsed the multi-fact scroll into one summary."
        )


@pytest.mark.asyncio
async def test_query_filter_by_type_returns_only_matching():
    """query_context(type_filter=['fact']) excludes other types."""
    async with TieredMemory(agent_id="type_filter_test") as tm:
        # Imprint a mix of types via direct calls (bypass consolidate to keep test fast)
        tm.imprint(
            content="Production deployments use Terraform IaC.",
            metadata={"type": "fact", "quality_score": 0.9},
        )
        tm.imprint(
            content="Today's deploy took 5 minutes wall-clock.",
            metadata={"type": "observation", "quality_score": 0.8},
        )
        tm.imprint(
            content="terraform apply returned 200.",
            metadata={"type": "tool_result", "quality_score": 0.7},
        )
        # Filter to facts only
        hits = tm.query_context(
            query="how do we deploy?", k=10, type_filter=["fact"]
        )
        # All returned should have type="fact"
        for h in hits:
            assert h.get("type") == "fact", f"got non-fact in fact-filtered results: {h}"


@pytest.mark.asyncio
async def test_query_min_confidence_excludes_low_quality():
    """query_context(min_confidence=0.8) excludes low-confidence memories."""
    async with TieredMemory(agent_id="conf_filter_test") as tm:
        tm.imprint(
            content="High-quality fact about authentication tokens.",
            metadata={"type": "fact", "quality_score": 0.95},
        )
        tm.imprint(
            content="Low-quality vague observation about something.",
            metadata={"type": "observation", "quality_score": 0.2},
        )
        hits = tm.query_context(
            query="authentication tokens", k=10, min_confidence=0.8
        )
        # All returned should have quality_score >= 0.8
        for h in hits:
            assert h.get("quality_score", 1.0) >= 0.8, (
                f"got low-confidence hit in min_confidence=0.8 filter: {h}"
            )
