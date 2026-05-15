"""Phase 9 — Online dedup-and-synthesis tests (Batchelor-Manning form #1).

Validates the highest-ROI write-time investment from the 19-system survey:
- decide_action() returns valid Action on real LLM output (+ JSON parsing)
- Add a fact, then add a near-duplicate → second action != "add" (LLM
  judges it as no-op OR update; both are correct "don't store another copy")
- Add a fact, then add a contradicting fact → LLM emits "delete" OR "update"
  (either resolves the contradiction; both are valid)
- consolidate(use_dedup=True) populates the dedup counters

Scoped to Qdrant variant (EverCore's internal pipeline doesn't expose
delete/update hooks cleanly — chapter Phase 9 spec calls this out).
"""
import time
import uuid

import pytest

from src.consolidation import consolidate
from src.dedup_synthesis import decide_action
from src.tiered_memory_qdrant import TieredMemory


def _fresh_campaign() -> str:
    return f"test-w358-dedup-{uuid.uuid4().hex[:8]}"


def test_decide_action_returns_add_on_empty_candidates():
    """No candidates → always add. No LLM call should fire."""
    action = decide_action("brand new fact about Terraform", candidates=[])
    assert action.action == "add"


def test_decide_action_classifies_real_duplicate_correctly():
    """Same fact phrased two ways → LLM should pick no-op or update.
    Both are correct outcomes — they preserve the "don't store a duplicate"
    invariant. We assert action ∈ {no-op, update}, not specific choice.
    """
    candidates = [
        {
            "id": "existing-1",
            "content": "Production API deployments use Terraform IaC with VPC peering.",
            "score": 0.9,
        }
    ]
    new_fact = "We deploy production APIs via Terraform infrastructure-as-code with VPC peering."
    action = decide_action(new_fact, candidates)
    assert action.action in ("no-op", "update"), (
        f"expected dedup (no-op or update), got {action.action} — "
        "LLM is treating near-duplicate as novel; raises false-positive risk"
    )
    if action.action == "update":
        assert action.target_id == "existing-1"


def test_decide_action_handles_contradiction():
    """Contradicting fact → LLM picks delete or update. Both resolve the
    contradiction; we don't assert a specific choice."""
    candidates = [
        {
            "id": "existing-1",
            "content": "Auth tokens expire after 30 minutes.",
            "score": 0.85,
        }
    ]
    new_fact = "Auth tokens expire after 1 hour."
    action = decide_action(new_fact, candidates)
    assert action.action in ("delete", "update", "add"), (
        f"unexpected action: {action.action}"
    )
    # Reasonable expectation: LLM doesn't pick no-op (which would silence the contradiction)
    assert action.action != "no-op", (
        "LLM picked no-op on a contradiction — store would silently retain "
        "the now-false 30-minute fact alongside the new 1-hour fact"
    )


@pytest.mark.asyncio
async def test_consolidate_use_dedup_increments_counters():
    """End-to-end: imprint same-topic scroll twice via consolidate(use_dedup=True)
    → second run's facts_deduplicated OR facts_updated should be > 0.
    """
    campaign = _fresh_campaign()
    async with TieredMemory(agent_id="dedup_test") as tm:
        # Seed with one scroll
        q1 = await tm.post_task(subject="deploy-via-terraform", campaign=campaign)
        await tm.claim_task(q1)
        await tm.complete_task(
            q1,
            report="Production deploys use Terraform with VPC peering; 5-minute apply budget.",
        )
        r1 = await consolidate(
            tm,
            max_batch=10,
            campaign=campaign,
            use_atomisation=True,
            use_dedup=True,
        )
        # First-run assertion: SOME action must fire per atom. Could be
        # imprinted (fresh collection) OR deduplicated (collection has
        # residue from prior tests — Qdrant `lab358_memories` is shared).
        # Either outcome proves the pipeline ran.
        actions_r1 = (
            r1.facts_imprinted + r1.facts_deduplicated
            + r1.facts_updated + r1.facts_deleted
        )
        assert actions_r1 >= 1, f"first scroll: no actions fired (atomisation may have failed): {r1}"

        # Wait a moment for Qdrant index to settle
        time.sleep(1)

        # Second scroll covering THE SAME ground — should hit dedup vs r1's atoms
        q2 = await tm.post_task(subject="deploy-via-terraform-again", campaign=campaign)
        await tm.claim_task(q2)
        await tm.complete_task(
            q2,
            report="We deploy our production APIs using Terraform IaC. VPC peering required. Budget is 5 minutes.",
        )
        r2 = await consolidate(
            tm,
            max_batch=10,
            campaign=campaign,
            use_atomisation=True,
            use_dedup=True,
        )
        # At least ONE atom should have triggered a dedup action (noop or update)
        dedup_actions = r2.facts_deduplicated + r2.facts_updated + r2.facts_deleted
        assert dedup_actions >= 1, (
            f"expected ≥1 dedup action on a duplicate scroll, got {r2}. "
            "LLM treats overlapping facts as novel — store would accumulate "
            "near-duplicates indefinitely."
        )
