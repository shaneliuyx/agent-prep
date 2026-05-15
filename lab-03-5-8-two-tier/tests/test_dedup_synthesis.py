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
    """Contradicting fact → LLM picks any non-silencing action. Auth-token-
    expiry change reads like config rotation (state evolution) so the
    6-action classifier (Phase 9.6) MAY pick `supersede` over `delete` /
    `update`. All four are acceptable — each resolves the contradiction
    without losing the new fact; we don't assert a specific choice.

    Hard invariant: classifier MUST NOT pick `no-op` or `add` — both
    silence the contradiction. no-op retains the now-false 30-minute
    fact alongside the new 1-hour fact; add accumulates both as
    independent memories (W3.5.8 §9.6 contract).

    See W3.5.8 BCJ Entry 15 for the 4→6 action assertion-set widening.
    """
    candidates = [
        {
            "id": "existing-1",
            "content": "Auth tokens expire after 30 minutes.",
            "score": 0.85,
        }
    ]
    new_fact = "Auth tokens expire after 1 hour."
    action = decide_action(new_fact, candidates)
    assert action.action in ("delete", "update", "supersede"), (
        f"unexpected action: {action.action} — see W3.5.8 §9.6 contract"
    )
    # If supersede, the classifier should bind the target and supply a reason
    # (config category is the canonical 'state evolution' bucket here).
    if action.action == "supersede":
        assert action.target_id == "existing-1"
        assert action.supersede_reason


def test_decide_action_emits_supersede_on_temporal_state_change():
    """Phase 9.5 — state evolution classification.

    Old fact + new fact that contradicts BUT reads as state change
    (linguistic cue "switched", large time gap in timestamps) should
    classify as supersede (preferred) — or update / delete as
    acceptable fallbacks. no-op or add would be wrong: both silence
    the temporal state-change signal the chapter Phase 9.5 is built
    to demonstrate.

    Validates BOTH Step 1 (5-action prompt) and Step 2 (timestamp
    injection in _format_candidates) together — without ts in the
    candidate, the LLM has no temporal cue and is far more likely
    to pick `update` (correction) over `supersede` (state change).
    """
    candidates = [
        {
            "id": "existing-react",
            "content": "User prefers React for frontend work.",
            "timestamp": "2024-01-15T10:00:00+00:00",
            "score": 0.82,
        }
    ]
    new_fact = "User has now switched to Vue for all new frontend projects."
    action = decide_action(new_fact, candidates)
    assert action.action in ("supersede", "update", "delete"), (
        f"expected state-change classification, got {action.action}. "
        "no-op/add silence the temporal signal — Phase 9.5 contract violated."
    )
    if action.action == "supersede":
        assert action.target_id == "existing-react", (
            "supersede must reference the contradicted candidate's id "
            "so downstream payload-patch (Step 3) can mark the chain"
        )
        assert action.supersede_reason, (
            "supersede must explain WHY it's state-change not correction — "
            "field is the audit hook for bitemporal queries"
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
