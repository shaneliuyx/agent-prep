"""Verifies the quality-gate audit extension in src/consolidation.py.

Three layers of coverage:
1. _audit_gate() helper — pure unit, no I/O dependencies.
2. Demote path — atom scoring below threshold writes a `demote` audit
   and skips imprint.
3. Promote path — atom scoring above threshold writes a `promote` audit
   AFTER the downstream imprint returns, with new_id chained.

Run: uv run pytest tests/test_consolidation_audit.py -v
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.audit import read_audit_log
from src.consolidation import _audit_gate


@pytest.fixture
def audit_log_path(tmp_path: Path):
    """Isolate audit log per test (avoids cross-test leakage)."""
    import src.audit as audit_mod
    original = audit_mod.DEFAULT_AUDIT_PATH
    test_path = tmp_path / "audit.jsonl"
    audit_mod.DEFAULT_AUDIT_PATH = test_path
    yield test_path
    audit_mod.DEFAULT_AUDIT_PATH = original


def test_audit_gate_emits_demote_with_correct_shape(audit_log_path: Path):
    """demote audit: target_id null, new_id null, metadata carries
    quest_id + quality_score + threshold + delta + fact_type + phase."""
    _audit_gate(
        decision="demote",
        actor_agent_id="agent_x",
        user_id="user_42",
        score=0.42,
        threshold=0.6,
        fact_preview="Some low-confidence claim about Terraform.",
        quest_id="QUEST-17",
        fact_type="observation",
    )
    assert audit_log_path.exists()
    entries = read_audit_log()
    assert len(entries) == 1
    e = entries[0]
    assert e["operation"] == "demote"
    assert e["target_id"] is None
    assert e["new_id"] is None
    assert e["actor_agent_id"] == "agent_x"
    assert e["user_id"] == "user_42"
    md = e["metadata"]
    assert md["quest_id"] == "QUEST-17"
    assert md["quality_score"] == 0.42
    assert md["threshold"] == 0.6
    assert md["delta"] == pytest.approx(-0.18, abs=1e-9)
    assert md["fact_type"] == "observation"
    assert md["phase"] == "pre_write_gate"


def test_audit_gate_emits_promote_with_new_id_chain(audit_log_path: Path):
    """promote audit: target_id null, new_id carries the downstream
    imprint's UUID so replay can chain gate→write."""
    _audit_gate(
        decision="promote",
        actor_agent_id="agent_x",
        user_id="user_42",
        score=0.87,
        threshold=0.6,
        fact_preview="High-confidence fact.",
        quest_id="QUEST-18",
        fact_type="fact",
        new_id="point-uuid-abc",
    )
    assert audit_log_path.exists()
    entries = read_audit_log()
    assert len(entries) == 1
    e = entries[0]
    assert e["operation"] == "promote"
    assert e["target_id"] is None
    assert e["new_id"] == "point-uuid-abc"
    assert e["metadata"]["delta"] == pytest.approx(0.27, abs=1e-9)


def test_audit_gate_promote_accepts_null_new_id_for_dedup_path(audit_log_path: Path):
    """When use_dedup=True, execute_action returns counts not UUIDs, so
    new_id stays None on the promote audit. Verify this is permitted."""
    _audit_gate(
        decision="promote",
        actor_agent_id="agent_x",
        user_id="",
        score=0.75,
        threshold=0.5,
        fact_preview="Fact handled by dedup pipeline.",
        quest_id="QUEST-99",
        new_id=None,
    )
    entries = read_audit_log()
    assert len(entries) == 1
    assert entries[0]["operation"] == "promote"
    assert entries[0]["new_id"] is None


@pytest.mark.asyncio
async def test_consolidate_emits_demote_for_low_confidence_atoms(audit_log_path: Path):
    """End-to-end: a scroll with an atom below threshold should produce
    a demote audit AND skip the imprint. Mocks the LLM atomiser to
    keep the test fast + deterministic; the consolidate() flow itself
    is exercised real (sqlite dedup table + control flow)."""
    import uuid
    from unittest.mock import AsyncMock, MagicMock

    from src.consolidation import consolidate

    # Fake atoms: one above threshold (will promote), one below (will demote).
    fake_atoms = [
        {"fact": "Terraform pattern works in production.", "type": "fact", "confidence": 0.85},
        {"fact": "Possibly the build is faster on Tuesdays.", "type": "observation", "confidence": 0.30},
    ]

    # Minimal TieredMemory stub matching the Protocol.
    tm = MagicMock()
    tm.agent_id = f"audit_consolidate_{uuid.uuid4().hex[:6]}"
    tm.user_id = "test_user"
    tm._http = MagicMock()
    tm.list_closed_quests = AsyncMock(return_value="QUEST-1: done\nQUEST-2: done")
    tm.get_scroll = AsyncMock(return_value="· [completed] some scroll text here")
    tm.imprint = MagicMock(return_value=f"new-point-{uuid.uuid4().hex[:8]}")

    # Isolate dedup sqlite per test.
    with patch("src.consolidation.DEDUP_DB", audit_log_path.parent / "dedup.sqlite"):
        with patch("src.consolidation.extract_atomic_facts", return_value=fake_atoms):
            result = await consolidate(
                tm,
                max_batch=10,
                promotion_threshold=0.6,
                use_atomisation=True,
                use_dedup=False,
            )

    # Each of 2 quests processed → 4 atoms total → 2 promote + 2 demote.
    promotes = read_audit_log(operation="promote")
    demotes = read_audit_log(operation="demote")
    assert len(promotes) == 2, f"expected 2 promotes, got {len(promotes)}"
    assert len(demotes) == 2, f"expected 2 demotes, got {len(demotes)}"

    # Every demote should target_id=null (pre-write gate).
    assert all(d["target_id"] is None for d in demotes)
    assert all(d["new_id"] is None for d in demotes)

    # Every promote should carry a new_id (non-dedup path captures the UUID).
    assert all(p["new_id"] is not None for p in promotes), \
        "use_dedup=False path should always chain new_id on promote"

    # Promote scores above threshold, demote scores below — delta sign check.
    assert all(p["metadata"]["delta"] >= 0 for p in promotes)
    assert all(d["metadata"]["delta"] < 0 for d in demotes)

    # tm.imprint called twice (once per promoted atom), zero for demoted.
    assert tm.imprint.call_count == 2
    assert result.facts_imprinted == 2
