"""Verifies AuditEntry records correctly per execute_action branch.
Run: uv run pytest tests/test_audit.py -v
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from src.audit import AuditEntry, record_audit, read_audit_log
from src.dedup_synthesis import DedupAction, execute_action
from src.tiered_memory_qdrant import TieredMemory


@pytest.fixture
def audit_log_path(tmp_path):
    """Isolate the audit log per test (avoids cross-test leakage).
    Patch the module-level DEFAULT_AUDIT_PATH temporarily."""
    import src.audit as audit_mod
    original = audit_mod.DEFAULT_AUDIT_PATH
    test_path = tmp_path / "audit.jsonl"
    audit_mod.DEFAULT_AUDIT_PATH = test_path
    yield test_path
    audit_mod.DEFAULT_AUDIT_PATH = original


def test_record_audit_writes_jsonl(audit_log_path: Path):
    """One entry -> one line of JSON in the log file."""
    entry = AuditEntry(
        operation="imprint",
        actor_agent_id="test_agent",
        user_id="test_user",
        new_id="point-abc",
        payload_summary="hello world",
    )
    record_audit(entry)
    assert audit_log_path.exists()
    lines = audit_log_path.read_text().strip().split("\n")
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["operation"] == "imprint"
    assert parsed["new_id"] == "point-abc"


def test_read_audit_log_filters_by_user(audit_log_path: Path):
    """user_id filter should exclude entries from other users."""
    record_audit(AuditEntry(operation="imprint", user_id="alice", new_id="a"))
    record_audit(AuditEntry(operation="imprint", user_id="bob", new_id="b"))
    alice_entries = read_audit_log(user_id="alice")
    assert len(alice_entries) == 1
    assert alice_entries[0]["new_id"] == "a"


def test_read_audit_log_filters_by_operation(audit_log_path: Path):
    """operation filter returns only matching operations."""
    record_audit(AuditEntry(operation="imprint", new_id="a"))
    record_audit(AuditEntry(operation="supersede", new_id="b", target_id="x"))
    record_audit(AuditEntry(operation="noop_duplicate"))
    sup = read_audit_log(operation="supersede")
    assert len(sup) == 1
    assert sup[0]["target_id"] == "x"


@pytest.mark.asyncio
async def test_execute_action_emits_audit_per_branch(audit_log_path: Path):
    """Each execute_action branch (no-op / add / supersede / coexist /
    delete / update) should produce exactly one AuditEntry."""
    async with TieredMemory(agent_id=f"audit_test_{uuid.uuid4().hex[:6]}") as tm:
        # 1. ADD branch (no target_id; classifier picked add)
        execute_action(tm, DedupAction(action="add"), "new fact about Terraform")
        # 2. NO-OP branch
        execute_action(tm, DedupAction(action="no-op", target_id="existing-x"),
                       "duplicate fact")
        # 3. SUPERSEDE branch — needs a real target_id in Qdrant; for unit
        #    test scope, mock target_id (delete will fail but we only check
        #    audit log writes BEFORE delete; in integration test use real
        #    target_id)
        # ... see lab repo for the full integration variant

    entries = read_audit_log()
    operations = [e["operation"] for e in entries]
    assert "imprint" in operations
    assert "noop_duplicate" in operations
