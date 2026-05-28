# tests/test_phase9.py — Phase 9 coverage: replay + portability + memory_tools
"""Phase 9 test suite. Three modules covered:
  - §9.3 replay.py             (TestReplayPureUnits — _FakeTM shim, no Qdrant)
  - §9.2 portability.py        (TestPortabilityMigration + TestPortabilityVersionGate)
  - §9.1.1 memory_tools.py     (TestMemoryToolsIntegration — live Qdrant + audit)

Run:
  uv run pytest tests/test_phase9.py -v
"""
from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

import pytest

from src.audit import read_audit_log
from src.memory_tools import (
    memory_export,
    memory_import,
    memory_query,
    memory_store,
    memory_supersede,
)
from src.portability import (
    CURRENT_VERSION,
    SUPPORTED_VERSIONS,
    _v1_to_v2,
    _v2_to_v3,
)
from src.portability import import_ as portability_import
from src.replay import replay_audit
from src.tiered_memory_qdrant import TieredMemory


# ─── Fixtures ────────────────────────────────────────────────────────────
@pytest.fixture
def isolated_audit(tmp_path):
    """Per-test audit log; restore module DEFAULT_AUDIT_PATH on teardown."""
    import src.audit as audit_mod
    original = audit_mod.DEFAULT_AUDIT_PATH
    audit_mod.DEFAULT_AUDIT_PATH = tmp_path / "audit.jsonl"
    yield audit_mod.DEFAULT_AUDIT_PATH
    audit_mod.DEFAULT_AUDIT_PATH = original


@pytest.fixture
def isolated_user_id():
    """Per-test user_id for namespace isolation (§7.5 pattern). Tests
    own the `async with TieredMemory(...)` lifecycle in their own task
    to avoid the pytest-asyncio + anyio TaskGroup task-affinity issue:
    pytest-asyncio's generator fixtures run setup and teardown in
    different tasks, but anyio TaskGroups (used internally by the MCP
    stdio_client backing TieredMemory) reject cross-task __aexit__.
    Keeping the lifecycle inside each test's task body resolves it."""
    return f"phase9_test_{uuid.uuid4().hex[:8]}"


class _FakeTM:
    """In-memory shim implementing TieredMemoryLike.imprint only. Replay
    unit tests use this to test audit-log-shape contract without needing
    a live Qdrant. Records each imprint call for assertion."""

    def __init__(self):
        self.calls: list[dict] = []

    def imprint(self, content: str, metadata: dict | None = None) -> str:
        pid = uuid.uuid4().hex
        self.calls.append({"content": content,
                           "metadata": metadata or {},
                           "id": pid})
        return pid


# ─── §9.3 replay.py — pure unit tests (no Qdrant) ────────────────────────
class TestReplayPureUnits:
    """Replay semantics on synthetic audit dicts via _FakeTM."""

    def test_imprint_replays_as_imprint(self):
        tm = _FakeTM()
        log = [{"operation": "imprint", "payload_summary": "fact A",
                "metadata": {}}]
        counts = replay_audit(tm, log)
        assert counts == {"replayed": 1, "skipped": 0, "unknown": 0}
        assert tm.calls[0]["content"] == "fact A"

    def test_promote_demote_skipped(self):
        """promote+demote are decision records, not writes — must not replay."""
        tm = _FakeTM()
        log = [
            {"operation": "promote", "payload_summary": "x", "metadata": {}},
            {"operation": "demote",  "payload_summary": "y", "metadata": {}},
        ]
        counts = replay_audit(tm, log)
        assert counts == {"replayed": 0, "skipped": 2, "unknown": 0}
        assert tm.calls == []   # nothing imprinted

    def test_supersede_preserves_supersedes_pointer(self):
        tm = _FakeTM()
        log = [{
            "operation": "supersede", "payload_summary": "Svelte",
            "target_id": "react-uuid", "metadata": {},
        }]
        replay_audit(tm, log)
        assert tm.calls[0]["metadata"]["supersedes"] == "react-uuid"

    def test_coexist_preserves_relates_to(self):
        tm = _FakeTM()
        log = [{
            "operation": "coexist", "payload_summary": "API keys never expire",
            "target_id": "auth-uuid", "metadata": {},
        }]
        replay_audit(tm, log)
        assert tm.calls[0]["metadata"]["relates_to"] == "auth-uuid"

    def test_unknown_op_logs_warning_and_counts(self, caplog):
        """Forward-compat canary: unknown op = WARNING + counter increment."""
        tm = _FakeTM()
        log = [{"operation": "future_op_v4", "id": "abc",
                "payload_summary": "x"}]
        with caplog.at_level(logging.WARNING, logger="src.replay"):
            counts = replay_audit(tm, log)
        assert counts == {"replayed": 0, "skipped": 0, "unknown": 1}
        assert "future_op_v4" in caplog.text
        assert tm.calls == []   # unknown op = no imprint

    def test_caller_metadata_supersedes_not_clobbered(self):
        """setdefault contract: caller-provided supersedes pointer wins."""
        tm = _FakeTM()
        log = [{
            "operation": "supersede", "payload_summary": "y",
            "target_id": "from_target", "metadata": {"supersedes": "caller_set"},
        }]
        replay_audit(tm, log)
        assert tm.calls[0]["metadata"]["supersedes"] == "caller_set"


# ─── §9.2 portability.py — migration unit tests (no Qdrant) ─────────────
class TestPortabilityMigration:
    """Pure dict transforms for the v1 → v3 chain."""

    def test_v1_to_v2_splits_content_to_payload_summary(self):
        v1 = {"version": "v1", "memories": [{"content": "x" * 200, "id": "p1"}]}
        v2 = _v1_to_v2(v1)
        assert v2["version"] == "v2"
        assert len(v2["memories"][0]["payload_summary"]) == 120
        assert v2["memories"][0]["metadata"]["legacy_v1_content"] == "x" * 200

    def test_v2_to_v3_coerces_upsert_to_imprint(self):
        v2 = {"version": "v2", "memories": [], "audit_log": [
            {"operation": "upsert", "id": "a"},      # legacy op name
            {"operation": "supersede", "id": "b"},   # current name
        ]}
        v3 = _v2_to_v3(v2)
        assert v3["version"] == "v3"
        assert v3["audit_log"][0]["operation"] == "imprint"
        assert v3["audit_log"][1]["operation"] == "supersede"

    def test_migrate_to_current_walks_v1_to_v3(self):
        """Full chain walk — v1 input lands at v3 with path ['v1','v2','v3']."""
        from src.portability import migrate_to_current
        v1 = {"version": "v1",
              "memories": [{"content": "f", "id": "p"}],
              "audit_log": []}
        out, path = migrate_to_current(v1)
        assert out["version"] == CURRENT_VERSION
        assert path == ["v1", "v2", "v3"]

    def test_migrate_skipped_on_current_version(self):
        """Walker exits immediately when already at CURRENT_VERSION."""
        from src.portability import migrate_to_current
        cur = {"version": CURRENT_VERSION, "memories": [], "audit_log": []}
        out, path = migrate_to_current(cur)
        assert path == [CURRENT_VERSION]


# ─── §9.2 portability.py — version-gate test (no Qdrant) ────────────────
class TestPortabilityVersionGate:
    """Unsupported version = clear error, not silent drift."""

    def test_import_rejects_unsupported_version(self, tmp_path, isolated_audit):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"version": "v99",
                                    "memories": [], "audit_log": []}))
        tm = _FakeTM()
        with pytest.raises(ValueError, match="not in supported set"):
            portability_import(tm, bad)


# ─── §9.1.1 memory_tools + §9.2 round-trip — integration ────────────────
class TestMemoryToolsIntegration:
    """Live Qdrant + isolated audit log. Each test exercises ONE tool +
    asserts the matching AuditEntry shape per §9's task matrix contract."""

    @pytest.mark.asyncio
    async def test_memory_store_emits_imprint_audit(self, isolated_user_id,
                                                     isolated_audit):
        async with TieredMemory(agent_id="phase9_agent",
                                user_id=isolated_user_id) as tm:
            new_id = memory_store(tm, content="fact about kubernetes",
                                  type="fact", tags=["infra", "k8s"],
                                  user_id=isolated_user_id)
        entries = read_audit_log()
        assert len(entries) == 1
        assert entries[0]["operation"] == "imprint"
        assert entries[0]["new_id"] == new_id
        assert entries[0]["metadata"]["tags"] == ["infra", "k8s"]

    @pytest.mark.asyncio
    async def test_memory_query_emits_no_audit(self, isolated_user_id,
                                                isolated_audit):
        """Reads don't audit (asymmetric policy: writes audit, reads don't)."""
        async with TieredMemory(agent_id="phase9_agent",
                                user_id=isolated_user_id) as tm:
            memory_store(tm, content="kubernetes scheduling primer",
                         user_id=isolated_user_id)
            results = memory_query(tm, query="kubernetes", k=3,
                                   user_id=isolated_user_id)
        entries = read_audit_log()
        assert len(entries) == 1            # only the store, not the query
        assert entries[0]["operation"] == "imprint"
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_memory_supersede_emits_supersede_audit(self,
                                                          isolated_user_id,
                                                          isolated_audit):
        async with TieredMemory(agent_id="phase9_agent",
                                user_id=isolated_user_id) as tm:
            old_id = memory_store(tm, content="user prefers React",
                                  user_id=isolated_user_id)
            new_id = memory_supersede(tm, old_id=old_id,
                                      new_content="user now prefers Svelte",
                                      reason="preference shifted",
                                      user_id=isolated_user_id)
        sup = [e for e in read_audit_log() if e["operation"] == "supersede"]
        assert len(sup) == 1
        assert sup[0]["target_id"] == old_id
        assert sup[0]["new_id"] == new_id
        assert sup[0]["metadata"]["supersede_reason"] == "preference shifted"

    @pytest.mark.asyncio
    async def test_export_import_round_trip(self, isolated_user_id, tmp_path,
                                             isolated_audit):
        """Single-source-of-truth import: replay reconstructs from audit log
        alone. Regression-catches the original double-write trap (if anyone
        re-adds memories[] iteration in import_, replay_counts doubles)."""
        bundle_path = tmp_path / "export.json"
        async with TieredMemory(agent_id="phase9_agent",
                                user_id=isolated_user_id) as tm:
            memory_store(tm, content="fact 1: terraform supports OpenTofu",
                         user_id=isolated_user_id)
            memory_store(tm, content="fact 2: pgvector supports HNSW",
                         user_id=isolated_user_id)
            bundle = memory_export(tm, user_id=isolated_user_id,
                                   out_path=bundle_path)
        assert bundle["version"] == CURRENT_VERSION
        assert len(bundle["audit_log"]) == 2

        # Replay into fresh TM namespace; audit log is single source of truth.
        user2 = f"phase9_replay_{uuid.uuid4().hex[:8]}"
        async with TieredMemory(agent_id="phase9_replay_agent",
                                user_id=user2) as tm2:
            result = memory_import(tm2, in_path=bundle_path)
        assert result["replay_counts"]["replayed"] == 2   # exactly 2, NOT 4
        assert result["replay_counts"]["unknown"] == 0
        assert result["migrated_through"] == [CURRENT_VERSION]