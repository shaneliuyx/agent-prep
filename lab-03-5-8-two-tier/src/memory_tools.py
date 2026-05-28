# src/memory_tools.py — 5-tool client-facing surface
"""The §9.1 portability matrix's 5-tool contract (+ audit_replay), as
plain Python functions. Each tool wraps a TieredMemory primitive AND
emits the right AuditEntry shape, so §9's test matrix assertions
("verify the audit log records X") hold.

Production deployment (out of scope for this chapter):
  - Wrap each function as an MCP tool via @mcp.tool decorator (W6.7)
  - OR expose via HTTP / CLI / direct import — transport is orthogonal.

Audit emission policy: writes emit AuditEntry; reads do NOT (asymmetric
by design — auditing every read would explode the log without info gain,
same as Kafka append-only convention)."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.audit import AuditEntry, read_audit_log, record_audit
from src.dedup_synthesis import _qdrant_delete           # §8.1 primitive
from src.portability import export as _portability_export
from src.portability import import_ as _portability_import
from src.replay import replay_audit
from src.tiered_memory import TieredMemory


def _summary(content: str, n: int = 120) -> str:
    """First n chars; mirrors §3.4 AuditEntry.payload_summary convention."""
    return content[:n]


# ─── Task 1: Store ────────────────────────────────────────────────────────
def memory_store(tm: TieredMemory, *, content: str, type: str = "fact",
                 tags: list[str] | None = None, user_id: str,
                 agent_id: str = "") -> str:
    """Write a single memory + emit `imprint` AuditEntry. Returns the
    new point's UUID."""
    metadata: dict[str, Any] = {"type": type, "tags": tags or []}
    new_id = tm.imprint(content=content, metadata=metadata)
    record_audit(AuditEntry(
        operation="imprint",
        actor_agent_id=agent_id,
        user_id=user_id,
        new_id=new_id,
        payload_summary=_summary(content),
        metadata=metadata,
    ))
    return new_id


# ─── Task 2: Query ────────────────────────────────────────────────────────
def memory_query(tm: TieredMemory, *, query: str, k: int = 5,
                 user_id: str = "") -> list[dict]:
    """Retrieve up to k memories ranked by relevance. No audit emission
    (reads don't mutate state). Note: TieredMemory.query_context uses
    `k` kwarg, not `top_k` — match the §8.7 Protocol signature."""
    return tm.query_context(query=query, k=k)


# ─── Task 3: Supersede ────────────────────────────────────────────────────
def memory_supersede(tm: TieredMemory, *, old_id: str, new_content: str,
                     reason: str, user_id: str, agent_id: str = "") -> str:
    """Mark `old_id` superseded by a fresh imprint. §8.6 Step 1+2 shape
    (hard-delete + supersedes-pointer + supersede AuditEntry). Step 3
    soft-delete swap is contract-free — same call site, different
    `_qdrant_delete` impl."""
    _qdrant_delete(tm, [old_id])
    metadata = {
        "supersedes": old_id,
        "supersede_reason": reason,
        "fact_kind": "state_evolution",
    }
    new_id = tm.imprint(content=new_content, metadata=metadata)
    record_audit(AuditEntry(
        operation="supersede",
        actor_agent_id=agent_id,
        user_id=user_id,
        target_id=old_id,
        new_id=new_id,
        payload_summary=_summary(new_content),
        metadata=metadata,
    ))
    return new_id


# ─── Task 4: Export ───────────────────────────────────────────────────────
def memory_export(tm: TieredMemory, *, user_id: str,
                  out_path: Path) -> dict[str, Any]:
    """Snapshot user's memories + audit log to disk. Returns the
    ExportData fields as a dict so JSON-only transports (MCP/HTTP) can
    serialise without dataclass support."""
    bundle = _portability_export(tm, user_id=user_id, out_path=out_path)
    return asdict(bundle)


# ─── Task 5: Import ───────────────────────────────────────────────────────
def memory_import(tm: TieredMemory, *, in_path: Path) -> dict[str, Any]:
    """Rehydrate tm from a bundle on disk. Returns version + migration
    path + replay counts (see §9.2's `import_` for the contract)."""
    return _portability_import(tm, in_path)


# ─── Bonus tool: audit_replay (mentioned in §9.1 diagram) ────────────────
def audit_replay(tm: TieredMemory, *,
                 user_id: str | None = None) -> dict[str, int]:
    """Re-emit all audit-log writes against tm. Read filter optional
    (user_id) — without it, replays the whole log. Returns the §9.3
    counts dict {replayed, skipped, unknown}."""
    log = read_audit_log(user_id=user_id) if user_id else read_audit_log()
    return replay_audit(tm, log)