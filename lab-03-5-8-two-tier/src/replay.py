# src/replay.py — reconstruct store state from audit log
"""Replay engine for AuditEntry logs. Reconstructs the store's state by
re-emitting every write captured in the audit log against an empty
target TieredMemory.

Idempotency contract: idempotent ONLY against an empty target. Caller
is responsible for resetting state (drop + recreate Qdrant collection,
or fresh EverCore Postgres schema) before invocation.

Op semantics (mirrors §3.4's 9-op Literal + §8.1/§8.6 dispatch):
  imprint              — fresh write; replay payload_summary as new fact
  promote              — quality-gate decision metadata; the actual write
                         is captured in the paired imprint (use_dedup=False)
                         or in update/delete/supersede/coexist (use_dedup=True).
                         SKIP at replay (double-write otherwise).
  demote               — quality-gate fail; NO write occurred; SKIP.
  update / delete      — replay payload_summary (the post-action content).
                         The original target_id was reconciled by ordering;
                         on an EMPTY target there's nothing to delete first.
  supersede            — replay payload_summary + propagate `supersedes`
                         pointer into metadata so chain traversal works.
  coexist              — replay payload_summary + propagate `relates_to`
                         pointer into metadata.
  noop_duplicate       — nothing happened; SKIP.
  compact              — offline audit-log housekeeping; SKIP at replay.
  <unknown>            — log warning + SKIP (forward-compat; non-zero
                         unknown count on replay = code stale vs log).
"""
from __future__ import annotations

import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class TieredMemoryLike(Protocol):
    """Structural subtype of both EverCore + Qdrant TieredMemory variants
    (same shape as §8.1's Protocol). Replay only needs imprint()."""
    def imprint(self, content: str,
                metadata: dict[str, Any] | None = None) -> str: ...


_OPS_REPLAY_AS_IMPRINT = {"imprint", "update", "delete", "supersede", "coexist"}
_OPS_SKIP = {"promote", "demote", "noop_duplicate", "compact"}


def replay_audit(tm: TieredMemoryLike, audit_log: list[dict]) -> dict[str, int]:
    """Replay every AuditEntry against tm. Returns per-bucket counts so
    tests can assert replay completion + surface forward-compat drift.

    Returns: {"replayed": int, "skipped": int, "unknown": int}
    Non-zero `unknown` on a clean replay means the log contains an op
    this code doesn't recognise — bump _OPS_REPLAY_AS_IMPRINT/_OPS_SKIP."""
    counts = {"replayed": 0, "skipped": 0, "unknown": 0}
    for entry in audit_log:
        op = entry.get("operation")
        if op in _OPS_SKIP:
            counts["skipped"] += 1
            continue
        if op not in _OPS_REPLAY_AS_IMPRINT:
            logger.warning(
                "replay_audit: unknown operation %r in entry %s",
                op, entry.get("id"),
            )
            counts["unknown"] += 1
            continue
        # Build metadata: start from entry's metadata, overlay supersede/coexist
        # pointers so downstream chain-walks still resolve after replay.
        meta = dict(entry.get("metadata") or {})
        if op == "supersede" and entry.get("target_id"):
            meta.setdefault("supersedes", entry["target_id"])
        elif op == "coexist" and entry.get("target_id"):
            meta.setdefault("relates_to", entry["target_id"])
        tm.imprint(content=entry["payload_summary"], metadata=meta)
        counts["replayed"] += 1
    return counts