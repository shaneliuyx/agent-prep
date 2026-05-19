"""Append-only audit-log primitive (Phase 3.4 + agentmemory pattern).

Every memory operation records an AuditEntry; downstream replay /
CT pipeline / cross-backend export consume this log.

Wire-in: src/dedup_synthesis.py:execute_action() calls record_audit()
in each branch. consolidate() in src/consolidation.py needs no change —
it already receives the counts dict from execute_action.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal


AuditOperation = Literal[
    "imprint",            # initial write (default add)
    "update",             # Phase 9 form #1 update (factual correction; same world-state)
    "supersede",          # Phase 9.6 supersede (state evolution; both retained)
    "coexist",            # Phase 9.6 coexist (scoped variant; both retained)
    "delete",             # Phase 9 delete (factually false; remove)
    "noop_duplicate",     # Phase 9 no-op (true duplicate; skip)
    "promote",            # Phase 3.3 quality gate promotion (above threshold)
    "demote",             # Phase 3.3 quality gate demotion (below threshold)
    "compact",            # offline housekeeping (batch dedup / cleanup)
]


DEFAULT_AUDIT_PATH = Path(__file__).resolve().parent.parent / "data" / "audit.jsonl"


@dataclass(frozen=True)
class AuditEntry:
    """One operation on the memory store; append-only.
    `metadata` carries operation-specific fields (supersede_reason,
    supersede_category, fact_kind, threshold, etc)."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    operation: AuditOperation = "imprint"
    actor_agent_id: str = ""
    user_id: str = ""
    target_id: str | None = None        # the existing point this op modifies (None for fresh add)
    new_id: str | None = None           # the new point produced (for imprint / supersede / update / coexist)
    payload_summary: str = ""           # first ~120 chars of fact content
    metadata: dict[str, Any] = field(default_factory=dict)


def record_audit(audit: AuditEntry, log_path: Path | None = None) -> None:
    """Append one AuditEntry to a JSONL log. Idempotent on the file system
    (parent dir auto-created). Single writer assumption; if multi-process
    writers are needed, wrap in fcntl.flock or switch to SQLite."""
    path = log_path or DEFAULT_AUDIT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(asdict(audit)) + "\n")


def read_audit_log(log_path: Path | None = None,
                   user_id: str | None = None,
                   operation: AuditOperation | None = None) -> list[dict]:
    """Read audit log with optional user_id + operation filters.
    Returns one dict per AuditEntry (asdict() shape). For replay or
    cross-backend export."""
    path = log_path or DEFAULT_AUDIT_PATH
    if not path.exists():
        return []
    entries = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if user_id is not None and entry.get("user_id") != user_id:
                continue
            if operation is not None and entry.get("operation") != operation:
                continue
            entries.append(entry)
    return entries
