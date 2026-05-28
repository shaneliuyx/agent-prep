# src/portability.py — versioned export with forward + backward compat
"""Export + import for TieredMemory bundles. Round-trippable across
the §9.1 multi-client portability matrix (Claude Code / Cursor / Gemini
CLI / Codex CLI / Hermes / OpenClaw / pi / OpenCode).

Schema versioning (semver-style):
  - CURRENT_VERSION  = the version this code WRITES.
  - SUPPORTED_VERSIONS = the set this code can READ (always includes CURRENT).
  - migrate_to_current(raw) walks the v_n → v_(n+1) chain to land at CURRENT.

Rule: old bundles remain importable; new bundles may not be readable by
old code. Standard write-newest / read-historical semver discipline.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol

from src.audit import read_audit_log
from src.replay import replay_audit


ExportVersion = Literal["v1", "v2", "v3"]
SUPPORTED_VERSIONS: set[ExportVersion] = {"v1", "v2", "v3"}
CURRENT_VERSION: ExportVersion = "v3"


class TieredMemoryLike(Protocol):
    """Structural subtype of both EverCore + Qdrant TieredMemory variants.
    Portability needs imprint (write side); `dump_all_for_user` is OPTIONAL
    and probed via hasattr at export time. Audit log is the canonical
    source of truth for import; memories[] is best-effort for human
    inspection."""
    def imprint(self, content: str,
                metadata: dict[str, Any] | None = None) -> str: ...


@dataclass
class ExportData:
    version: ExportVersion
    schema_url: str              # e.g. https://.../schemas/v3.json
    exported_at: str             # ISO 8601 UTC
    user_id: str
    memories: list[dict]         # full Qdrant payload OR EverCore episode rows
    audit_log: list[dict]        # AuditEntry records (§3.4 + §8.7 wire-in)


def export(tm: TieredMemoryLike, user_id: str, out_path: Path) -> ExportData:
    """Snapshot a user's memories + audit log to disk. Returns the
    ExportData so in-process callers (round-trip tests) can assert
    without re-reading the file.

    Memories[] is best-effort: if the backend variant ships
    `dump_all_for_user`, use it; otherwise empty list. Import path
    reconstructs from audit_log alone (single source of truth)."""
    memories: list[dict] = []
    if hasattr(tm, "dump_all_for_user"):
        memories = tm.dump_all_for_user(user_id)
    bundle = ExportData(
        version=CURRENT_VERSION,
        schema_url=f"https://github.com/shaneliuyx/agent-prep/schemas/{CURRENT_VERSION}.json",
        exported_at=datetime.now(timezone.utc).isoformat(),
        user_id=user_id,
        memories=memories,
        audit_log=read_audit_log(user_id=user_id),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(bundle), indent=2))
    return bundle


def import_(tm: TieredMemoryLike, in_path: Path) -> dict[str, Any]:
    """Read a bundle; migrate to CURRENT_VERSION if older; rehydrate tm
    via replay_audit ONLY (audit log is the single source of truth — the
    memories[] field is for human-readable inspection, not for import,
    to avoid the double-write trap).

    Returns: {"version_read": str, "migrated_through": list[str],
              "replay_counts": dict[str, int]}"""
    raw = json.loads(in_path.read_text())
    v = raw.get("version")
    if v not in SUPPORTED_VERSIONS:
        raise ValueError(
            f"export version {v!r} not in supported set {SUPPORTED_VERSIONS}; "
            f"upgrade your reader OR re-export from the source"
        )
    bundle, migration_path = migrate_to_current(raw)
    replay_counts = replay_audit(tm, bundle["audit_log"])
    return {
        "version_read": v,
        "migrated_through": migration_path,
        "replay_counts": replay_counts,
    }


def migrate_to_current(raw: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Walk the version chain v_n → v_(n+1) → ... → CURRENT_VERSION.

    Returns: (migrated_bundle, list_of_versions_traversed).
    One-step-per-version: each transform is small, auditable, reversible-
    in-spirit. An external reviewer reads one function per schema bump."""
    path = [raw["version"]]
    bundle = dict(raw)
    while bundle["version"] != CURRENT_VERSION:
        next_step = _MIGRATIONS[bundle["version"]]
        bundle = next_step(bundle)
        path.append(bundle["version"])
    return bundle, path


def _v1_to_v2(b: dict[str, Any]) -> dict[str, Any]:
    """v1 → v2: split single `content` field into `payload_summary` +
    `metadata.legacy_v1_content`. v1 conflated them; v2 follows §3.4's
    AuditEntry shape (payload_summary capped at ~120 chars)."""
    out = dict(b)
    out["version"] = "v2"
    for m in out["memories"]:
        if "payload_summary" not in m and "content" in m:
            m["payload_summary"] = m["content"][:120]
            m.setdefault("metadata", {})["legacy_v1_content"] = m["content"]
    return out


def _v2_to_v3(b: dict[str, Any]) -> dict[str, Any]:
    """v2 → v3: add `schema_url` (was previously inferable from version
    string only); coerce legacy audit op names to §3.4's 9-op enum
    (`upsert` → `imprint`)."""
    out = dict(b)
    out["version"] = "v3"
    out.setdefault(
        "schema_url",
        "https://github.com/shaneliuyx/agent-prep/schemas/v3.json",
    )
    for entry in out.get("audit_log", []):
        if entry.get("operation") == "upsert":  # v2-era op name
            entry["operation"] = "imprint"
    return out


_MIGRATIONS = {
    "v1": _v1_to_v2,
    "v2": _v2_to_v3,
}