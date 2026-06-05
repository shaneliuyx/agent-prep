"""Online dedup-and-synthesis (Batchelor-Manning 2026 form #1, extended).

Implements the "pay at write time" pattern: when a new fact arrives, query
the existing store for top-k semantically nearest candidates, then issue
ONE LLM call to decide an action. Execute + emit AuditEntry per action.

Six actions (Phase 9.6 — bitemporal extension):
  - add       : novel fact, no overlap
  - update    : new fact refines/corrects one candidate (same world-state)
  - supersede : new fact contradicts one candidate, BOTH WERE TRUE AT
                THEIR OWN TIMES (state evolution). Old marked superseded_by.
  - coexist   : new fact appears to contradict one candidate but applies
                to a DIFFERENT scope. Both true under different conditions.
  - delete    : old fact was factually false (hallucination); scrub it
  - no-op     : true duplicate, skip

Audit emission (Phase 3.4 + agentmemory pattern):
Every state-mutating branch emits one AuditEntry to data/audit.jsonl.
Downstream consumers: replay / CT pipeline / cross-backend export.

Scoped to the Qdrant TieredMemory variant for clean composition (EverCore
has its own internal extraction pipeline that doesn't expose delete/update
hooks cleanly).

Step 3 (WIRED 2026-06-05): supersede uses payload-patch SOFT-DELETE. The old
fact is kept and patched with `superseded_by` (the new fact's id); the new fact
carries the `supersedes` back-pointer. `query_context` excludes superseded facts
by default (`is_empty: superseded_by`) so live recall is identical to the old
hard-delete behavior, while `include_superseded=True` walks the full history —
true bitemporal semantics without losing the old content.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, Protocol

from openai import OpenAI

from src.audit import AuditEntry, record_audit


class TieredMemoryLike(Protocol):
    """Both EverCore and Qdrant variants — same surface; Pyright sees them
    as distinct classes without this Protocol shim. The audit code reads
    `agent_id` + `user_id` via getattr() defensive — they're instance
    attrs not class-level annotations, so we DON'T declare them in the
    Protocol (would force Pyright to require them as class attrs and
    miss runtime-only instance attrs)."""
    _http: Any

    def imprint(self, content: str, metadata: dict[str, Any] | None = ...) -> str: ...
    def query_context(
        self,
        query: str,
        k: int = ...,
        min_confidence: float = ...,
        type_filter: list[str] | None = ...,
        include_superseded: bool = ...,
    ) -> list[dict[str, Any]]: ...


DEDUP_PROMPT = """You are deduplicating an agent's long-term memory store.

NEW FACT (just observed at {now}):
{new_fact}

CANDIDATE EXISTING FACTS (top-k by semantic similarity, with timestamps):
{candidates}

Decide ONE action. Emit JSON.

Actions:

- "add": novel fact, no overlap with any candidate.

- "update": new fact REFINES one candidate (more detail, fixes an error
            in the SAME world-state). Old fact was wrong or incomplete;
            new fact is the corrected/expanded version. Old and new
            CANNOT both be true at the same time.
            Linguistic cues: "actually", "correction", "I was wrong";
            short time gap (seconds/minutes); same scope.

- "supersede": new fact CONTRADICTS one candidate but BOTH WERE TRUE
            AT THEIR OWN TIMES. State changed (preference shifted,
            config rotated, scope evolved, user switched tools/jobs).
            Old is historical truth; new is current truth. BOTH kept;
            old marked superseded_by new.
            Linguistic cues: "now", "switched to", "changed", "as of",
            "currently", "no longer"; larger time gap (hours/days+).
            Example: old="user likes React" (2024-01) + new="user
            prefers Vue now" (2026-05) -> supersede.

- "coexist": new fact APPEARS TO CONTRADICT one candidate but actually
            applies to a DIFFERENT scope or context. Both true at the
            same time under different conditions.
            Example: old="auth tokens expire after 30 min" (web app)
            + new="API keys never expire" (machine-to-machine) -> coexist.

- "delete": old fact was FACTUALLY FALSE — hallucination, parse error,
            mis-extraction. New fact replaces it cleanly. No value in
            keeping the old for audit. Rare; prefer supersede when
            ambiguous.

- "no-op": new fact is a true DUPLICATE of one candidate. No imprint.
            MUST include `target_id` of the duplicated candidate so
            downstream audit / replay can trace the duplicate chain.

Output JSON (no markdown fence, no prose):
{{"action": "add" | "update" | "supersede" | "coexist" | "delete" | "no-op",
  "target_id": "<id of related existing fact; required for update / supersede / coexist / delete>",
  "merged_content": "<for update only — combined fact text>",
  "supersede_reason": "<for supersede only — one sentence why this is state change not factual error>",
  "supersede_category": "<for supersede only — one of: preference, status, config, scope, identity, other>",
  "relates_to": "<for coexist only — target_id of the related candidate>"}}

Return ONLY the JSON."""


Action = Literal["add", "update", "supersede", "coexist", "delete", "no-op"]
_VALID_ACTIONS: tuple[str, ...] = (
    "add", "update", "supersede", "coexist", "delete", "no-op",
)


@dataclass
class DedupAction:
    action: Action
    target_id: str | None = None
    merged_content: str | None = None
    supersede_reason: str | None = None
    supersede_category: str | None = None
    relates_to: str | None = None


def _format_candidates(candidates: list[dict]) -> str:
    """Surfaces a `timestamp` field per candidate so the classifier can
    distinguish factual correction (short gap) from state evolution
    (large gap). Keys probed: timestamp, created_at, imprinted_at."""
    if not candidates:
        return "(none)"
    lines = []
    for c in candidates[:5]:
        cid = c.get("id") or c.get("point_id") or "?"
        content = c.get("content") or c.get("summary") or ""
        ts = (
            c.get("timestamp")
            or c.get("created_at")
            or c.get("imprinted_at")
            or "?"
        )
        score = c.get("score", 0.0)
        lines.append(
            f'  - id={cid!r}  imprinted={ts}  score={score:.3f}  '
            f'content="{content[:200]}"'
        )
    return "\n".join(lines)


def decide_action(new_fact: str, candidates: list[dict]) -> DedupAction:
    """LLM-mediated decision: one of 6 actions.
    Graceful fallbacks:
    - empty candidates -> add (no LLM call)
    - malformed JSON   -> add (safe default; loss mode = duplication)
    - unknown action   -> add (same)"""
    if not candidates:
        return DedupAction(action="add")

    client = OpenAI(
        base_url=os.getenv("OMLX_BASE_URL"),
        api_key=os.getenv("OMLX_API_KEY"),
    )
    now_iso = datetime.now(timezone.utc).isoformat()
    prompt = DEDUP_PROMPT.format(
        new_fact=new_fact,
        candidates=_format_candidates(candidates),
        now=now_iso,
    )
    resp = client.chat.completions.create(
        model=os.getenv("MODEL_HAIKU", "gpt-oss-20b-MXFP4-Q8"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=800,
    )
    raw = (resp.choices[0].message.content or "").strip()

    # Strip optional markdown fence
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return DedupAction(action="add")

    action = parsed.get("action")
    if action not in _VALID_ACTIONS:
        return DedupAction(action="add")

    # Defensive: classifier may emit no-op without target_id (prompt
    # requires it but LLM compliance drifts). Default to the highest-
    # similarity candidate so audit log isn't lossy on duplicate chains.
    target_id = parsed.get("target_id")
    if action == "no-op" and not target_id and candidates:
        target_id = candidates[0].get("id") or candidates[0].get("point_id")

    return DedupAction(
        action=action,
        target_id=target_id,
        merged_content=parsed.get("merged_content"),
        supersede_reason=parsed.get("supersede_reason"),
        supersede_category=parsed.get("supersede_category"),
        relates_to=parsed.get("relates_to"),
    )


def execute_action(tm: TieredMemoryLike, action: DedupAction, new_fact: str,
                   metadata: dict | None = None) -> dict:
    """Apply a DedupAction against a Qdrant TieredMemory; emit AuditEntry
    per operation. Returns counters for caller aggregation.

    Counter dict shape:
      {imprinted, updated, deleted, noop, superseded, coexisted}

    Each call increments exactly one PRIMARY counter (matches action).
    `imprinted` is SECONDARY — increments on any write (add / update /
    supersede / coexist / delete-then-add). Callers aggregating
    "total writes" should sum `imprinted` alone.
    """
    counts = {
        "imprinted": 0, "updated": 0, "deleted": 0, "noop": 0,
        "superseded": 0, "coexisted": 0,
    }
    md = metadata or {}
    payload_summary = new_fact[:120]
    actor = getattr(tm, "agent_id", "")
    user = getattr(tm, "user_id", "")

    def _audit(operation, *, target_id=None, new_id=None, summary=payload_summary, **meta):
        record_audit(AuditEntry(
            operation=operation,
            actor_agent_id=actor, user_id=user,
            target_id=target_id, new_id=new_id,
            payload_summary=summary,
            metadata=meta,
        ))

    if action.action == "no-op":
        counts["noop"] += 1
        _audit("noop_duplicate", target_id=action.target_id,
               reason="true_duplicate_per_classifier")
        return counts

    if action.action == "delete" and action.target_id:
        _qdrant_delete(tm, [action.target_id])
        # Per Phase 9 spec: delete is followed by add (new fact replaces old)
        new_id = tm.imprint(content=new_fact, metadata=md)
        counts["deleted"] += 1
        counts["imprinted"] += 1
        _audit("delete", target_id=action.target_id, new_id=new_id,
               reason="factually_false_per_classifier")
        return counts

    if action.action == "update" and action.target_id:
        _qdrant_delete(tm, [action.target_id])
        merged = action.merged_content or new_fact
        new_id = tm.imprint(content=merged, metadata=md)
        counts["updated"] += 1
        counts["imprinted"] += 1
        _audit("update", target_id=action.target_id, new_id=new_id,
               summary=merged[:120],
               reason="factual_correction_same_world_state", merged=True)
        return counts

    if action.action == "supersede" and action.target_id:
        # Step 3 WIRED: payload-patch SOFT-DELETE (was hard-delete). Imprint
        # the new fact FIRST (we need its id for the old fact's back-pointer),
        # then patch the OLD point with `superseded_by` instead of deleting it.
        # query_context filters superseded facts out of live recall by default
        # (is_empty: superseded_by), so accuracy is identical to the hard-delete
        # era; include_superseded=True walks the chain both ways for audit.
        superseded_at = datetime.now(timezone.utc).isoformat()
        new_id = tm.imprint(content=new_fact, metadata={
            **md,
            "supersedes": action.target_id,
            "supersede_reason": action.supersede_reason,
            "supersede_category": action.supersede_category,
            "fact_kind": "state_evolution",
        })
        _qdrant_supersede(tm, action.target_id, {
            "superseded_by": new_id,
            "superseded_at": superseded_at,
            "supersede_reason": action.supersede_reason,
            "supersede_category": action.supersede_category,
        })
        counts["superseded"] += 1
        counts["imprinted"] += 1
        _audit("supersede", target_id=action.target_id, new_id=new_id,
               supersede_category=action.supersede_category,
               supersede_reason=action.supersede_reason,
               fact_kind="state_evolution")
        return counts

    if action.action == "coexist" and (action.relates_to or action.target_id):
        related = action.relates_to or action.target_id
        new_id = tm.imprint(content=new_fact, metadata={
            **md, "relates_to": related, "fact_kind": "scoped_variant",
        })
        counts["coexisted"] += 1
        counts["imprinted"] += 1
        _audit("coexist", target_id=related, new_id=new_id,
               relates_to=related, fact_kind="scoped_variant")
        return counts

    # Default: add (no candidates OR LLM picked add OR malformed-output fallback)
    new_id = tm.imprint(content=new_fact, metadata=md)
    counts["imprinted"] += 1
    _audit("imprint", new_id=new_id,
           reason="novel_per_classifier_or_empty_candidates")
    return counts


def _qdrant_delete(tm: TieredMemoryLike, point_ids: list[str]) -> None:
    """Delete points by ID via Qdrant's points/delete endpoint."""
    from src.tiered_memory_qdrant import COLLECTION

    r = tm._http.post(
        f"/collections/{COLLECTION}/points/delete",
        json={"points": point_ids},
    )
    r.raise_for_status()


def _qdrant_supersede(tm: TieredMemoryLike, point_id: str, patch: dict[str, Any]) -> None:
    """Soft-delete: PATCH a point's payload (set `superseded_by` etc.) instead
    of removing it. Qdrant's set-payload endpoint merges `patch` into the
    existing payload, leaving the vector + original content intact so the fact
    stays queryable for audit (query_context(include_superseded=True))."""
    from src.tiered_memory_qdrant import COLLECTION

    r = tm._http.post(
        f"/collections/{COLLECTION}/points/payload",
        json={"payload": patch, "points": [point_id]},
    )
    r.raise_for_status()
