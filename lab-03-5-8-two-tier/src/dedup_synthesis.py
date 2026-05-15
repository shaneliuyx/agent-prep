"""Online dedup-and-synthesis (Batchelor-Manning 2026 form #1, extended).

Implements the "pay at write time" pattern: when a new fact arrives, query
the existing store for top-k semantically nearest candidates, then issue
ONE LLM call to decide an action. Execute.

Six actions (Phase 9.5 — bitemporal extension):
  - add       : novel fact, no overlap
  - update    : new fact refines/corrects one candidate (same world-state)
  - supersede : new fact contradicts one candidate, BOTH WERE TRUE AT
                THEIR OWN TIMES (state evolution — preference shift,
                config rotation, scope change). Old marked superseded_by.
  - coexist   : new fact appears to contradict one candidate but applies
                to a DIFFERENT scope. Both true under different conditions.
  - delete    : old fact was factually false (hallucination); scrub it
  - no-op     : true duplicate, skip

The supersede / coexist split (vs flat overwrite) is the contribution.
Most flat-write systems collapse all contradictions to "overwrite" and
lose audit trail. Splitting lets bitemporal queries answer "what did the
agent believe at t₀?" instead of just "what does it believe now?".

Article's claim from the 19-system corpus: this is the HIGHEST-ROI
write-time form — compounds across every subsequent read.

Scoped to the Qdrant TieredMemory variant for clean composition (EverCore
has its own internal extraction pipeline that doesn't expose delete/update
hooks cleanly).

Step 3 (deferred): supersede currently uses HARD-DELETE for the old fact.
Once `_qdrant_supersede` (payload-patch soft-delete) lands, the new fact's
`supersedes` pointer + query-time filter give true bitemporal semantics
without losing the old content.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, Protocol

from openai import OpenAI


class TieredMemoryLike(Protocol):
    """Both EverCore and Qdrant variants — same surface; Pyright sees them
    as distinct classes without this Protocol shim."""
    _http: Any

    def imprint(self, content: str, metadata: dict[str, Any] | None = ...) -> str: ...
    def query_context(
        self,
        query: str,
        k: int = ...,
        min_confidence: float = ...,
        type_filter: list[str] | None = ...,
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
    """Render top-k candidates as numbered list for the LLM prompt.

    Surfaces a `timestamp` field per candidate so the classifier can
    distinguish:
      - short gap (sec/min)   -> likely factual correction (update)
      - large gap (hours+)    -> likely state evolution (supersede)

    Keys probed in order:
      - "timestamp"      (Qdrant payload, Step 2 default)
      - "created_at"     (EverCore episode response)
      - "imprinted_at"   (legacy callers)
    Falls back to "?" if none present — classifier degrades gracefully
    (loses temporal signal, still has semantic + linguistic cues).
    """
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
    """LLM-mediated decision: add / update / delete / no-op.

    Graceful failure modes:
    - LLM returns malformed JSON → default to "add" (safe fallback;
      no data loss, may accumulate near-duplicates which is acceptable
      vs the alternative of silently dropping the fact)
    - LLM returns unknown action → "add" with warning logged via target_id
    - Empty candidates → return add immediately (no LLM call needed)
    """
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
        raw = raw.strip("`")
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

    return DedupAction(
        action=action,
        target_id=parsed.get("target_id"),
        merged_content=parsed.get("merged_content"),
        supersede_reason=parsed.get("supersede_reason"),
        supersede_category=parsed.get("supersede_category"),
        relates_to=parsed.get("relates_to"),
    )


def execute_action(tm: TieredMemoryLike, action: DedupAction, new_fact: str,
                   metadata: dict | None = None) -> dict:
    """Apply a DedupAction against a Qdrant TieredMemory.

    Returns a dict with counters for the caller to aggregate:
      {"imprinted", "updated", "deleted", "noop", "superseded", "coexisted"}

    Each call increments exactly one *primary* counter (the action's
    classification). `imprinted` is a secondary counter that ALSO
    increments whenever a new fact is written (add / update-via-imprint
    / supersede / coexist / delete-then-add). Callers aggregating
    "total writes" should sum `imprinted` alone.
    """
    counts = {
        "imprinted": 0, "updated": 0, "deleted": 0, "noop": 0,
        "superseded": 0, "coexisted": 0,
    }

    if action.action == "no-op":
        counts["noop"] += 1
        return counts

    if action.action == "delete" and action.target_id:
        _qdrant_delete(tm, [action.target_id])
        # Per spec: delete is followed by add (the new fact is the replacement)
        tm.imprint(content=new_fact, metadata=metadata or {})
        counts["deleted"] += 1
        counts["imprinted"] += 1
        return counts

    if action.action == "update" and action.target_id:
        _qdrant_delete(tm, [action.target_id])
        merged = action.merged_content or new_fact
        tm.imprint(content=merged, metadata=metadata or {})
        counts["updated"] += 1
        counts["imprinted"] += 1
        return counts

    if action.action == "supersede" and action.target_id:
        # NOTE (Phase 9.5 — Step 3 deferred): the soft-delete payload-patch
        # path (`_qdrant_supersede`) is not yet wired. Until it lands the
        # old fact's content is hard-deleted. Classification IS preserved
        # via the new fact's `supersedes` pointer, so downstream chain
        # traversal can still walk forward — just can't recover old text.
        # Step 3 swaps `_qdrant_delete` -> payload-patch with zero
        # contract change at this layer.
        _qdrant_delete(tm, [action.target_id])
        supersede_meta = {
            **(metadata or {}),
            "supersedes": action.target_id,
            "supersede_reason": action.supersede_reason,
            "supersede_category": action.supersede_category,
            "fact_kind": "state_evolution",
        }
        tm.imprint(content=new_fact, metadata=supersede_meta)
        counts["superseded"] += 1
        counts["imprinted"] += 1
        return counts

    if action.action == "coexist" and (action.relates_to or action.target_id):
        coexist_meta = {
            **(metadata or {}),
            "relates_to": action.relates_to or action.target_id,
            "fact_kind": "scoped_variant",
        }
        tm.imprint(content=new_fact, metadata=coexist_meta)
        counts["coexisted"] += 1
        counts["imprinted"] += 1
        return counts

    # Default: add
    tm.imprint(content=new_fact, metadata=metadata or {})
    counts["imprinted"] += 1
    return counts


def _qdrant_delete(tm: TieredMemoryLike, point_ids: list[str]) -> None:
    """Delete points by ID via Qdrant's points/delete endpoint."""
    from src.tiered_memory_qdrant import COLLECTION

    r = tm._http.post(
        f"/collections/{COLLECTION}/points/delete",
        json={"points": point_ids},
    )
    r.raise_for_status()
