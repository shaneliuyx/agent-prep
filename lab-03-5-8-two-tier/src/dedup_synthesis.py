"""Online dedup-and-synthesis (Batchelor-Manning 2026 form #1).

Implements the "pay at write time" pattern: when a new fact arrives, query
the existing store for top-k semantically nearest candidates, then issue
ONE LLM call to decide an action (add / update / delete / no-op). Execute.

Article's claim from the 19-system corpus: this is the HIGHEST-ROI
write-time form — compounds across every subsequent read. Lab measures
the cost (one LLM call + delete + imprint per action) against the
recall+precision lift on subsequent retrieval.

Scoped to the Qdrant TieredMemory variant for clean composition (EverCore
has its own internal extraction pipeline that doesn't expose delete/update
hooks cleanly).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
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

NEW FACT:
{new_fact}

CANDIDATE EXISTING FACTS (top-k by semantic similarity):
{candidates}

Decide ONE action. Emit JSON:
{{"action": "add" | "update" | "delete" | "no-op",
  "target_id": "<id of existing fact, only for update/delete>",
  "merged_content": "<refined fact text, only for update>"}}

Rules:
- "add": new fact is genuinely novel; no overlap with candidates
- "update": new fact REFINES one of the candidates (additional detail,
            corrected number, expanded scope). Use the candidate's `id`
            as target_id; emit merged_content combining old + new.
- "delete": new fact CONTRADICTS one of the candidates (incompatible
            statement of the same world-state). Use the candidate's `id`
            as target_id; the caller will then add the new fact as a
            separate step. No merged_content.
- "no-op": new fact is a DUPLICATE — candidates already cover it. No imprint.

Return ONLY the JSON object. No prose, no markdown fence."""


Action = Literal["add", "update", "delete", "no-op"]


@dataclass
class DedupAction:
    action: Action
    target_id: str | None = None
    merged_content: str | None = None


def _format_candidates(candidates: list[dict]) -> str:
    """Render top-k candidates as numbered list for the LLM prompt."""
    if not candidates:
        return "(none)"
    lines = []
    for c in candidates[:5]:
        cid = c.get("id") or c.get("point_id") or "?"
        content = c.get("content") or c.get("summary") or ""
        score = c.get("score", 0.0)
        lines.append(f'  - id={cid!r}  score={score:.3f}  content="{content[:200]}"')
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
    prompt = DEDUP_PROMPT.format(
        new_fact=new_fact, candidates=_format_candidates(candidates)
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
    if action not in ("add", "update", "delete", "no-op"):
        return DedupAction(action="add")

    return DedupAction(
        action=action,
        target_id=parsed.get("target_id"),
        merged_content=parsed.get("merged_content"),
    )


def execute_action(tm: TieredMemoryLike, action: DedupAction, new_fact: str,
                   metadata: dict | None = None) -> dict:
    """Apply a DedupAction against a Qdrant TieredMemory.

    Returns a dict with counters for the caller to aggregate:
      {"imprinted": int, "updated": int, "deleted": int, "noop": int}

    Each call increments exactly one counter.
    """
    counts = {"imprinted": 0, "updated": 0, "deleted": 0, "noop": 0}

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
