"""Consolidation pipeline — moves closed guild quests into EverCore as
semantic imprints. Runs periodically (cron / scheduled task / Airflow).

Three load-bearing properties:
  1. Idempotency — local SQLite dedup table keyed by QUEST-ID
                   (semantic search over short ID strings false-negatives —
                   see Bad-Case Journal Entry 4)
  2. Ordering — quests processed in QUEST-ID order (monotonic, server-assigned)
  3. Failure handling — leave unconsolidated on EverCore failure, retry next run

NOTE on guild's API surface (W3.5.5 §1.3 BCJ): guild has NO scroll_list_closed
or scroll_mark_consolidated primitive. Closed quests come from quest_list
(status='done'); scroll text per quest comes from quest_scroll(quest_id);
'already consolidated' state lives in a local SQLite table on the consolidator
side, NOT in guild (guild's append-only lore is the wrong primitive for this).
"""
from __future__ import annotations

import json
import os
import re
import sqlite3

from src.llm_retry import chat_with_retry
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from typing import Any, Literal, Protocol

from src.audit import AuditEntry, record_audit

from typing import TYPE_CHECKING
if TYPE_CHECKING:  # avoid runtime import; ThreeTierMemory only used as a type hint
    from src.three_tier_memory import ThreeTierMemory


class TieredMemoryLike(Protocol):
    """Structural type covering both EverCore (src.tiered_memory.TieredMemory)
    and Qdrant (src.tiered_memory_qdrant.TieredMemory) variants.

    Pyright otherwise sees them as distinct classes; this Protocol asserts
    they share the contract `consolidate()` actually depends on. Shape
    matches src.dedup_synthesis.TieredMemoryLike (which adds `_http` for
    the Qdrant-specific dedup-delete codepath).
    """
    agent_id: str
    _http: Any

    async def list_closed_quests(self, campaign: str | None = None) -> str: ...
    async def get_scroll(self, quest_id: str) -> str: ...
    def imprint(self, content: str, metadata: dict[str, Any] | None = ...) -> str: ...
    def query_context(
        self,
        query: str,
        k: int = ...,
        min_confidence: float = ...,
        type_filter: list[str] | None = ...,
    ) -> list[dict[str, Any]]: ...


QUEST_ID_RE = re.compile(r"QUEST-\d+")
DEDUP_DB = Path(".guild_consolidation_state.sqlite")


SUMMARIZE_PROMPT = """Summarize this task scroll into a single semantic fact.

Output ONE sentence (MAXIMUM 25 words) describing what was learned or
accomplished, in present tense, suitable for storing as a long-term memory.

Examples:
  Scroll: "deployed-via-terraform; ran terraform apply, got 200, verified"
    Output: Production deployments use Terraform IaC pattern with apply + verify.

  Scroll: "user-auth-tokens-expire-after-30min; tested with stale token, got 401"
    Output: Authentication tokens expire after 30 minutes and return 401 when stale.

Skip scrolls that don't encode reusable knowledge (in-progress notes,
failed attempts, debug traces) — output exactly: SKIP."""


# Atomisation prompt (Batchelor-Manning 2026 form #2). Returns a JSON list
# of typed atomic facts. Each fact is ONE self-contained proposition.
ATOMIZE_PROMPT = """Extract ALL distinct atomic facts from this task scroll.

Output a JSON array. Each element:
  {"fact": str, "type": str, "confidence": number}

Rules:
- `fact`: one self-contained proposition (≤ 25 words, present tense, no anaphora)
- `type`: one of "fact" | "observation" | "tool_result" | "skill"
    - "fact": durable knowledge ("Production deploys use Terraform")
    - "observation": time/context-bound ("Today's run took 5 min")
    - "tool_result": output of a tool execution ("terraform apply returned 200")
    - "skill": reusable procedure ("To rotate auth tokens: stop service, run keycloak-rotation, restart")
- `confidence`: 0.0-1.0, your judgment of fact reliability + reusability

Output exactly `[]` (empty array) if the scroll encodes no reusable knowledge.

Example:
  Scroll: "deployed via terraform; ran apply got 200; verified VPC peering with data-lake; first-deploy budget 5 minutes"
  Output: [
    {"fact": "Production deployments use Terraform IaC with VPC peering to the data-lake account.", "type": "fact", "confidence": 0.95},
    {"fact": "First-deploy wall-clock budget is 5 minutes.", "type": "fact", "confidence": 0.9},
    {"fact": "terraform apply returned HTTP 200 on this run.", "type": "tool_result", "confidence": 0.85}
  ]

Return ONLY the JSON array. No prose, no markdown fence, no explanation."""


@dataclass
class ConsolidationResult:
    scrolls_seen: int
    scrolls_imprinted: int
    scrolls_skipped: int
    errors: list[str]
    # Quality-gate counter — incremented only when promotion_threshold is set
    # and the summary scored below it. Kept separate from scrolls_skipped so
    # operators can distinguish summarizer-SKIP from quality-gate-DEMOTE.
    scrolls_demoted: int = 0
    # Atomisation counter — total atomic facts imprinted across all scrolls.
    # facts_imprinted >= scrolls_imprinted because one scroll yields N facts.
    facts_imprinted: int = 0
    # Online-dedup counters (Batchelor-Manning form #1 + Phase 9.5
    # bitemporal extension) — only populated when use_dedup=True.
    # Each atom takes exactly one primary action.
    facts_deduplicated: int = 0   # action="no-op" — fact already known
    facts_updated: int = 0        # action="update" — same world-state correction
    facts_deleted: int = 0        # action="delete" — old fact was false
    facts_superseded: int = 0     # action="supersede" — state evolved (old kept, marked)
    facts_coexisted: int = 0      # action="coexist" — scoped variant; both true
    # L3 hyperedge counters (Phase 8 — written alongside L2 by consolidate_with_l3).
    edges_imprinted: int = 0      # typed hyperedges POSTed to HyperMem (L3)
    edges_skipped_dedup: int = 0  # edges already seen (idempotency-key hit)


def _ensure_dedup_table(db_path: Path | None = None) -> sqlite3.Connection:
    # Resolve default at CALL time so tests can monkeypatch
    # `src.consolidation.DEDUP_DB` and have it reach this function. Default-arg
    # binding evaluates DEDUP_DB at module-load time, which silently ignores
    # the patch — a real testability bug we hit on §3.4 audit-extension tests.
    if db_path is None:
        db_path = DEDUP_DB
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS imprinted (quest_id TEXT PRIMARY KEY)"
    )
    return conn


def summarize_scroll(scroll_text: str) -> str | None:
    """[Legacy] Single-summary version. Returns one ~25-word summary.

    Use extract_atomic_facts() instead for the form-#2 (atomisation)
    pipeline. Kept for backward-compat with §3.2 tests until they migrate.
    """
    client = OpenAI(
        base_url=os.getenv("LLM_BASE_URL", os.getenv("OMLX_BASE_URL")),
        api_key=os.getenv("LLM_API_KEY", os.getenv("OMLX_API_KEY")),
    )
    resp = chat_with_retry(client,
        model=os.getenv("MODEL_HAIKU", "claude-haiku-4-5-20251001"),
        # USER-role only: VibeProxy (:8317) cloaks as Claude Code on a real
        # system role and refuses non-coding tasks (W3.5.8 BCJ Entry 19). Fold
        # the instruction into the user turn — same task, no cloak.
        messages=[
            {"role": "user", "content": f"{SUMMARIZE_PROMPT}\n\n---\n\n{scroll_text}"},
        ],
        temperature=0.0,
        max_tokens=400,
    )
    summary = (resp.choices[0].message.content or "").strip()
    if summary.upper() == "SKIP" or not summary:
        return None
    return summary


def _strip_scroll_wrapper(scroll_text: str) -> str:
    """Strip guild's metadata wrapper from a scroll, keeping only the
    substantive content (journal entries + completion report).

    guild's quest_scroll() output looks like:
        📜 QUEST-N [P2 · done]  <subject>
          owner: agent
          notes: K
            · [spec] subject: X; priority: ...
            · [checkpoint] accepted by agent — starting fresh
            · [completed] <the actual report text we want>
            · [journal] <agent-written progress notes>

    The LLM is confused by the metadata header and emits 0 facts on the
    wrapped form. Pull only the lines tagged [completed] or [journal] —
    those are the substantive content. Falls back to the full text if
    no tagged lines are found (defensive).
    """
    keep = []
    for line in scroll_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("· [completed]"):
            keep.append(stripped[len("· [completed]"):].strip())
        elif stripped.startswith("· [journal]"):
            keep.append(stripped[len("· [journal]"):].strip())
    return " ".join(keep) if keep else scroll_text


def extract_atomic_facts(scroll_text: str) -> list[dict]:
    """LLM-extract N typed atomic facts from a scroll.

    Returns list of dicts {fact, type, confidence}. Empty list if scroll
    encodes no reusable knowledge (replaces old SKIP sentinel).

    max_tokens=800 gives reasoning models room for chain-of-thought AND
    JSON output (~3-5 facts × ~50 tokens each + JSON overhead).

    Resilient to malformed JSON: if parsing fails OR a fence is wrapped
    around the array, strip and retry; on second failure, fall back to
    one-fact list using the raw text as the summary (so the pipeline
    degrades gracefully instead of hard-failing on LLM output drift).
    """
    import json

    # Strip guild's metadata wrapper — the LLM extracts 0 facts on the
    # wrapped form because the header looks like noise. Substantive
    # content lives in [completed] + [journal] tagged lines only.
    content = _strip_scroll_wrapper(scroll_text)

    client = OpenAI(
        base_url=os.getenv("LLM_BASE_URL", os.getenv("OMLX_BASE_URL")),
        api_key=os.getenv("LLM_API_KEY", os.getenv("OMLX_API_KEY")),
    )
    resp = chat_with_retry(client,
        model=os.getenv("MODEL_HAIKU", "claude-haiku-4-5-20251001"),
        # USER-role only — see SUMMARIZE note above (VibeProxy system-role cloak).
        messages=[
            {"role": "user", "content": f"{ATOMIZE_PROMPT}\n\n---\n\n{content}"},
        ],
        temperature=0.0,
        max_tokens=800,
    )
    raw = (resp.choices[0].message.content or "").strip()
    if not raw or raw == "[]":
        return []

    # Strip optional ```json ... ``` fence
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        facts = json.loads(raw)
    except json.JSONDecodeError:
        # Graceful fallback: treat raw as one fact, default type+confidence.
        return [{"fact": raw[:200], "type": "fact", "confidence": 0.5}]

    if not isinstance(facts, list):
        return []

    # Validate + coerce each entry
    out = []
    for f in facts:
        if not isinstance(f, dict) or "fact" not in f:
            continue
        out.append({
            "fact": str(f["fact"])[:300],
            "type": str(f.get("type", "fact")),
            "confidence": float(f.get("confidence", 0.5)),
        })
    return out


# ── Helper: emit one quality-gate audit (§3.4 optional extension) ───────
# Centralised so atomisation + legacy paths share identical audit shape.
# Pre-write semantics: target_id=None because the fact hasn't been
# imprinted yet. On `promote`, new_id is filled by the caller AFTER the
# downstream imprint returns the UUID (best-effort — None is acceptable
# when use_dedup=True since execute_action returns counts not UUIDs).
def _audit_gate(
    *,
    decision: Literal["promote", "demote"],
    actor_agent_id: str,
    user_id: str,
    score: float,
    threshold: float,
    fact_preview: str,
    quest_id: str,
    fact_type: str = "fact",
    new_id: str | None = None,
) -> None:
    record_audit(AuditEntry(
        operation=decision,
        actor_agent_id=actor_agent_id,
        user_id=user_id,
        target_id=None,
        new_id=new_id,
        payload_summary=fact_preview[:120],
        metadata={
            "quest_id": quest_id,
            "quality_score": round(score, 3),
            "threshold": round(threshold, 3),
            "delta": round(score - threshold, 3),
            "fact_type": fact_type,
            "phase": "pre_write_gate",
        },
    ))


async def consolidate(
    tm: TieredMemoryLike,
    max_batch: int = 50,
    campaign: str | None = None,
    promotion_threshold: float | None = None,
    use_atomisation: bool = False,
    use_dedup: bool = False,
) -> ConsolidationResult:
    """One batch run. Pulls closed quests from guild, imprints into EverCore.

    Idempotency: local SQLite table tracks imprinted QUEST-IDs (EXACT match,
    not semantic search — see BCJ Entry 4 for why semantic dedup fails on
    short ID strings).

    Ordering: quests processed in QUEST-ID order (server-assigned monotonic
    integers); the latest imprint reflects the most recent state.

    Promotion gate (§3.3): when `promotion_threshold` is a float in [0.0, 1.0],
    each summarized scroll is scored via `quality_gate.quality_score()` and
    only imprinted if `score >= promotion_threshold`. Demoted scrolls land
    in `result.scrolls_demoted`. When `promotion_threshold is None`, the
    gate is bypassed and every non-SKIP summary imprints (legacy behavior).

    Atomisation (Batchelor-Manning 2026 form #2): when `use_atomisation=True`,
    extract N typed atomic facts per scroll via `extract_atomic_facts()` and
    imprint each as a separate memory with its own type + confidence in
    metadata. `result.facts_imprinted` counts the per-fact imprints (always
    >= scrolls_imprinted). When False (default), uses the legacy single-
    summary path via `summarize_scroll()` — keeps §3.2 tests passing.
    """
    from src.quality_gate import quality_score  # local import avoids circular ref
    # 1. List closed quests via quest_list(status='done')
    list_text = await tm.list_closed_quests(campaign=campaign)
    # Numerical sort by the integer suffix of QUEST-N. Plain `sorted()` is
    # alphabetical, which orders QUEST-1, QUEST-10, QUEST-11, ..., QUEST-2,
    # QUEST-20 — and silently never reaches QUEST-100+ in production once
    # the system has accumulated three-digit quests. Sort numerically and
    # take the OLDEST `max_batch` (lowest QUEST-N) so consolidation never
    # leaves long-waiting quests stranded behind a flood of newer ones.
    quest_ids = sorted(
        set(QUEST_ID_RE.findall(list_text)),
        key=lambda q: int(q.split("-", 1)[1]),
    )[:max_batch]

    # 2. Load local dedup state
    dedup = _ensure_dedup_table()
    imprinted_before = {
        row[0] for row in dedup.execute("SELECT quest_id FROM imprinted")
    }

    result = ConsolidationResult(
        scrolls_seen=len(quest_ids),
        scrolls_imprinted=0,
        scrolls_skipped=0,
        errors=[],
    )
    # Audit-emitter context — captured once per batch (cheap).
    # getattr() falls back gracefully if a future backend omits user_id.
    actor_id = tm.agent_id
    user_id = getattr(tm, "user_id", "")

    # 3. Per-quest: fetch scroll, summarize (or atomise), imprint, record dedup row
    for quest_id in quest_ids:
        if quest_id in imprinted_before:
            continue
        try:
            scroll_text = await tm.get_scroll(quest_id)

            # Derive subject once — used by both atomisation + legacy paths.
            subject = scroll_text.split("\n", 1)[0][:80].strip() or quest_id

            if use_atomisation:
                # Form #2 (atomisation): N typed facts per scroll.
                atoms = extract_atomic_facts(scroll_text)
                if not atoms:
                    result.scrolls_skipped += 1
                    continue
                # Imprint each atomic fact as a separate memory.
                fact_count = 0
                for atom in atoms:
                    fact_content = atom["fact"]
                    atom_type = atom["type"]
                    atom_conf = atom["confidence"]
                    # Quality gate applies per-atom on its self-reported confidence.
                    # §3.4 audit extension: emit demote (skip) or track promote
                    # (deferred until after imprint so new_id can be chained).
                    gate_passed_score: float | None = None
                    if promotion_threshold is not None:
                        if atom_conf < promotion_threshold:
                            _audit_gate(
                                decision="demote",
                                actor_agent_id=actor_id,
                                user_id=user_id,
                                score=atom_conf,
                                threshold=promotion_threshold,
                                fact_preview=fact_content,
                                quest_id=quest_id,
                                fact_type=atom_type,
                            )
                            continue
                        gate_passed_score = atom_conf
                    atom_meta: dict[str, object] = {
                        "quest_id": quest_id,
                        "agent_id": tm.agent_id,
                        "source": "guild_consolidation",
                        "subject": subject,
                        "type": atom_type,
                        "quality_score": round(atom_conf, 3),
                    }
                    new_point_id: str | None = None
                    if use_dedup:
                        # Form #1 (online dedup-and-synthesis): query top-k,
                        # LLM decides add/update/delete/no-op, execute.
                        from src.dedup_synthesis import decide_action, execute_action
                        candidates = tm.query_context(fact_content, k=5)
                        action = decide_action(fact_content, candidates)
                        counts = execute_action(
                            tm, action, fact_content, metadata=atom_meta
                        )
                        result.facts_imprinted += counts["imprinted"]
                        result.facts_updated += counts["updated"]
                        result.facts_deleted += counts["deleted"]
                        result.facts_deduplicated += counts["noop"]
                        result.facts_superseded += counts.get("superseded", 0)
                        result.facts_coexisted += counts.get("coexisted", 0)
                        # `fact_count` tracks any non-noop action so the scroll
                        # itself still counts as "imprinted" downstream.
                        if action.action != "no-op":
                            fact_count += 1
                        # new_point_id stays None — execute_action returns
                        # counts, not UUIDs. Chain via metadata.quest_id.
                    else:
                        new_point_id = tm.imprint(content=fact_content, metadata=atom_meta)
                        result.facts_imprinted += 1
                        fact_count += 1
                    # §3.4 audit extension: emit promote AFTER imprint so
                    # new_id can chain to the gate decision.
                    if gate_passed_score is not None:
                        _audit_gate(
                            decision="promote",
                            actor_agent_id=actor_id,
                            user_id=user_id,
                            score=gate_passed_score,
                            threshold=promotion_threshold,  # type: ignore[arg-type]
                            fact_preview=fact_content,
                            quest_id=quest_id,
                            fact_type=atom_type,
                            new_id=new_point_id,  # None if use_dedup=True
                        )
                if fact_count == 0:
                    result.scrolls_demoted += 1
                    continue
                dedup.execute(
                    "INSERT OR IGNORE INTO imprinted (quest_id) VALUES (?)",
                    (quest_id,),
                )
                dedup.commit()
                result.scrolls_imprinted += 1
                continue

            # Legacy single-summary path (default for backwards compat with §3.2 tests).
            summary = summarize_scroll(scroll_text)
            if summary is None:
                result.scrolls_skipped += 1
                continue

            # §3.3 quality-gate check before imprint (active iff threshold set).
            # §3.4 audit extension: emit demote (skip) OR defer promote until
            # after imprint so new_id can chain.
            score: float | None = None
            if promotion_threshold is not None:
                score = quality_score(summary, tm=tm)
                if score < promotion_threshold:
                    _audit_gate(
                        decision="demote",
                        actor_agent_id=actor_id,
                        user_id=user_id,
                        score=score,
                        threshold=promotion_threshold,
                        fact_preview=summary,
                        quest_id=quest_id,
                        fact_type="fact",
                    )
                    result.scrolls_demoted += 1
                    continue

            metadata: dict[str, object] = {
                "quest_id": quest_id,
                "agent_id": tm.agent_id,
                "source": "guild_consolidation",
                "subject": subject,
                "type": "fact",
            }
            if score is not None:
                metadata["quality_score"] = round(score, 3)

            new_summary_point_id = tm.imprint(content=summary, metadata=metadata)
            result.facts_imprinted += 1
            # §3.4 audit extension: promote AFTER imprint with new_id chained.
            if score is not None:
                _audit_gate(
                    decision="promote",
                    actor_agent_id=actor_id,
                    user_id=user_id,
                    score=score,
                    threshold=promotion_threshold,  # type: ignore[arg-type]
                    fact_preview=summary,
                    quest_id=quest_id,
                    fact_type="fact",
                    new_id=new_summary_point_id,
                )
            dedup.execute(
                "INSERT OR IGNORE INTO imprinted (quest_id) VALUES (?)",
                (quest_id,),
            )
            dedup.commit()
            result.scrolls_imprinted += 1
        except Exception as e:                                       # noqa: BLE001
            result.errors.append(f"{quest_id}: {type(e).__name__}: {e}")

    dedup.close()
    return result


# ── Phase 8: L3 (HyperMem) hyperedge extension ───────────────────────
# Extends the L2 consolidate() above with typed-hyperedge extraction written
# to the L3 HyperMem tier. NOTE the adaptation vs the chapter sketch: the real
# consolidate() is async and pulls closed quests from guild itself (no scrolls
# arg), so consolidate_with_l3() awaits it for the L2 path and takes an explicit
# `scrolls` list only for L3 edge extraction. Edge writes are idempotent via a
# dedup table (mirrors _ensure_dedup_table's `imprinted` pattern).
import hashlib

EDGE_EXTRACT_PROMPT = """Extract typed entity-relations from this scroll.
Each relation is a hyperedge connecting >=2 typed entities.

Entity types: user, project, topic, tech, person, system, event
Relations: worked-on, uses, depends-on, mentions, after, before, related-to

Output JSON array of {nodes: [{type, id}, ...], relation: <verb>}.
Output ONLY the JSON array. If no extractable relations, output [].

SCROLL: {scroll_text}"""


def _ensure_edge_dedup_table(db_path: Path | None = None) -> sqlite3.Connection:
    """Edge idempotency table (twin of _ensure_dedup_table's `imprinted`)."""
    if db_path is None:
        db_path = DEDUP_DB
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE IF NOT EXISTS edges_imprinted (edge_key TEXT PRIMARY KEY)")
    return conn


def _edge_already_imprinted(key: str, db_path: Path | None = None) -> bool:
    conn = _ensure_edge_dedup_table(db_path)
    try:
        row = conn.execute(
            "SELECT 1 FROM edges_imprinted WHERE edge_key = ?", (key,)
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def _record_edge_imprint(key: str, db_path: Path | None = None) -> None:
    conn = _ensure_edge_dedup_table(db_path)
    try:
        conn.execute("INSERT OR IGNORE INTO edges_imprinted (edge_key) VALUES (?)", (key,))
        conn.commit()
    finally:
        conn.close()


def _edge_idempotency_key(scroll_id: str, edge: dict) -> str:
    """Idempotent hash: scroll_id + relation + canonicalized sorted entity list."""
    canonical_nodes = sorted(f"{n['type']}:{n['id']}" for n in edge["nodes"])
    payload = f"{scroll_id}|{edge['relation']}|{'|'.join(canonical_nodes)}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def extract_typed_edges(scroll_text: str) -> list[dict]:
    """One LLM call -> JSON array of typed hyperedges (same client pattern as
    summarize_scroll). Returns [] on empty/parse failure."""
    client = OpenAI(base_url=os.getenv("LLM_BASE_URL", os.getenv("OMLX_BASE_URL")), api_key=os.getenv("LLM_API_KEY", os.getenv("OMLX_API_KEY")))
    resp = chat_with_retry(client,
        model=os.getenv("MODEL_HAIKU", "claude-haiku-4-5-20251001"),
        messages=[{"role": "user", "content": EDGE_EXTRACT_PROMPT.format(scroll_text=scroll_text)}],
        temperature=0.0,
        max_tokens=800,
    )
    raw = (resp.choices[0].message.content or "").strip()
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


async def consolidate_with_l3(
    tm: "ThreeTierMemory",
    scrolls: list[dict],
    promotion_threshold: float | None = None,
) -> ConsolidationResult:
    """Phase 8 extended consolidate: L2 imprints (via the async consolidate())
    PLUS L3 typed hyperedges POSTed to HyperMem.

    `scrolls` is a list of {"quest_id": str, "text": str} for the L3 edge pass.
    L3 writes are deduped by _edge_idempotency_key so re-runs are idempotent.
    """
    # L2 path — unchanged behavior; consolidate() pulls its own closed quests.
    result = await consolidate(tm, promotion_threshold=promotion_threshold)

    # L3 extension — extract + write typed hyperedges per supplied scroll.
    edges_imprinted = 0
    edges_skipped_dedup = 0
    for scroll in scrolls:
        for edge in extract_typed_edges(scroll["text"]):
            key = _edge_idempotency_key(scroll["quest_id"], edge)
            if _edge_already_imprinted(key):
                edges_skipped_dedup += 1
                continue
            tm._hypermem.post("/api/v1/edges", json={
                **edge,
                "user_id": tm.user_id,
                "provenance_scroll": scroll["quest_id"],
                "idempotency_key": key,
            })
            _record_edge_imprint(key)
            edges_imprinted += 1

    result.edges_imprinted = edges_imprinted
    result.edges_skipped_dedup = edges_skipped_dedup
    return result
