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

import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from src.tiered_memory import TieredMemory


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


def _ensure_dedup_table(db_path: Path = DEDUP_DB) -> sqlite3.Connection:
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
        base_url=os.getenv("OMLX_BASE_URL"),
        api_key=os.getenv("OMLX_API_KEY"),
    )
    resp = client.chat.completions.create(
        model=os.getenv("MODEL_HAIKU", "gpt-oss-20b-MXFP4-Q8"),
        messages=[
            {"role": "system", "content": SUMMARIZE_PROMPT},
            {"role": "user", "content": scroll_text},
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
        base_url=os.getenv("OMLX_BASE_URL"),
        api_key=os.getenv("OMLX_API_KEY"),
    )
    resp = client.chat.completions.create(
        model=os.getenv("MODEL_HAIKU", "gpt-oss-20b-MXFP4-Q8"),
        messages=[
            {"role": "system", "content": ATOMIZE_PROMPT},
            {"role": "user", "content": content},
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


async def consolidate(
    tm: TieredMemory,
    max_batch: int = 50,
    campaign: str | None = None,
    promotion_threshold: float | None = None,
    use_atomisation: bool = False,
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
                    if promotion_threshold is not None and atom_conf < promotion_threshold:
                        continue
                    atom_meta: dict[str, object] = {
                        "quest_id": quest_id,
                        "agent_id": tm.agent_id,
                        "source": "guild_consolidation",
                        "subject": subject,
                        "type": atom_type,
                        "quality_score": round(atom_conf, 3),
                    }
                    tm.imprint(content=fact_content, metadata=atom_meta)
                    fact_count += 1
                if fact_count == 0:
                    result.scrolls_demoted += 1
                    continue
                result.facts_imprinted += fact_count
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
            score: float | None = None
            if promotion_threshold is not None:
                score = quality_score(summary, tm=tm)
                if score < promotion_threshold:
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

            tm.imprint(content=summary, metadata=metadata)
            result.facts_imprinted += 1
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