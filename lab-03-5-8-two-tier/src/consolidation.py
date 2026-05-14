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


@dataclass
class ConsolidationResult:
    scrolls_seen: int
    scrolls_imprinted: int
    scrolls_skipped: int
    errors: list[str]


def _ensure_dedup_table(db_path: Path = DEDUP_DB) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS imprinted (quest_id TEXT PRIMARY KEY)"
    )
    return conn


def summarize_scroll(scroll_text: str) -> str | None:
    """LLM-summarize a scroll into one semantic-fact sentence.
    Returns None if scroll should be skipped (no reusable knowledge).

    max_tokens=400 leaves room for reasoning models (e.g. gpt-oss-20b) whose
    chain-of-thought lands in `reasoning_content` BEFORE the final `content`
    is emitted. A budget of 80 (final-answer-only) gets eaten by CoT and
    leaves `content=None` + `finish_reason='length'`. The 25-word output
    cap is enforced by the system prompt, not by the token ceiling.
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


async def consolidate(
    tm: TieredMemory,
    max_batch: int = 50,
    campaign: str | None = None,
) -> ConsolidationResult:
    """One batch run. Pulls closed quests from guild, imprints into EverCore.

    Idempotency: local SQLite table tracks imprinted QUEST-IDs (EXACT match,
    not semantic search — see BCJ Entry 4 for why semantic dedup fails on
    short ID strings).

    Ordering: quests processed in QUEST-ID order (server-assigned monotonic
    integers); the latest imprint reflects the most recent state.
    """
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

    # 3. Per-quest: fetch scroll, summarize, imprint, record dedup row
    for quest_id in quest_ids:
        if quest_id in imprinted_before:
            continue
        try:
            scroll_text = await tm.get_scroll(quest_id)
            summary = summarize_scroll(scroll_text)
            if summary is None:
                result.scrolls_skipped += 1
                continue
            tm.imprint(
                content=summary,
                metadata={
                    "quest_id": quest_id,
                    "agent_id": tm.agent_id,
                    "source": "guild_consolidation",
                },
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