"""Vote layer — runs both classifiers in parallel, agrees or escalates.

Tie-break on disagreement: prefer the HEAVIER tier (opus > sonnet > haiku)
and the HEAVIER mode (deliberate > react > minimal). Safety bias —
over-spending compute is recoverable; under-spending produces a bad answer.

Every disagreement is logged to SQLite for offline calibration analysis.
The disagreement log is the cheapest signal for "the taxonomy needs work".
"""
from __future__ import annotations

import asyncio
import sqlite3
import time
from pathlib import Path

from src.router import RouterVerdict, classify
from src.router_bart import classify_bart


LOG_DB = Path(".router_vote_log.sqlite")
TIER_ORDER = ["haiku", "sonnet", "opus"]
MODE_ORDER = ["minimal", "react", "deliberate"]


def _ensure_log() -> sqlite3.Connection:
    conn = sqlite3.connect(LOG_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS disagreements (
            ts REAL,
            prompt TEXT,
            qwen_tier TEXT, qwen_mode TEXT, qwen_conf REAL,
            bart_tier TEXT, bart_mode TEXT, bart_conf REAL,
            final_tier TEXT, final_mode TEXT
        )
    """)
    return conn


def _max(a: str, b: str, order: list[str]) -> str:
    return a if order.index(a) >= order.index(b) else b


async def router_vote(prompt: str) -> RouterVerdict:
    """Run both classifiers in parallel. Agree → emit. Disagree → escalate."""
    # Run in parallel; BART is sync but we wrap it via to_thread
    qwen_task = asyncio.to_thread(classify, prompt)
    bart_task = asyncio.to_thread(classify_bart, prompt)
    qwen_v, bart_v = await asyncio.gather(qwen_task, bart_task)

    agree = (qwen_v.tier == bart_v.tier and qwen_v.mode == bart_v.mode)

    if agree:
        return qwen_v

    # Disagreement → escalate to the heavier of the two on each axis
    final_tier = _max(qwen_v.tier, bart_v.tier, TIER_ORDER)
    final_mode = _max(qwen_v.mode, bart_v.mode, MODE_ORDER)
    final = RouterVerdict(tier=final_tier, mode=final_mode, confidence=0.5)

    # Log
    conn = _ensure_log()
    conn.execute(
        "INSERT INTO disagreements VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (time.time(), prompt[:500],
         qwen_v.tier, qwen_v.mode, qwen_v.confidence,
         bart_v.tier, bart_v.mode, bart_v.confidence,
         final.tier, final.mode),
    )
    conn.commit()
    conn.close()
    return final