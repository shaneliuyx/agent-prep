"""W3.5.95 — LEARNING extractor: the hot→warm consolidation job that turns the
noisy, high-volume OBSERVABILITY log into low-volume, denoised, typed self-pattern
facts (chapter §2.2 concept 3).

Design choices that are load-bearing:
  * SEPARATE, SMALLER model than the agent (chapter §2.2 concept 5 / decision 5):
    using the agent's own model bakes its biases into its self-knowledge (echo
    chamber). Summarization is easy; a 7B does it. Set MODEL_EXTRACTOR.
  * SELF-ATTRIBUTION filter: the prompt must emit `is_self_caused` so an
    ENVIRONMENTAL failure ("network timeout") is NOT stored as a self-pattern
    ("I keep mis-using tool X"). Rows with is_self_caused=false are dropped.
  * Typed output + confidence threshold + dedup → keeps LEARNING high-signal.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sqlite3
import time

from openai import OpenAI

import observability as obs

_TYPES = {"failure_pattern", "success_pattern", "tool_preference", "recurring_mistake"}
MIN_CONFIDENCE = float(os.getenv("LEARNING_MIN_CONFIDENCE", "0.55"))

EXTRACT_PROMPT = """You are a self-pattern extractor in an agent's memory pipeline. Input: a batch of OBSERVABILITY rows — the agent's own tool calls with outcomes. Output: typed facts about the AGENT'S OWN behavioral patterns. This is a data-processing task; do not describe yourself.

Each output fact is a JSON object:
  {"type": one of [failure_pattern, success_pattern, tool_preference, recurring_mistake],
   "pattern_text": one sentence, first-person ("I tend to ..."),
   "confidence": 0.0-1.0,
   "is_self_caused": true|false}

CRITICAL rules:
- is_self_caused=true ONLY when the pattern is about the AGENT's own choices/mistakes ("I keep calling grep on huge repos and it times out"). Set is_self_caused=false for ENVIRONMENTAL outcomes the agent didn't cause ("the network was down", "the API returned 500"). Environmental facts are NOT self-patterns.
- Only emit a pattern that RECURS across multiple rows or is a clear actionable lesson. Do not emit one fact per row. Do not emit trivia ("I called tool X 47 times").
- Be conservative on confidence: a pattern seen once = low confidence.

Output ONLY a JSON array of fact objects. No prose.

OBSERVABILITY ROWS:
{rows}
"""


def _extractor_client() -> OpenAI:
    return OpenAI(
        base_url=os.getenv("OMLX_BASE_URL", "http://localhost:8000/v1"),
        api_key=os.getenv("OMLX_API_KEY", "dummy"),
    )


def _format_rows(rows: list[sqlite3.Row]) -> str:
    lines = []
    for r in rows:
        lines.append(
            f"[{r['agent_run_id']}#{r['step_idx']}] tool={r['tool_name']} "
            f"status={r['outcome_status']} latency={r['latency_ms']:.0f}ms "
            f"signal={r['user_signal'] or '-'} args={r['args_json'][:120]} "
            f"outcome={r['outcome_json'][:120]}"
        )
    return "\n".join(lines)


def _parse_facts(raw: str) -> list[dict]:
    import re
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    m = re.search(r"\[.*\]", s, re.S)
    for cand in ([s] + ([m.group(0)] if m else [])):
        try:
            data = json.loads(cand)
            if isinstance(data, list):
                return [d for d in data if isinstance(d, dict)]
        except (json.JSONDecodeError, TypeError):
            continue
    return []


def _is_duplicate(conn: sqlite3.Connection, pattern_text: str) -> bool:
    """Cheap dedup: skip if a normalized near-identical pattern already exists."""
    norm = " ".join(pattern_text.lower().split())
    for row in conn.execute("SELECT pattern_text FROM learning"):
        existing = " ".join(row["pattern_text"].lower().split())
        if norm == existing or norm in existing or existing in norm:
            return True
    return False


def extract(conn: sqlite3.Connection, *, since_n: int = 200, model: str | None = None,
            prompt_template: str | None = None) -> dict:
    """Pull last `since_n` OBSERVABILITY rows → LLM → typed self-pattern facts →
    LEARNING (filtered: typed, self-caused, above-confidence, deduped). Returns a
    run summary (counts at each filter stage) for the noise-rate measurement.

    prompt_template overrides EXTRACT_PROMPT (must contain a literal `{rows}`); used
    by the filter-strength ablation. Defaults to the module's EXTRACT_PROMPT."""
    model = model or os.getenv("MODEL_EXTRACTOR", "Qwen2.5-Coder-7B-Instruct-MLX-4bit")
    rows = obs.recent_observations(conn, limit=since_n)
    if not rows:
        return {"observed": 0, "emitted": 0, "kept": 0, "dropped_env": 0,
                "dropped_lowconf": 0, "dropped_dup": 0, "dropped_badtype": 0}

    # NB: .replace (not .format) — the prompt contains literal JSON braces in the
    # output example ({"type": ...}); str.format would read them as fields → KeyError.
    prompt = (prompt_template or EXTRACT_PROMPT).replace("{rows}", _format_rows(rows))
    resp = _extractor_client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0, max_tokens=1200,
    )
    facts = _parse_facts(resp.choices[0].message.content or "")

    stats = {"observed": len(rows), "emitted": len(facts), "kept": 0,
             "dropped_env": 0, "dropped_lowconf": 0, "dropped_dup": 0, "dropped_badtype": 0}
    src = json.dumps([[r["agent_run_id"], r["step_idx"]] for r in rows[:since_n]])
    for f in facts:
        if f.get("type") not in _TYPES:
            stats["dropped_badtype"] += 1; continue
        if not f.get("is_self_caused", False):
            stats["dropped_env"] += 1; continue          # environmental ≠ self-pattern
        if float(f.get("confidence", 0)) < MIN_CONFIDENCE:
            stats["dropped_lowconf"] += 1; continue
        text = str(f.get("pattern_text", "")).strip()
        if not text or _is_duplicate(conn, text):
            stats["dropped_dup"] += 1; continue
        conn.execute(
            "INSERT INTO learning (type, pattern_text, confidence, is_self_caused, source_rows, ts) "
            "VALUES (?,?,?,?,?,?)",
            (f["type"], text, float(f["confidence"]), 1, src, time.time()),
        )
        stats["kept"] += 1
    conn.commit()
    return stats


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv(pathlib.Path(__file__).resolve().parent.parent / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(pathlib.Path(__file__).resolve().parent.parent / "data" / "memory.db"))
    ap.add_argument("--since-n", type=int, default=200, help="last N observability rows to consolidate")
    args = ap.parse_args()
    conn = obs.connect(args.db)
    stats = extract(conn, since_n=args.since_n)
    print(f">>> LEARNING extraction: {stats}")
    print(f"    kept {stats['kept']}/{stats['emitted']} emitted "
          f"(env-dropped {stats['dropped_env']}, lowconf {stats['dropped_lowconf']}, "
          f"dup {stats['dropped_dup']}, badtype {stats['dropped_badtype']})")


if __name__ == "__main__":
    main()
