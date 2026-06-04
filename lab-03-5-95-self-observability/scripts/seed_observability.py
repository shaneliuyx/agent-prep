"""Seed synthetic OBSERVABILITY rows so the LEARNING extractor + paired-trial have
real data to run against (no W4 lab needed).

Encodes THREE recurring SELF-patterns the extractor should surface, a batch of
ENVIRONMENTAL failures it must DROP (self-attribution filter), and noise it
should ignore. Deterministic — re-runnable for a fresh measurement.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import observability as obs  # noqa: E402

DB = pathlib.Path(__file__).resolve().parent.parent / "data" / "memory.db"

# (tool, args, outcome, status, latency_ms) templates — repeated to make patterns RECUR.
SELF_PATTERNS = [
    # 1. grep times out on large repos (self-caused: agent keeps choosing grep there)
    ("grep", {"query": "callers of parse", "scope": "large monorepo"},
     {"note": "timed out after 30s on the large repo"}, "timeout", 30000.0),
    # 2. find is slow on deep trees (agent defaults to find)
    ("find", {"name": "config.yaml", "scope": "deeply nested project"},
     {"note": "took 18s walking the deep tree"}, "ok", 18000.0),
    # 3. agent over-uses web_search for things already in local notes
    ("web_search", {"q": "my caching decision"},
     {"note": "found nothing; the answer was in local notes the whole time"}, "ok", 2200.0),
]
ENVIRONMENTAL = [  # must be DROPPED — not the agent's fault
    ("sql_query", {"q": "SELECT * FROM users"},
     {"note": "database connection refused — network was down"}, "error", 500.0),
    ("web_search", {"q": "weather"},
     {"note": "provider returned HTTP 500"}, "error", 800.0),
]
NOISE = [  # ordinary successful calls, no pattern
    ("read_local_notes", {"file": "notes.md"}, {"note": "read 2KB"}, "ok", 12.0),
    ("python_repl", {"code": "2+2"}, {"note": "4"}, "ok", 8.0),
    ("rg", {"query": "TODO"}, {"note": "3 hits"}, "ok", 90.0),
]


def main() -> None:
    DB.parent.mkdir(exist_ok=True)
    conn = obs.connect(str(DB))
    conn.execute("DELETE FROM observability")  # fresh seed
    conn.commit()
    step = 0
    # Each self-pattern recurs 6×; environmental 4× each; noise once each (×3 cycles).
    for _ in range(6):
        for tool, args, outcome, status, lat in SELF_PATTERNS:
            obs.log_observation(conn, agent_run_id=f"seed-{step//9}", step_idx=step,
                                tool_name=tool, args=args, outcome=outcome,
                                outcome_status=status, latency_ms=lat); step += 1
    for _ in range(4):
        for tool, args, outcome, status, lat in ENVIRONMENTAL:
            obs.log_observation(conn, agent_run_id=f"seed-{step//9}", step_idx=step,
                                tool_name=tool, args=args, outcome=outcome,
                                outcome_status=status, latency_ms=lat); step += 1
    for _ in range(3):
        for tool, args, outcome, status, lat in NOISE:
            obs.log_observation(conn, agent_run_id=f"seed-{step//9}", step_idx=step,
                                tool_name=tool, args=args, outcome=outcome,
                                outcome_status=status, latency_ms=lat); step += 1
    n = conn.execute("SELECT COUNT(*) c FROM observability").fetchone()["c"]
    print(f">>> seeded {n} OBSERVABILITY rows "
          f"({6*len(SELF_PATTERNS)} self-pattern, {4*len(ENVIRONMENTAL)} environmental, "
          f"{3*len(NOISE)} noise) → {DB}")


if __name__ == "__main__":
    main()
