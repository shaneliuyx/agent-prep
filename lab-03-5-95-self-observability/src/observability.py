"""W3.5.95 — OBSERVABILITY: the agent's append-only behavioral log, treated as a
first-class memory tier (read at decision time), not a debug artifact.

Two SQLite tables live here:
  OBSERVABILITY — one append-only row per tool call / decision / outcome.
  LEARNING      — typed self-pattern facts (written ONLY by learning_extractor).

Design (chapter §2.2):
  * append-only: PK (agent_run_id, step_idx); no updates/deletes except retention.
  * PII-scrubbed at the WRITE boundary (see pii_scrub) — raw tool args routinely
    carry keys/paths/PII; scrub before persisting (opt-out via raw_args=True).
  * indexed for the recall layer's query patterns: (tool_name, ts), (run, step).
"""
from __future__ import annotations

import json
import sqlite3
import time
from typing import Any

# PII / secret scrubbing at the write boundary lives in pii_scrub: Microsoft
# Presidio (NER + pattern recognizers) when available, regex fallback otherwise.
# Re-export so callers can use obs.scrub_pii unchanged. (See the pii_scrub block.)
from pii_scrub import scrub_pii  # noqa: F401


_SCHEMA = """
CREATE TABLE IF NOT EXISTS observability (
    ts            REAL NOT NULL,
    agent_run_id  TEXT NOT NULL,
    step_idx      INTEGER NOT NULL,
    tool_name     TEXT NOT NULL,
    args_json     TEXT NOT NULL,
    outcome_json  TEXT NOT NULL,
    outcome_status TEXT NOT NULL,   -- ok | error | timeout
    latency_ms    REAL NOT NULL,
    user_signal   TEXT,             -- thumbs_up | thumbs_down | silent | NULL
    PRIMARY KEY (agent_run_id, step_idx)
);
CREATE INDEX IF NOT EXISTS ix_obs_tool_ts ON observability (tool_name, ts);
CREATE INDEX IF NOT EXISTS ix_obs_ts      ON observability (ts);

CREATE TABLE IF NOT EXISTS learning (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    type         TEXT NOT NULL,     -- failure_pattern|success_pattern|tool_preference|recurring_mistake
    pattern_text TEXT NOT NULL,
    confidence   REAL NOT NULL,
    is_self_caused INTEGER NOT NULL,  -- 1 = the agent's own pattern; 0 = environmental (filtered out)
    source_rows  TEXT NOT NULL,     -- JSON list of (run_id, step_idx) provenance
    ts           REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_learn_type ON learning (type, ts);
"""


def connect(db_path: str, *, check_same_thread: bool = True) -> sqlite3.Connection:
    """Open the memory DB, ensure both tables + indices exist. WAL for concurrent
    read (the recall layer reads while the agent writes).

    check_same_thread=False is needed when an agent FRAMEWORK executes the
    instrumented tools off the connection's creating thread — e.g. smolagents'
    CodeAgent runs the model's code in a worker thread, so the tool's
    log_observation() write happens on a different thread than connect(). WAL +
    the GIL + smolagents' sequential single-tool-at-a-time loop keep writes
    serialized, so this is safe here. (BCJ Entry 6.)"""
    conn = sqlite3.connect(db_path, check_same_thread=check_same_thread)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    return conn


def log_observation(
    conn: sqlite3.Connection, *, agent_run_id: str, step_idx: int, tool_name: str,
    args: Any, outcome: Any, outcome_status: str, latency_ms: float,
    user_signal: str | None = None, raw_args: bool = False,
) -> None:
    """Append ONE observation row. args/outcome are JSON-serialized; args are
    PII-scrubbed unless raw_args=True (debug opt-out). Append-only: a duplicate
    (run_id, step_idx) is a programming error — surfaced, not silently updated."""
    args_json = json.dumps(args, default=str)
    if not raw_args:
        args_json = scrub_pii(args_json)
    outcome_json = scrub_pii(json.dumps(outcome, default=str))[:2000]  # truncate + scrub
    conn.execute(
        "INSERT INTO observability (ts, agent_run_id, step_idx, tool_name, args_json, "
        "outcome_json, outcome_status, latency_ms, user_signal) VALUES (?,?,?,?,?,?,?,?,?)",
        (time.time(), agent_run_id, step_idx, tool_name, args_json, outcome_json,
         outcome_status, latency_ms, user_signal),
    )
    conn.commit()


def recent_observations(conn: sqlite3.Connection, limit: int = 50) -> list[sqlite3.Row]:
    """Last `limit` rows by time — the recall layer's 'what did I just do' window."""
    return conn.execute(
        "SELECT * FROM observability ORDER BY ts DESC LIMIT ?", (limit,)
    ).fetchall()


def observations_by_tool(conn: sqlite3.Connection, tool_name: str) -> list[sqlite3.Row]:
    """All rows for one tool (uses ix_obs_tool_ts) — 'how has X behaved for me'."""
    return conn.execute(
        "SELECT * FROM observability WHERE tool_name = ? ORDER BY ts DESC", (tool_name,)
    ).fetchall()


def observations_by_run(conn: sqlite3.Connection, agent_run_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM observability WHERE agent_run_id = ? ORDER BY step_idx", (agent_run_id,)
    ).fetchall()


def stamp_user_signal(conn: sqlite3.Connection, agent_run_id: str, step_idx: int,
                      signal: str) -> None:
    """The ONE permitted mutation: attach a user-satisfaction signal to a row
    after the fact (the user reacts to an outcome the agent already produced)."""
    conn.execute(
        "UPDATE observability SET user_signal=? WHERE agent_run_id=? AND step_idx=?",
        (signal, agent_run_id, step_idx),
    )
    conn.commit()
