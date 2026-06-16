"""
src/obs.py — SQLite observability for the ReAct loop.

Schema: one row per agent event (tool call OR final answer OR max_iter exceeded).
The table name is 'agent_events'. The database is at data/agent_obs.db.

DDL (also in the module-level INIT_DDL string for reference):

  CREATE TABLE IF NOT EXISTS agent_events (
      id                INTEGER PRIMARY KEY AUTOINCREMENT,
      ts                TEXT    NOT NULL,          -- ISO-8601 UTC timestamp
      run_id            TEXT    NOT NULL,          -- ties all events for one agent_run() together
      iteration         INTEGER NOT NULL,          -- loop iteration index (0-based)
      event_type        TEXT    NOT NULL,          -- 'tool_call' | 'final_answer' | 'max_iter_exceeded'
      tool_name         TEXT,                      -- NULL for final_answer events
      prompt_tokens     INTEGER NOT NULL DEFAULT 0,
      completion_tokens INTEGER NOT NULL DEFAULT 0,
      tool_latency_ms   INTEGER NOT NULL DEFAULT 0,
      tool_error        TEXT                       -- NULL if no error
  );

Usage:
  from src.obs import log_event, query_run
  log_event(run_id="run_123", iteration=0, event_type="tool_call", ...)
  rows = query_run("run_123")
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_DB_PATH = Path(__file__).parent.parent / "data" / "agent_obs.db"

INIT_DDL = """
CREATE TABLE IF NOT EXISTS agent_events (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                TEXT    NOT NULL,
    run_id            TEXT    NOT NULL,
    iteration         INTEGER NOT NULL,
    event_type        TEXT    NOT NULL,
    tool_name         TEXT,
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    tool_latency_ms   INTEGER NOT NULL DEFAULT 0,
    tool_error        TEXT
);

CREATE INDEX IF NOT EXISTS idx_agent_events_run_id ON agent_events (run_id);
CREATE INDEX IF NOT EXISTS idx_agent_events_ts     ON agent_events (ts);
"""


def _conn() -> sqlite3.Connection:
    """Open (or create) the SQLite database; initialize schema on first open."""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.executescript(INIT_DDL)
    conn.commit()
    return conn


def log_event(
    run_id: str,
    iteration: int,
    event_type: str,
    tool_name: str | None,
    prompt_tokens: int,
    completion_tokens: int,
    tool_latency_ms: int,
    tool_error: str | None,
) -> None:
    """Insert one event row. Called from agent_run() on every iteration."""
    ts = datetime.now(timezone.utc).isoformat()
    conn = _conn()
    conn.execute(
        """
        INSERT INTO agent_events
            (ts, run_id, iteration, event_type, tool_name,
             prompt_tokens, completion_tokens, tool_latency_ms, tool_error)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (ts, run_id, iteration, event_type, tool_name,
         prompt_tokens, completion_tokens, tool_latency_ms, tool_error),
    )
    conn.commit()
    conn.close()


def query_run(run_id: str) -> list[dict[str, Any]]:
    """Return all events for a given run_id, ordered by iteration."""
    conn = _conn()
    cur = conn.execute(
        "SELECT * FROM agent_events WHERE run_id = ? ORDER BY iteration, id",
        (run_id,),
    )
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, row)) for row in cur.fetchall()]
    conn.close()
    return rows


def run_summary(run_id: str) -> dict[str, Any]:
    """Aggregate stats for one run — total tokens, tool call count, error count."""
    conn = _conn()
    cur = conn.execute(
        """
        SELECT
            COUNT(*)                                    AS total_events,
            SUM(prompt_tokens)                          AS total_prompt_tokens,
            SUM(completion_tokens)                      AS total_completion_tokens,
            SUM(CASE WHEN tool_name IS NOT NULL THEN 1 ELSE 0 END) AS tool_calls,
            SUM(CASE WHEN tool_error IS NOT NULL THEN 1 ELSE 0 END) AS tool_errors,
            MAX(tool_latency_ms)                        AS max_tool_latency_ms,
            AVG(tool_latency_ms)                        AS avg_tool_latency_ms
        FROM agent_events
        WHERE run_id = ?
        """,
        (run_id,),
    )
    cols = [d[0] for d in cur.description]
    row = cur.fetchone()
    conn.close()
    return dict(zip(cols, row)) if row else {}