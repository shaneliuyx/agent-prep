"""cost_meter.py — per-node cost accounting with retry-safe idempotency.

WHY this module exists, separately from graph_store:
A durable runtime *retries* nodes (graph_store.mark_failed requeues a node up to
max_attempts). The naive cost ledger double-bills every retry: the same node runs
twice, you record two cost rows, your dashboard over-reports spend. The fix is an
idempotency key — UNIQUE(run_id, node_name, attempt) — plus INSERT OR IGNORE, so
re-recording the SAME (node, attempt) is a no-op. This mirrors the W4.5 lesson:
the cost meter is a side-ledger keyed by the *unit of work*, not by wall-clock
events, so it stays correct under the exact failure mode (retry) the runtime is
built to survive.

WHY record a cloud-equivalent USD when local oMLX is $0:
The headline artifact of this lab is "topology changes throughput". A *routing*
or *topology* win is only legible if you price tokens against a real cloud
baseline — otherwise every number is $0 and the comparison is invisible. So we
record local latency truthfully but attach a cloud-equivalent USD from a public
rate card. The local run is free; the number tells you what it WOULD cost in the
cloud, which is the decision-relevant quantity (build-vs-buy, route-vs-not).
"""
from __future__ import annotations

import csv
import sqlite3
import time
from types import TracebackType
from typing import Any

# Representative public per-1M-token rates (USD), input/output, as a cloud
# baseline. These are tier anchors (haiku/sonnet/opus class), not a live price
# feed — they exist so routing/topology wins are measurable, not to bill anyone.
RATE_CARD: dict[str, tuple[float, float]] = {
    # model/tier            (usd_per_1M_in, usd_per_1M_out)
    "haiku": (0.80, 4.00),
    "sonnet": (3.00, 15.00),
    "opus": (15.00, 75.00),
    # local oMLX model id maps to the haiku tier as its cloud-equivalent anchor
    "Qwen2.5-Coder-7B-Instruct-MLX-4bit": (0.80, 4.00),
    "Qwen2.5-Coder-14B-Instruct-MLX-4bit": (0.80, 4.00),
}
_DEFAULT_RATE = (0.80, 4.00)  # unknown model → cheapest tier (don't overstate)


def _usd_for(model: str, tokens_in: int, tokens_out: int) -> float:
    """Cloud-equivalent USD for one node call from the static rate card."""
    rate_in, rate_out = RATE_CARD.get(model, _DEFAULT_RATE)
    return (tokens_in / 1_000_000) * rate_in + (tokens_out / 1_000_000) * rate_out


class _Measurement:
    """Mutable handle yielded by `CostMeter.meter(...)`. The caller records token
    counts after the real call returns (we cannot know usage before the call)."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.tokens_in = 0
        self.tokens_out = 0
        self.start = time.perf_counter()
        self.ms = 0.0

    def record(self, tokens_in: int, tokens_out: int) -> None:
        """Caller sets real usage (from the OpenAI response) before context exit."""
        self.tokens_in = int(tokens_in)
        self.tokens_out = int(tokens_out)


class CostMeter:
    """Idempotent per-node cost ledger over its own SQLite table.

    Kept in a *separate* table (not graph_store's `nodes`) so cost accounting is
    an independent concern that can be swapped/extended without touching the
    durability core. One row per (run_id, node_name, attempt)."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS node_cost (
                    run_id     TEXT NOT NULL,
                    node_name  TEXT NOT NULL,
                    attempt    INTEGER NOT NULL,
                    model      TEXT NOT NULL,
                    tokens_in  INTEGER NOT NULL,
                    tokens_out INTEGER NOT NULL,
                    ms         REAL NOT NULL,
                    usd        REAL NOT NULL,
                    ts         REAL NOT NULL,
                    UNIQUE(run_id, node_name, attempt)
                )
                """
            )

    def meter(self, run_id: str, node_name: str, attempt: int,
              model: str) -> "_MeterContext":
        """Context manager: times the wrapped call and inserts one cost row on
        exit. Usage: `with cm.meter(...) as m: ...; m.record(t_in, t_out)`."""
        return _MeterContext(self, run_id, node_name, attempt, model)

    def _insert(self, run_id: str, node_name: str, attempt: int, model: str,
                tokens_in: int, tokens_out: int, ms: float) -> None:
        """INSERT OR IGNORE on the idempotency key — re-recording a retried
        (node, attempt) that already landed is a deliberate no-op, never a
        double-bill."""
        usd = _usd_for(model, tokens_in, tokens_out)
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO node_cost(run_id, node_name, attempt, model,"
                " tokens_in, tokens_out, ms, usd, ts) VALUES (?,?,?,?,?,?,?,?,?)",
                (run_id, node_name, attempt, model, tokens_in, tokens_out, ms,
                 usd, time.time()),
            )

    def cost_report(self, run_id: str) -> dict[str, Any]:
        """Aggregate totals for one run: tokens, cloud-equivalent USD, node count."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) n, COALESCE(SUM(tokens_in),0) ti,"
                " COALESCE(SUM(tokens_out),0) to_, COALESCE(SUM(ms),0) ms,"
                " COALESCE(SUM(usd),0) usd FROM node_cost WHERE run_id=?",
                (run_id,),
            ).fetchone()
        return {
            "run_id": run_id,
            "nodes_billed": row["n"],
            "tokens_in": row["ti"],
            "tokens_out": row["to_"],
            "tokens_total": row["ti"] + row["to_"],
            "ms_total": round(row["ms"], 2),
            "usd_cloud_equivalent": round(row["usd"], 6),
        }

    def export_csv(self, run_id: str, path: str) -> None:
        """Dump the per-node ledger for one run to CSV (audit / spreadsheet)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT node_name, attempt, model, tokens_in, tokens_out, ms, usd"
                " FROM node_cost WHERE run_id=? ORDER BY node_name, attempt",
                (run_id,),
            ).fetchall()
        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["node_name", "attempt", "model", "tokens_in",
                             "tokens_out", "ms", "usd_cloud_equivalent"])
            for r in rows:
                writer.writerow([r["node_name"], r["attempt"], r["model"],
                                 r["tokens_in"], r["tokens_out"],
                                 round(r["ms"], 3), round(r["usd"], 6)])


class _MeterContext:
    """The actual `with`-protocol object. Splitting it from `_Measurement` keeps
    the handle the caller mutates (record) separate from the lifecycle (enter/
    exit) — the caller never sees the DB write, only `.record(...)`."""

    def __init__(self, meter: CostMeter, run_id: str, node_name: str,
                 attempt: int, model: str) -> None:
        self._meter = meter
        self._run_id = run_id
        self._node = node_name
        self._attempt = attempt
        self._model = model
        self._m: _Measurement | None = None

    def __enter__(self) -> _Measurement:
        self._m = _Measurement(self._model)
        return self._m

    def __exit__(self, exc_type: type[BaseException] | None,
                 exc: BaseException | None, tb: TracebackType | None) -> None:
        assert self._m is not None
        self._m.ms = (time.perf_counter() - self._m.start) * 1000.0
        # Record even on exception: a node that ran and failed still consumed
        # tokens/latency, and the idempotency key keeps a later retry honest.
        self._meter._insert(self._run_id, self._node, self._attempt, self._model,
                            self._m.tokens_in, self._m.tokens_out, self._m.ms)
