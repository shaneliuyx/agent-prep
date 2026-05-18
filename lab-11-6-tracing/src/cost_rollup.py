"""Phase 3 — DuckDB-backed cost + latency rollups over span parquet log.

Two queries that matter for senior interviews:
  1. cost_per_role_per_day  -> catches budget drift
  2. p99_latency_per_model  -> catches SLA risk

Append-only parquet log is the data substrate; DuckDB executes ad-hoc SQL.
Pairs with Langfuse self-hosted for the trace-tree UI.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


DEFAULT_SPANS_PATH = str(Path(__file__).resolve().parent.parent / "data" / "spans.parquet")


def export_span_to_parquet(span_data: dict[str, Any],
                          path: str = DEFAULT_SPANS_PATH) -> None:
    """Append one span row. Real production: use pyarrow ParquetWriter for
    proper append semantics (or partition by hour/day for query efficiency)."""
    df = pd.DataFrame([span_data])
    if "timestamp" in span_data:
        df["timestamp"] = pd.to_datetime(span_data["timestamp"])

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        existing = pd.read_parquet(out_path)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_parquet(out_path, engine="pyarrow", index=False)


def cost_per_role_per_day(path: str = DEFAULT_SPANS_PATH) -> pd.DataFrame:
    """Daily cost breakdown by role + model.
    Use to catch: budget drift, unexpected role traffic, model-tier overrun."""
    return duckdb.sql(f"""
        SELECT
            DATE_TRUNC('day', timestamp) AS day,
            role,
            model_name,
            COUNT(*) AS calls,
            SUM(tokens_in) AS tot_in,
            SUM(tokens_out) AS tot_out,
            SUM(cost_usd) AS tot_cost,
            ROUND(SUM(cost_usd) / NULLIF(COUNT(*), 0), 4) AS cost_per_call
        FROM '{path}'
        WHERE span_name = 'llm_call'
        GROUP BY day, role, model_name
        ORDER BY day DESC, tot_cost DESC
    """).df()


def p99_latency_per_model(path: str = DEFAULT_SPANS_PATH,
                         hours: int = 24) -> pd.DataFrame:
    """Latency distribution per model over the last N hours.
    Use to catch: SLA risk (p99 vs target), tail-degradation episodes."""
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    return duckdb.sql(f"""
        SELECT
            model_name,
            COUNT(*) AS calls,
            ROUND(QUANTILE_CONT(duration_ms, 0.50), 0) AS p50_ms,
            ROUND(QUANTILE_CONT(duration_ms, 0.95), 0) AS p95_ms,
            ROUND(QUANTILE_CONT(duration_ms, 0.99), 0) AS p99_ms,
            ROUND(MAX(duration_ms), 0) AS max_ms
        FROM '{path}'
        WHERE span_name = 'llm_call'
          AND timestamp > '{cutoff}'
        GROUP BY model_name
        ORDER BY p99_ms DESC
    """).df()
