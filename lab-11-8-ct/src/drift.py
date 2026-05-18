"""Phase 1 — PSI-based data-drift detector.

Population Stability Index:
  $\\text{PSI}(P, Q) = \\sum_i (p_i - q_i) \\ln(p_i / q_i)$

Bins on reference quantiles (skew-aware). Add tiny epsilon to prevent log(0).

Thresholds:
  PSI < 0.10   stable
  0.10–0.25    moderate; investigate
  > 0.25       significant; retrain

Fire retrain only after `should_trigger_retrain` confirms N consecutive
days over threshold (default 3) — single-day spikes are usually traffic
anomalies, not real drift.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def psi(reference: Sequence[float],
        current: Sequence[float],
        n_bins: int = 10) -> float:
    """Compute Population Stability Index between two distributions."""
    ref_arr = np.asarray(reference, dtype=float)
    cur_arr = np.asarray(current, dtype=float)

    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(ref_arr, quantiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref_arr, bins=bin_edges)
    cur_counts, _ = np.histogram(cur_arr, bins=bin_edges)

    eps = 1e-6
    ref_prop = (ref_counts + eps) / (ref_counts.sum() + n_bins * eps)
    cur_prop = (cur_counts + eps) / (cur_counts.sum() + n_bins * eps)

    return float(((cur_prop - ref_prop) * np.log(cur_prop / ref_prop)).sum())


def detect_drift(parquet_path: str,
                 reference_days: int = 7,
                 current_days: int = 1,
                 feature: str = "tokens_in") -> dict:
    """Read span parquet log; compute PSI between reference + current windows."""
    import duckdb
    import pandas as pd

    df = duckdb.sql(f"""
        SELECT timestamp, {feature}
        FROM '{parquet_path}'
        WHERE span_name = 'llm_call' AND {feature} IS NOT NULL
    """).df()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    now = df["timestamp"].max()
    current_arr = np.asarray(
        df[df["timestamp"] >= now - pd.Timedelta(days=current_days)][feature].to_numpy(),
        dtype=float,
    )
    ref_mask = (
        (df["timestamp"] >= now - pd.Timedelta(days=reference_days + current_days)) &
        (df["timestamp"] < now - pd.Timedelta(days=current_days))
    )
    reference_arr = np.asarray(df[ref_mask][feature].to_numpy(), dtype=float)

    if len(current_arr) < 50 or len(reference_arr) < 50:
        return {"feature": feature, "psi": None, "verdict": "insufficient_data",
                "ref_n": len(reference_arr), "cur_n": len(current_arr)}

    p = psi(reference_arr.tolist(), current_arr.tolist())
    verdict = "stable" if p < 0.1 else "moderate" if p < 0.25 else "significant"
    return {"feature": feature, "psi": round(p, 4), "verdict": verdict,
            "ref_n": len(reference_arr), "cur_n": len(current_arr)}


def should_trigger_retrain(history: list[dict],
                          days_consecutive: int = 3) -> bool:
    """Fire retrain only if PSI > 0.25 for N consecutive daily measurements.
    Reduces false positives from one-day spikes."""
    if len(history) < days_consecutive:
        return False
    recent = history[-days_consecutive:]
    return all(h.get("verdict") == "significant" for h in recent)
