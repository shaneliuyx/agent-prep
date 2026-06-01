# scripts/aggregate_results.py — W3.5.9 multi-backend comparison aggregator
"""Aggregate the driver's combined results JSONL into a backend x axis matrix.

`src/run_longmemeval_slice.py` writes ONE `data/results_w358.jsonl`; each line is

    {"question_id", "question_type", "question", "gold",
     "<backend>": {"score", "correct", "wall_imprint", "wall_retrieve", ...}, ...}

with one nested result block per backend that ran (run_one writes
`record[backend] = result`). This reads that file and prints:
  1. per-question_type x per-backend mean score
  2. per-backend median wall-clock per question (imprint + retrieve)

Run from the lab root::

    uv run python scripts/aggregate_results.py
    uv run python scripts/aggregate_results.py --backends qdrant,mem0,atomic_fact
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
from collections import defaultdict

LAB_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = LAB_ROOT / "data" / "results_w358.jsonl"
ALL_BACKENDS = ["qdrant", "evercore", "mem0", "atomic_fact", "hybrid", "three_tier"]


def load_records(path: pathlib.Path) -> list[dict]:
    """Read the combined results JSONL (one record per question)."""
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def aggregate(records: list[dict], backends: list[str]) -> dict:
    """Align the nested per-backend blocks by question_type into score lists +
    per-backend wall-clock lists."""
    axis_backend: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    walls: dict[str, list[float]] = defaultdict(list)
    for rec in records:
        axis = rec.get("question_type", "unknown")
        for b in backends:
            res = rec.get(b)
            if not isinstance(res, dict):
                continue  # backend didn't run for this question
            axis_backend[axis][b].append(float(res.get("score", 0.0)))
            walls[b].append(float(res.get("wall_imprint", 0.0))
                            + float(res.get("wall_retrieve", 0.0)))
    medians = {b: (statistics.median(w) if w else 0.0) for b, w in walls.items()}
    return {"axis_backend": dict(axis_backend), "wall_medians": medians}


def print_matrix(agg: dict, backends: list[str]) -> None:
    """Render the comparison table as markdown (copy-paste into the chapter)."""
    print("\n## Aggregate score (mean per backend, per axis)\n")
    print("| Axis | " + " | ".join(backends) + " |")
    print("|" + "---|" * (len(backends) + 1))
    for axis, per_b in sorted(agg["axis_backend"].items()):
        row = [axis]
        for b in backends:
            scores = per_b.get(b, [])
            row.append(f"{sum(scores) / len(scores):.2f} (n={len(scores)})" if scores else "-")
        print("| " + " | ".join(row) + " |")
    print("\n## Wall-clock median per question (seconds)\n")
    print("| Backend | Median wall/Q |")
    print("|---|---|")
    for b in backends:
        if b in agg["wall_medians"]:
            print(f"| {b} | {agg['wall_medians'][b]:.2f} s |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", default=",".join(ALL_BACKENDS),
                    help="comma-separated column order / subset")
    ap.add_argument("--results", default=str(DEFAULT_RESULTS),
                    help="path to the combined results JSONL")
    args = ap.parse_args()
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    path = pathlib.Path(args.results)
    if not path.exists():
        raise SystemExit(f"no results at {path} - run src.run_longmemeval_slice first")
    agg = aggregate(load_records(path), backends)
    print_matrix(agg, backends)


if __name__ == "__main__":
    main()
