"""§8.7.4 — Aggregate ``data/results_w358.jsonl`` into a printable
results table.

Reads the line-per-question JSONL written by ``src/run_longmemeval_slice``
and prints:
    1. Per-backend mean correctness (over the whole slice).
    2. Per-question_type × per-backend means.
    3. Per-question grid: gold vs each backend's predicted + verdict.
    4. Wall-clock medians per backend.
    5. Qdrant SKIP rate (how often summarize_scroll gated a session).

Run from lab root::

    uv run python scripts/aggregate_results.py
"""
from __future__ import annotations

import json
import pathlib
import statistics
from collections import defaultdict

LAB_ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS_PATH = LAB_ROOT / "data" / "results_w358.jsonl"

BACKENDS = ("qdrant", "evercore")


def load() -> list[dict]:
    if not RESULTS_PATH.exists():
        raise SystemExit(f"no results yet at {RESULTS_PATH}")
    return [json.loads(line) for line in RESULTS_PATH.read_text().splitlines() if line.strip()]


def _mean(xs: list[float]) -> float:
    return statistics.mean(xs) if xs else float("nan")


def _median(xs: list[float]) -> float:
    return statistics.median(xs) if xs else float("nan")


def summarise(records: list[dict]) -> None:
    n = len(records)
    print(f"\n=== §8.7.4 LongMemEval slice results ({n} questions) ===\n")

    # 1. Per-backend mean
    print("# 1. Per-backend mean correctness")
    for b in BACKENDS:
        scores = [r[b]["score"] for r in records if b in r]
        oks = sum(1 for s in scores if s == 1.0)
        print(f"  {b:9s}  {oks}/{len(scores)} correct  mean={_mean(scores):.3f}")

    # 2. Per-question_type × per-backend
    print("\n# 2. Per-question_type × per-backend")
    by_type: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        qt = r["question_type"]
        for b in BACKENDS:
            if b in r:
                by_type[qt][b].append(r[b]["score"])
    types = sorted(by_type.keys())
    header = f"  {'question_type':22s}" + " ".join(f"{b:>14s}" for b in BACKENDS) + "    n"
    print(header)
    for qt in types:
        n_qt = max(len(by_type[qt][b]) for b in BACKENDS) if by_type[qt] else 0
        row = f"  {qt:22s}"
        for b in BACKENDS:
            scores = by_type[qt][b]
            oks = sum(1 for s in scores if s == 1.0)
            row += f"  {oks:>2d}/{len(scores):<2d} ({_mean(scores):.2f})"
        row += f"    {n_qt}"
        print(row)

    # 3. Per-question grid
    print("\n# 3. Per-question grid (gold vs predicted, verdicts)")
    print(f"  {'qid':10s} {'type':16s} {'gold':>8s}    Qd   EC   Qd_pred → EC_pred")
    for r in records:
        qd_ok = "✓" if r.get("qdrant", {}).get("score") == 1.0 else "✗"
        ec_ok = "✓" if r.get("evercore", {}).get("score") == 1.0 else "✗"
        qd_pred = (r.get("qdrant", {}).get("predicted", "")[:40] + "…") if len(r.get("qdrant", {}).get("predicted", "")) > 40 else r.get("qdrant", {}).get("predicted", "")
        ec_pred = (r.get("evercore", {}).get("predicted", "")[:40] + "…") if len(r.get("evercore", {}).get("predicted", "")) > 40 else r.get("evercore", {}).get("predicted", "")
        gold = (r["gold"][:8] + "…") if len(r["gold"]) > 8 else r["gold"]
        print(f"  {r['question_id'][:10]:10s} {r['question_type'][:16]:16s} {gold:>8s}     {qd_ok}    {ec_ok}   {qd_pred!r} → {ec_pred!r}")

    # 4. Wall-clock medians
    print("\n# 4. Wall-clock medians (seconds, per backend, OK questions only)")
    print(f"  {'backend':9s}  {'imprint':>10s}  {'retrieve':>10s}  {'read':>10s}")
    for b in BACKENDS:
        ok = [r[b] for r in records if r.get(b, {}).get("status") == "ok"]
        imps = [x["wall_imprint"] for x in ok]
        rets = [x["wall_retrieve"] for x in ok]
        reads = [x["wall_read"] for x in ok]
        print(f"  {b:9s}  {_median(imps):>10.2f}  {_median(rets):>10.2f}  {_median(reads):>10.2f}")

    # 5. Qdrant SKIP rate
    print("\n# 5. Qdrant summarize_scroll SKIP rate")
    total_sess = 0
    skip_sess = 0
    for r in records:
        meta = r.get("qdrant", {}).get("imprint_meta", []) or []
        total_sess += len(meta)
        skip_sess += sum(1 for m in meta if not m.get("imprinted", True))
    if total_sess:
        print(f"  {skip_sess}/{total_sess} sessions SKIPped  ({skip_sess/total_sess:.1%})")
    else:
        print("  (no qdrant imprint_meta available)")


if __name__ == "__main__":
    summarise(load())
