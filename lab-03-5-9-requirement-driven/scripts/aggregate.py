# scripts/aggregate.py — W3.5.9 per-run results merger + matrix
"""Merge every per-run results file under data/results/ into one canonical
matrix, latest-per-cell wins.

WHY this exists (the data-hygiene fix): the driver used to write a single
`data/results_w358.jsonl` and `unlink` it at the start of every run, so a probe,
smoke, --qid re-run, or single --backend run DESTROYED the prior full run's raw
data (it happened 3× in one session). The driver now writes a distinct
`data/results/run_<tag>.jsonl` per run and never clobbers. This script reconciles
those files into the canonical `data/results/merged.jsonl` + prints the matrix.

Merge semantics (cell = one (question_id, backend) pair):
  * Walk run files in mtime order (oldest → newest).
  * For each question_id, accumulate a merged record; question_type/question/gold
    take the latest file's value; each backend block takes the latest file that
    actually ran that backend. So: full run + later --backend mem0 re-run →
    mem0 cell replaced, every other cell preserved. KU-only --qid re-run →
    only those qids' cells refreshed. No manual backup, no hand-reconstruction.

Run from the lab root::

    uv run python -m scripts.aggregate
    uv run python -m scripts.aggregate --backends qdrant,mem0,atomic_fact,ensemble
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
from collections import defaultdict

LAB_ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS_DIR = LAB_ROOT / "data" / "results"
MERGED_PATH = RESULTS_DIR / "merged.jsonl"
ALL_BACKENDS = ["qdrant", "evercore", "mem0", "atomic_fact",
                "hybrid", "three_tier", "ensemble"]
# Broken-gold question excluded from quality analysis — mirrors QUALITY_EXCLUDE
# in src/run_longmemeval_slice.py (keep in sync). Its memories still merge; it
# is only dropped from the accuracy matrix.
QUALITY_EXCLUDE = {"0a995998"}


def merge_runs(run_files: list[pathlib.Path]) -> dict[str, dict]:
    """Reconcile per-run files into {question_id: merged_record}, latest-per-cell.

    run_files MUST already be sorted oldest→newest so later files overwrite.
    """
    merged: dict[str, dict] = {}
    for path in run_files:
        for ln in path.read_text().splitlines():
            if not ln.strip():
                continue
            rec = json.loads(ln)
            qid = rec.get("question_id")
            if qid is None:
                continue
            slot = merged.setdefault(qid, {})
            for k, v in rec.items():
                # Meta keys: latest wins. Backend blocks: latest that RAN wins —
                # a run that didn't touch backend B carries no B key, so B's
                # earlier value survives untouched.
                slot[k] = v
    return merged


def aggregate(records: list[dict], backends: list[str]) -> dict:
    """Per-(axis × backend) accuracy from the `correct` verdict (judge), plus
    per-backend median wall-clock. correct=None (deferred / judge-failed under
    503 cooldown) is counted separately so a thin cell never inflates accuracy.
    """
    # axis_backend[axis][b] = [list of correct bools]; pending = #correct-None
    axis_backend: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    pending: dict[str, int] = defaultdict(int)
    walls: dict[str, list[float]] = defaultdict(list)
    for rec in records:
        if rec.get("question_id") in QUALITY_EXCLUDE:
            continue
        axis = rec.get("question_type", "unknown")
        for b in backends:
            res = rec.get(b)
            if not isinstance(res, dict):
                continue  # backend didn't run for this question
            correct = res.get("correct")
            if correct is None:
                pending[b] += 1
            else:
                axis_backend[axis][b].append(bool(correct))
            walls[b].append(float(res.get("wall_imprint", 0.0))
                            + float(res.get("wall_retrieve", 0.0)))
    medians = {b: (statistics.median(w) if w else 0.0) for b, w in walls.items()}
    return {"axis_backend": dict(axis_backend), "pending": dict(pending),
            "wall_medians": medians}


def _acc(bools: list[bool]) -> str:
    return f"{100 * sum(bools) / len(bools):.0f}% (n={len(bools)})" if bools else "-"


def print_matrix(agg: dict, backends: list[str]) -> None:
    """Render the comparison table as markdown (paste into RESULTS.md / chapter)."""
    axes = sorted(agg["axis_backend"])
    print("\n## Accuracy (judged correct, % per backend)\n")
    print("| Axis | " + " | ".join(backends) + " |")
    print("|" + "---|" * (len(backends) + 1))
    overall: dict[str, list[bool]] = defaultdict(list)
    for axis in axes:
        per_b = agg["axis_backend"][axis]
        row = [axis]
        for b in backends:
            bools = per_b.get(b, [])
            overall[b].extend(bools)
            row.append(_acc(bools))
        print("| " + " | ".join(row) + " |")
    print("| **overall** | "
          + " | ".join(f"**{_acc(overall.get(b, []))}**" for b in backends) + " |")
    if any(agg["pending"].values()):
        print("\n_pending (correct=None, needs rejudge): "
              + ", ".join(f"{b}={n}" for b, n in sorted(agg["pending"].items()) if n)
              + "_")
    print("\n## Wall-clock median per question (imprint + retrieve, seconds)\n")
    print("| Backend | Median wall/Q |")
    print("|---|---|")
    for b in backends:
        if b in agg["wall_medians"]:
            print(f"| {b} | {agg['wall_medians'][b]:.2f} s |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", default=",".join(ALL_BACKENDS),
                    help="comma-separated column order / subset")
    ap.add_argument("--results-dir", default=str(RESULTS_DIR),
                    help="dir of per-run run_*.jsonl files")
    ap.add_argument("--no-write", action="store_true",
                    help="print the matrix but do not (re)write merged.jsonl")
    args = ap.parse_args()
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]

    rdir = pathlib.Path(args.results_dir)
    run_files = sorted(rdir.glob("run_*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not run_files:
        raise SystemExit(
            f"no run_*.jsonl in {rdir} - run src.run_longmemeval_slice first")
    print(f">>> merging {len(run_files)} run file(s): "
          + ", ".join(p.name for p in run_files))

    merged = merge_runs(run_files)
    records = list(merged.values())
    if not args.no_write:
        rdir.mkdir(parents=True, exist_ok=True)
        with MERGED_PATH.open("w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        print(f">>> merged {len(records)} questions → {MERGED_PATH}")

    print_matrix(aggregate(records, backends), backends)


if __name__ == "__main__":
    main()
