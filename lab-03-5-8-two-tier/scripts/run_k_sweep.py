"""K_min volume-floor inflection sweep — W3.5.8 §5.3.4 follow-up.

Chapter §5.3.4 measured two volume regimes:
  K=1 triple  → catastrophic (Qwen3.6-27B 30→25%, Opus 70→45%, both −30 to −35pts)
  K=14-57    → safe (+5pt lift uniformly)

The phase transition between regimes is unmeasured. `K_min=8` was a
hand-picked conservative constant in the gap. This sweep samples K values
in (1, 14] to pin the actual inflection.

Methodology:
  - 1 compose model: Qwen3.5-27B-Claude-Opus-distill (the strongest local;
    matches §5.3.5 N=100 board's 77% baseline)
  - K ∈ {2, 4, 6, 8, 10, 12} — 6 sweep points spanning the gap
  - N=10 questions per K (first 10 of longmemeval_oracle slice — same as
    §5.3.2 contaminated-then-clean baseline)
  - K_TARGET env var → inserts "emit EXACTLY K triples total" constraint
    into ATOMISE_SYSTEM (per scripts/run_longmemeval_oracle.py:_atomise_system_for_run)

Output per K: results/k_sweep_k{K}.json (same shape as longmemeval_oracle JSON)
Aggregate: results/k_sweep_summary.json (K → accuracy + triples_emitted stats)

Wall-clock budget: ~2.5 hr for 6 K × N=10 (Qwen-distill ~150s per Q with
atomise enabled; subprocess-isolated per K to prevent ATOMISE_SYSTEM bleed).

Run:
    uv run python scripts/run_k_sweep.py --ks 2,4,6,8,10,12 --limit 10
    uv run python scripts/run_k_sweep.py --ks 8 --limit 3    # smoke test
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = LAB_ROOT / "results"
RUNNER = LAB_ROOT / "scripts" / "run_longmemeval_oracle.py"


def _run_one_k(k: int, limit: int, campaign_root: str) -> dict:
    """Spawn run_longmemeval_oracle.py as a subprocess with K_TARGET set.
    Subprocess isolation prevents per-K ATOMISE_SYSTEM bleed across runs."""
    campaign = f"{campaign_root}-k{k}"
    out_path = RESULTS_DIR / f"k_sweep_k{k}.json"
    env = os.environ.copy()
    env["K_TARGET"] = str(k)
    env["ATOMISE_AT_READ"] = "1"  # sweep only makes sense with read-time atomise

    print(f"\n[K={k}] launching subprocess (campaign={campaign}, limit={limit})...")
    t0 = time.time()
    proc = subprocess.run(
        [
            "uv", "run", "python", str(RUNNER),
            "--limit", str(limit),
            "--campaign", campaign,
            "--out", str(out_path),
        ],
        cwd=LAB_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    wall = time.time() - t0
    if proc.returncode != 0:
        print(f"[K={k}] FAILED rc={proc.returncode} in {wall:.0f}s")
        print(f"  stderr tail: {proc.stderr[-400:]}")
        return {"k": k, "error": True, "wall_s": wall, "stderr_tail": proc.stderr[-400:]}

    data = json.loads(out_path.read_text())
    correct = data.get("correct", 0)
    total = data.get("total_questions", 0)
    accuracy = data.get("accuracy", 0.0)
    triples_emitted = [
        q.get("triples_emitted", 0) for q in data.get("per_question", [])
        if q.get("triples_emitted") is not None
    ]
    mean_triples = (sum(triples_emitted) / len(triples_emitted)) if triples_emitted else 0
    print(f"[K={k}] {correct}/{total} ({accuracy:.1%}); mean_triples_emitted={mean_triples:.1f}; wall {wall:.0f}s")
    return {
        "k": k,
        "campaign": campaign,
        "n_questions": total,
        "correct": correct,
        "accuracy": accuracy,
        "mean_triples_emitted": mean_triples,
        "triples_range": [min(triples_emitted), max(triples_emitted)] if triples_emitted else None,
        "wall_s": wall,
        "out_path": str(out_path.relative_to(LAB_ROOT)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ks", required=True,
                   help="comma-separated K values, e.g. '2,4,6,8,10,12'")
    p.add_argument("--limit", type=int, default=10,
                   help="questions per K (default 10; §5.3.2 used 20)")
    p.add_argument("--out", default="results/k_sweep_summary.json",
                   help="aggregated board output")
    args = p.parse_args()

    ks = [int(k) for k in args.ks.split(",")]
    campaign_root = f"k-sweep-{int(time.time())}"

    print(f"=== K_min inflection sweep ===")
    print(f"K values: {ks}")
    print(f"Questions per K: {args.limit}")
    print(f"Compose model: from MODEL_COMPOSE env (default per runner)")
    print(f"Estimated wall: ~{len(ks) * args.limit * 150 / 60:.0f} min")

    sweep_results = []
    t0_total = time.time()
    for k in ks:
        r = _run_one_k(k, args.limit, campaign_root)
        sweep_results.append(r)

    summary = {
        "campaign_root": campaign_root,
        "ks": ks,
        "limit_per_k": args.limit,
        "total_wall_min": round((time.time() - t0_total) / 60, 1),
        "per_k": sweep_results,
        "board": [
            {
                "k": r["k"],
                "accuracy": r.get("accuracy"),
                "mean_triples": r.get("mean_triples_emitted"),
                "wall_s": r.get("wall_s"),
            }
            for r in sweep_results if not r.get("error")
        ],
    }
    out_path = LAB_ROOT / args.out
    out_path.write_text(json.dumps(summary, indent=2))

    print(f"\n=== sweep complete ({summary['total_wall_min']} min total) ===")
    print(f"K     acc    mean_triples   wall_s")
    print(f"---   ----   ------------   ------")
    for b in summary["board"]:
        print(f"{b['k']:>3}   {b['accuracy']:>4.1%}    {b['mean_triples']:>6.1f}      {b['wall_s']:>6.0f}")
    print(f"\nSummary → {out_path.relative_to(LAB_ROOT)}")


if __name__ == "__main__":
    main()
