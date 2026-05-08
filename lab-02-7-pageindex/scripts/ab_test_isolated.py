"""Run v1 + v2 each in its own subprocess, then merge + compare.

Avoids the same-process oMLX KV-cache pollution that hit the prior in-process
A/B harness (W2.7 Bad-Case Entry 6 — pre-test cache state from one variant
poisons the next variant's tool-routing).
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

_LAB_ROOT = Path(__file__).resolve().parents[1]
_VENV_PY = "/Users/yuxinliu/.openharness-venv/bin/python"


def run_variant(variant: str) -> dict:
    print(f"\n=== Spawning {variant} subprocess ===")
    t0 = time.time()
    cp = subprocess.run(
        [_VENV_PY, str(_LAB_ROOT / "scripts" / "run_one_variant.py"), variant],
        capture_output=True, text=True, cwd=str(_LAB_ROOT),
    )
    elapsed = time.time() - t0
    print(f"=== {variant} subprocess exit={cp.returncode} elapsed={elapsed:.1f}s ===")
    print(cp.stdout)
    if cp.returncode != 0:
        print(f"STDERR:\n{cp.stderr}")
        if cp.returncode == 2:
            print(f"oMLX cache poisoned during {variant}. Pause + retry.")
            return {"variant": variant, "error": "cache_poisoned"}
        return {"variant": variant, "error": f"exit_{cp.returncode}"}
    return json.loads((_LAB_ROOT / "results" / f"ab_{variant}.json").read_text())


def main():
    print(f"Lab root: {_LAB_ROOT}")
    print(f"Eval: data/eval.json (8q) + data/eval_v2.json (8 NEW q) = 16q total\n")

    a1 = run_variant("v1")
    if "error" in a1:
        print(f"ABORTED on v1: {a1['error']}")
        sys.exit(1)

    print("\n--- pause 5s between subprocesses (let oMLX cache settle) ---")
    time.sleep(5)

    a2 = run_variant("v2")
    if "error" in a2:
        print(f"ABORTED on v2: {a2['error']}")
        sys.exit(1)

    # Merge + compare
    print("\n=== A/B Summary (16 questions: 8 original + 8 NEW) ===")
    print(f"  Aggregate (judge):  v1={a1['agg_judge']:.3f}  v2={a2['agg_judge']:.3f}  Δ={a2['agg_judge']-a1['agg_judge']:+.3f}")
    print(f"  Aggregate (substr): v1={a1['agg_sub']:.3f}  v2={a2['agg_sub']:.3f}  Δ={a2['agg_sub']-a1['agg_sub']:+.3f}")
    print(f"  Aggregate (lat):    v1={a1['agg_lat']:.1f}s  v2={a2['agg_lat']:.1f}s  Δ={a2['agg_lat']-a1['agg_lat']:+.1f}s")
    print("\n  Per-category (judge):")
    cats = sorted(set(a1.get("per_cat", {})) | set(a2.get("per_cat", {})))
    for cat in cats:
        v1c = a1["per_cat"].get(cat, 0)
        v2c = a2["per_cat"].get(cat, 0)
        print(f"    {cat:30s}  v1={v1c:.2f}  v2={v2c:.2f}  Δ={v2c-v1c:+.2f}")

    out = {"v1": a1, "v2": a2,
           "delta_judge": a2["agg_judge"] - a1["agg_judge"],
           "delta_sub":   a2["agg_sub"] - a1["agg_sub"],
           "delta_lat":   a2["agg_lat"] - a1["agg_lat"]}
    (_LAB_ROOT / "results" / "ab_v1_v2_isolated.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {_LAB_ROOT / 'results' / 'ab_v1_v2_isolated.json'}")


if __name__ == "__main__":
    main()
