"""Phase 2 — eval-gated deployment comparator.

Gate rule:
  deploy iff  $\\text{eval}_{\\text{cand}} \\geq \\text{eval}_{\\text{prod}} - \\epsilon$

Exit code 0 = PASS (PR can merge).
Exit code 1 = FAIL (regression beyond tolerance).
Exit code 2 = MISSING (metric not in one of the files).
"""
from __future__ import annotations

import argparse
import json
import sys


def compare(candidate: dict,
            baseline: dict,
            primary_metric: str,
            tolerance: float) -> tuple[int, str]:
    """Return (exit_code, message)."""
    c = candidate.get(primary_metric)
    b = baseline.get(primary_metric)
    if c is None or b is None:
        return 2, f"missing metric {primary_metric}"

    delta = c - b
    msg = f"{primary_metric}: candidate={c:.4f} baseline={b:.4f} delta={delta:+.4f}"

    if delta < -tolerance:
        return 1, f"{msg}\nFAIL: degraded by {-delta:.4f} > tolerance {tolerance}"
    return 0, f"{msg}\nPASS"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--tolerance", type=float, default=0.02)
    ap.add_argument("--primary-metric", default="ragas_faithfulness")
    args = ap.parse_args()

    cand = json.loads(open(args.candidate).read())
    base = json.loads(open(args.baseline).read())

    code, msg = compare(cand, base, args.primary_metric, args.tolerance)
    print(msg)
    sys.exit(code)


if __name__ == "__main__":
    main()
