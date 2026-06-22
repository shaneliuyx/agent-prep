"""test_cost_ceiling.py — fan-out cost aggregation + ceiling (deer-flow pattern #2).

The lesson: a per-child cap does NOT bound the total (N children at cap = N x cap);
a parent-level ceiling on the running SUM does, by aborting the fan-out on breach.
"""
from __future__ import annotations

import os
import sys

import pytest

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from cost_ceiling import (  # noqa: E402
    BudgetExceeded,
    FanoutBudget,
    run_fanout,
)

PER_CHILD = 100.0
N = 10


def _child() -> tuple[float, object]:
    return (PER_CHILD, "ok")


def test_per_child_cap_does_not_bound_total() -> None:
    # each child is capped at PER_CHILD, but with no effective ceiling the TOTAL
    # is N x PER_CHILD — the exact blow-up the parent ceiling exists to stop
    out = run_fanout([_child] * N, FanoutBudget(ceiling=10_000))
    assert out.spent == PER_CHILD * N == 1000.0
    assert out.ran == N and not out.aborted


def test_ceiling_aborts_the_whole_fanout() -> None:
    # ceiling 350: sums 100,200,300,400 -> child #4 crosses -> abort, 6 skipped
    out = run_fanout([_child] * N, FanoutBudget(ceiling=350))
    assert out.aborted
    assert out.ran == 4
    assert out.spent == 400.0
    assert out.total == N  # 6 children never ran


def test_under_ceiling_runs_everything() -> None:
    out = run_fanout([_child] * N, FanoutBudget(ceiling=2_000))
    assert not out.aborted and out.ran == N and out.spent == 1000.0


def test_budget_charge_raises_on_breach_and_tracks_remaining() -> None:
    b = FanoutBudget(ceiling=250)
    b.charge(100)
    assert b.remaining == 150
    b.charge(100)
    assert b.remaining == 50
    with pytest.raises(BudgetExceeded) as ei:
        b.charge(100)  # 300 > 250
    assert ei.value.spent == 300 and ei.value.ceiling == 250
    assert b.remaining == 0.0  # clamps, never negative
