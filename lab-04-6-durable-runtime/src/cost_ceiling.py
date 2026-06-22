"""cost_ceiling.py — fan-out cost aggregation + parent-level ceiling (deer-flow #2).

Phase 4's `CostMeter` is a per-NODE ledger. It records what each node spent; it
does NOT bound a FAN-OUT. The trap: if you spawn N children and cap each at a
per-child `max_tokens`, the *total* is still N × that cap — a per-child limit is
not a total limit. The bound has to live at the PARENT: aggregate every child's
cost into one running sum and abort the whole fan-out the moment the sum crosses
a ceiling.

deer-flow (`subagents/token_collector.py`): a collector sums subagent token usage
up to the parent run, and the ceiling is checked against the running sum, not per
child. This is the deterministic, LLM-free lab version of that mechanism.

Concurrency note: `run_fanout` here charges sequentially, so the ceiling is exact.
Under a *parallel* pool (Phase 2's `worker_pool`) the check races the in-flight
children, so the real bound is `ceiling + (in-flight count × per-child cost)` —
you can only abort children that have not started. Sequential keeps the lab
deterministic; the overshoot bound is the honest production caveat.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable


class BudgetExceeded(RuntimeError):
    """Raised by `FanoutBudget.charge` when the running sum crosses the ceiling."""

    def __init__(self, spent: float, ceiling: float) -> None:
        super().__init__(f"fan-out cost {spent} exceeded ceiling {ceiling}")
        self.spent = spent
        self.ceiling = ceiling


@dataclass
class FanoutBudget:
    """A parent-level running cost sum with a hard ceiling.

    `charge(cost)` adds to the running `spent` and raises `BudgetExceeded` once
    the *aggregate* crosses `ceiling` — the whole point is that this is checked
    on the SUM, so N cheap children still trip it, which N per-child caps never
    would.
    """

    ceiling: float
    spent: float = 0.0

    def charge(self, cost: float) -> None:
        self.spent += cost
        if self.spent > self.ceiling:
            raise BudgetExceeded(self.spent, self.ceiling)

    @property
    def remaining(self) -> float:
        return max(0.0, self.ceiling - self.spent)


@dataclass(frozen=True)
class FanoutOutcome:
    results: list  # [(cost, result), ...] for children that actually ran
    ran: int
    total: int
    spent: float
    aborted: bool


def run_fanout(
    children: Iterable[Callable[[], tuple[float, object]]],
    budget: FanoutBudget,
) -> FanoutOutcome:
    """Run `children`, charging each one's cost to the shared parent `budget`,
    and ABORT the whole fan-out (skip every remaining child) the instant the
    running sum crosses the ceiling.

    `children` is an iterable of callables ``() -> (cost, result)``. The child
    that trips the ceiling has already run (you can't un-spend it), so it is
    counted in `ran`/`spent`; the children after it never start.
    """
    children = list(children)
    results: list[tuple[float, object]] = []
    for i, child in enumerate(children):
        cost, res = child()
        results.append((cost, res))
        try:
            budget.charge(cost)
        except BudgetExceeded:
            return FanoutOutcome(results, ran=i + 1, total=len(children),
                                 spent=budget.spent, aborted=True)
    return FanoutOutcome(results, ran=len(children), total=len(children),
                         spent=budget.spent, aborted=False)


def _demo() -> None:
    """Self-check — a per-child cap does not bound the total; the ceiling does."""
    PER_CHILD, N = 100.0, 10

    def child() -> tuple[float, object]:
        return (PER_CHILD, "ok")

    # No effective ceiling: total is N x per-child — the blow-up the ceiling stops.
    loose = run_fanout([child] * N, FanoutBudget(ceiling=10_000))
    assert loose.spent == PER_CHILD * N == 1000.0, loose
    assert not loose.aborted and loose.ran == N

    # Ceiling 350: aborts on the child that crosses (sum 400 > 350) -> 4 ran, 6 skipped.
    capped = run_fanout([child] * N, FanoutBudget(ceiling=350))
    assert capped.aborted and capped.ran == 4 and capped.spent == 400.0, capped
    saved = loose.spent - capped.spent

    print(f"cost_ceiling._demo OK — uncapped spent={loose.spent} ran={loose.ran}; "
          f"ceiling=350 spent={capped.spent} ran={capped.ran} "
          f"(saved {saved} by aborting {capped.total - capped.ran} children)")


if __name__ == "__main__":
    _demo()
