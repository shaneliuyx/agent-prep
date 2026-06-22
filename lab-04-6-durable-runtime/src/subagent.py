"""subagent.py — subagent status contract + polling-timeout safety-net.

A parent that spawns an async/background subagent cannot trust the child's
self-reported status forever: a hung child reports ``running`` indefinitely, and
the task's *own* timeout never fires if the child simply lies about being alive.
The fix (deer-flow `subagents/status_contract.py` + `task_tool.py`) is two
independent timers — the task budget, and the *parent's* poll-loop patience —
plus a **closed, enumerated status contract** parsed identically on both sides.

This is the in-process lab version (W4.6): a fixed status vocabulary, a
cross-boundary ``parse_status`` (the contract), and ``poll_until_terminal``
which adds the polling-timeout SAFETY-NET on top of any "ask the child for its
status" callable. Pure stdlib, LLM-free, deterministic — the same shape as the
rest of this lab's durability primitives. The agentkit shared lib carries the
productionised mirror at ``agentkit/runtime/subagent.py``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

# The closed status contract — both producer and consumer must agree on these.
RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"
CANCELLED = "cancelled"
TIMED_OUT = "timed_out"                   # the task exceeded its OWN declared budget
POLLING_TIMED_OUT = "polling_timed_out"   # parent's safety-net: still RUNNING, assumed stuck
TERMINAL = frozenset({COMPLETED, FAILED, CANCELLED, TIMED_OUT, POLLING_TIMED_OUT})

# Prefix → status. deer-flow stamps status via these prefixes; the consumer falls
# back to the same prefixes. One mapping, both sides — so it can't drift apart.
_PREFIXES = (
    ("task succeeded", COMPLETED),
    ("task failed", FAILED),
    ("task cancelled", CANCELLED),
    ("task polling timed out", POLLING_TIMED_OUT),  # before "timed out" — longer match first
    ("task timed out", TIMED_OUT),
    ("error", FAILED),
)


def parse_status(text: str) -> str:
    """Map a status line to a contract status (the cross-boundary parser).
    Unknown / non-terminal text ⇒ ``running``."""
    low = (text or "").strip().lower()
    for prefix, status in _PREFIXES:
        if low.startswith(prefix):
            return status
    return RUNNING


@dataclass(frozen=True)
class SubagentResult:
    status: str        # one of TERMINAL
    detail: str
    elapsed_s: float


def poll_until_terminal(
    poll: Callable[[], tuple[str, str]],
    *,
    poll_timeout: float,
    interval: float = 0.05,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> SubagentResult:
    """Poll a background subagent until it reaches a terminal status, OR the
    parent's patience (``poll_timeout``) runs out while it is still ``running``.

    ``poll()`` returns ``(status, detail)`` where status is from the contract
    (terminal) or ``running``. The SAFETY-NET: once we have been polling for
    ``poll_timeout`` and the child still claims ``running``, we stop trusting it
    and return ``polling_timed_out`` — the canonical "stuck subagent" rescue.

    ``clock``/``sleep`` are injected so a test proves the safety-net with zero
    real waiting (advance a fake clock); production uses the real ones.
    """
    start = clock()
    while True:
        status, detail = poll()
        elapsed = clock() - start
        if status in TERMINAL:
            return SubagentResult(status, detail, elapsed)
        if elapsed >= poll_timeout:
            return SubagentResult(
                POLLING_TIMED_OUT,
                f"still RUNNING after {poll_timeout}s; assumed stuck",
                elapsed,
            )
        sleep(interval)


def _demo() -> None:
    """Self-check — deterministic (injected clock; no real waiting)."""
    assert parse_status("Task Succeeded. Result: ok") == COMPLETED
    assert parse_status("Task polling timed out after 15 minutes") == POLLING_TIMED_OUT
    assert parse_status("Task timed out. Error: 900 seconds") == TIMED_OUT
    assert parse_status("Investigating ...") == RUNNING

    t = [0.0]

    def clk() -> float:
        return t[0]

    def slp(s: float) -> None:
        t[0] += s

    stuck = poll_until_terminal(lambda: (RUNNING, "wip"), poll_timeout=1.0,
                                interval=0.1, clock=clk, sleep=slp)
    assert stuck.status == POLLING_TIMED_OUT and stuck.elapsed_s >= 1.0, stuck

    t[0] = 0.0
    calls = [0]

    def healthy() -> tuple[str, str]:
        calls[0] += 1
        return (COMPLETED, "done") if calls[0] >= 3 else (RUNNING, "wip")

    ok = poll_until_terminal(healthy, poll_timeout=10.0, interval=0.1,
                             clock=clk, sleep=slp)
    assert ok.status == COMPLETED, ok
    print("subagent._demo OK")


if __name__ == "__main__":
    _demo()
