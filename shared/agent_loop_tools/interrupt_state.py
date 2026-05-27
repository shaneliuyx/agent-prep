"""Graceful-stop state machine for long-running agent loops.

Source: ported from kunchenguid/gnhf src/core/interrupt-state.ts (33 LOC,
MIT-licensed). gnhf uses this to handle Ctrl-C semantics across overnight
agent runs — first Ctrl-C requests graceful stop (let current iteration
finish, then exit cleanly); second Ctrl-C force-stops (abort current
iteration); third Ctrl-C exits the process.

Original TS reference:
  https://github.com/kunchenguid/gnhf/blob/main/src/core/interrupt-state.ts

This Python port is a near-line-for-line translation. Pure functions, no
side effects, fully unit-testable. Use this in any lab that has a
multi-iteration loop you want to interrupt cleanly without dropping work
that's mid-flight.

Status enum:
  running   — loop is executing an iteration
  waiting   — loop is between iterations (e.g. exponential backoff sleep)
  aborted   — loop was force-stopped; cleanup pending
  stopped   — loop terminated cleanly after graceful-stop request

Disposition (what to do on the NEXT interrupt signal):
  request-graceful-stop — first Ctrl-C; ask loop to finish current
                          iteration then exit. State unchanged.
  force-stop            — second Ctrl-C while graceful already requested;
                          abort current iteration immediately.
  exit                  — already aborted; exit the process now.

Hint (what to TELL THE USER about the next interrupt):
  resume      — graceful stop hasn't been requested; their Ctrl-C will
                request it
  force-stop  — they've already requested graceful; their next Ctrl-C
                will force-stop
  exit        — they've already force-stopped; their next Ctrl-C exits
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


InterruptHint = Literal["resume", "force-stop", "exit"]
InterruptDisposition = Literal[
    "request-graceful-stop",
    "force-stop",
    "exit",
]


@dataclass(frozen=True)
class InterruptStateSnapshot:
    """Snapshot of orchestrator state needed to compute interrupt behavior.
    Mirror of gnhf's `InterruptStateSnapshot` interface."""

    status: Literal["running", "waiting", "aborted", "stopped"]
    graceful_stop_requested: bool


def get_interrupt_disposition(state: InterruptStateSnapshot) -> InterruptDisposition:
    """What should the NEXT interrupt signal do?

    Pure function; no side effects. Mirror of gnhf's `getInterruptDisposition`.
    """
    if state.status == "aborted":
        return "exit"
    if state.graceful_stop_requested or state.status == "stopped":
        return "force-stop"
    return "request-graceful-stop"


def get_interrupt_hint(state: InterruptStateSnapshot) -> InterruptHint:
    """What should we TELL THE USER about the next interrupt?

    User-facing hint that maps from disposition to a verb. Mirror of gnhf's
    `getInterruptHint`.
    """
    disposition = get_interrupt_disposition(state)
    if disposition == "exit":
        return "exit"
    if disposition == "force-stop":
        return "force-stop"
    return "resume"
