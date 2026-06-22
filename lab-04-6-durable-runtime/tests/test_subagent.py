"""test_subagent.py — status contract + polling safety-net (deer-flow pattern #1).

Deterministic (injected clock; no real waiting), plus one real-clock measurement
that the safety-net fires in ~poll_timeout of wall time.
"""
from __future__ import annotations

import os
import sys
import time

import pytest

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from subagent import (  # noqa: E402
    COMPLETED,
    FAILED,
    POLLING_TIMED_OUT,
    RUNNING,
    TIMED_OUT,
    parse_status,
    poll_until_terminal,
)


@pytest.mark.parametrize("text,expected", [
    ("Task Succeeded. Result: ok", COMPLETED),
    ("Task failed. Error: boom", FAILED),
    ("Task polling timed out after 15 minutes", POLLING_TIMED_OUT),
    ("Task timed out. Error: 900 seconds", TIMED_OUT),
    ("Error: Unknown subagent type 'foo'", FAILED),
    ("Investigating ...", RUNNING),
])
def test_parse_status_contract(text: str, expected: str) -> None:
    assert parse_status(text) == expected


def _fake_clock():
    t = [0.0]
    return (lambda: t[0]), (lambda s: t.__setitem__(0, t[0] + s))


def test_stuck_subagent_hits_polling_safety_net() -> None:
    # child lies about being alive forever → parent's poll_timeout must rescue it
    clk, slp = _fake_clock()
    r = poll_until_terminal(lambda: (RUNNING, "wip"), poll_timeout=1.0,
                            interval=0.1, clock=clk, sleep=slp)
    assert r.status == POLLING_TIMED_OUT
    assert r.elapsed_s >= 1.0


def test_healthy_subagent_completes_without_safety_net() -> None:
    clk, slp = _fake_clock()
    calls = [0]

    def poll():
        calls[0] += 1
        return (COMPLETED, "done") if calls[0] >= 3 else (RUNNING, "wip")

    r = poll_until_terminal(poll, poll_timeout=10.0, interval=0.1, clock=clk, sleep=slp)
    assert r.status == COMPLETED and r.detail == "done"


def test_terminal_status_returns_immediately() -> None:
    clk, slp = _fake_clock()
    r = poll_until_terminal(lambda: (FAILED, "RuntimeError"), poll_timeout=5.0,
                            clock=clk, sleep=slp)
    assert r.status == FAILED


def test_polling_timeout_is_real_wall_clock() -> None:
    # the safety-net fires in ~poll_timeout of REAL time (small, so the test is fast)
    start = time.monotonic()
    r = poll_until_terminal(lambda: (RUNNING, "wip"), poll_timeout=0.2, interval=0.02)
    assert r.status == POLLING_TIMED_OUT
    assert 0.15 <= (time.monotonic() - start) <= 1.0
