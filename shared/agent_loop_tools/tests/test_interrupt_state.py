"""Tests for interrupt_state — mirrors the test cases gnhf's
src/core/interrupt-state.test.ts covers (pure function state machine)."""
from agent_loop_tools import (
    InterruptStateSnapshot,
    get_interrupt_disposition,
    get_interrupt_hint,
)


def test_running_with_no_graceful_request_returns_request_graceful_stop():
    state = InterruptStateSnapshot(status="running", graceful_stop_requested=False)
    assert get_interrupt_disposition(state) == "request-graceful-stop"
    assert get_interrupt_hint(state) == "resume"


def test_graceful_already_requested_returns_force_stop():
    state = InterruptStateSnapshot(status="running", graceful_stop_requested=True)
    assert get_interrupt_disposition(state) == "force-stop"
    assert get_interrupt_hint(state) == "force-stop"


def test_aborted_returns_exit():
    state = InterruptStateSnapshot(status="aborted", graceful_stop_requested=False)
    assert get_interrupt_disposition(state) == "exit"
    assert get_interrupt_hint(state) == "exit"


def test_stopped_returns_force_stop():
    """Loop already stopped; next Ctrl-C is the user trying to bail HARDER —
    treat as force-stop request to surface clear signal."""
    state = InterruptStateSnapshot(status="stopped", graceful_stop_requested=False)
    assert get_interrupt_disposition(state) == "force-stop"


def test_waiting_status_is_interruptible_like_running():
    state = InterruptStateSnapshot(status="waiting", graceful_stop_requested=False)
    assert get_interrupt_disposition(state) == "request-graceful-stop"


def test_snapshot_is_frozen():
    """Snapshot dataclass is frozen — guards against accidental mutation."""
    state = InterruptStateSnapshot(status="running", graceful_stop_requested=False)
    try:
        state.status = "aborted"   # type: ignore[misc]
    except Exception:
        return   # FrozenInstanceError or similar — good
    raise AssertionError("InterruptStateSnapshot should be frozen but was mutated")
