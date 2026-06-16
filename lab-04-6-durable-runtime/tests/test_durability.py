"""test_durability.py — the money test: prove a hard kill loses no work.

The whole lab exists to defend one invariant: a process can die mid-run and the
run resumes from the last persisted node with nothing lost and nothing done
twice. We exercise that with REAL process death — a child multiprocessing.Process
claims+runs nodes, gets `.terminate()`d (SIGTERM→ but we escalate to a hard kill
to simulate kill -9), then a FRESH GraphStore in the parent calls recover_run and
finishes the rest. Assertions:
  * every node ends DONE, run status 'done'
  * the event log (replay_run) shows each node 'done' exactly once → no double-work
  * orphaned RUNNING nodes were recovered (a 'recovered' event exists)

All deterministic + LLM-free (tool nodes sleep). A separate unit test proves two
FileLock holders are mutually exclusive.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
import tempfile
import time

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from file_lock import FileLock, LockTimeout  # noqa: E402
from graph_store import DONE, GraphStore  # noqa: E402
from handlers import tool_handler  # noqa: E402


def _chain_dag(n: int = 5, sleep_s: float = 0.3) -> dict:
    """Linear chain of `n` tool nodes — deterministic, slow enough that we can
    reliably terminate the child mid-run."""
    nodes = {f"n{i}": {"type": "tool", "payload": {"sleep_s": sleep_s}}
             for i in range(1, n + 1)}
    edges = [[f"n{i}", f"n{i + 1}"] for i in range(1, n)]
    return {"nodes": nodes, "edges": edges}


def _child_worker(db: str, lock: str, run_id: str) -> None:
    """Run in a child process: claim+execute nodes one at a time, forever. The
    parent kills this mid-run. Uses its OWN GraphStore (separate connection),
    exactly as a separate process would."""
    store = GraphStore(db, lock_path=lock)
    worker_id = f"child-{os.getpid()}"
    while True:
        node = store.claim_ready_node(run_id, worker_id)
        if node is None:
            time.sleep(0.02)
            continue
        result = tool_handler(node)
        store.mark_done(run_id, node.name, result)


def test_hard_kill_recovers_and_completes_without_lost_or_double_work() -> None:
    tmp = tempfile.mkdtemp(prefix="durable_test_")
    db = os.path.join(tmp, "rt.db")
    lock = os.path.join(tmp, "rt.lock")

    store = GraphStore(db, lock_path=lock)
    graph_id = store.create_graph("kill-test", _chain_dag(n=5, sleep_s=0.3))
    run_id = store.start_run(graph_id, "test")

    # Start a child that drains the chain, then hard-kill it mid-run.
    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=_child_worker, args=(db, lock, run_id))
    proc.start()

    # Wait until at least one node is DONE (child is partway), then kill -9.
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        done = [n for n, s in store.node_states(run_id).items() if s == DONE]
        if done:
            break
        time.sleep(0.02)
    assert done, "child made no progress before kill — test setup failed"

    # Hard kill: SIGKILL the child mid-node so a RUNNING row is orphaned. This is
    # the kill -9 scenario the runtime must survive.
    proc.kill()
    proc.join(timeout=5.0)
    assert not proc.is_alive()

    done_before = {n for n, s in store.node_states(run_id).items() if s == DONE}
    assert len(done_before) < 5, "child finished everything — increase node count/sleep"

    # FRESH GraphStore (simulates a restarted process). Recover orphans, finish.
    store2 = GraphStore(db, lock_path=lock)
    recovered = store2.recover_run(run_id)
    # An orphaned RUNNING node should have been reset to READY.
    assert recovered, "no orphaned RUNNING node was recovered after hard kill"

    # Drain the rest single-threaded in-process.
    worker_id = "recovery-worker"
    while True:
        node = store2.claim_ready_node(run_id, worker_id)
        if node is None:
            if store2.run_status(run_id) != "running":
                break
            continue
        store2.mark_done(run_id, node.name, tool_handler(node))

    # ── assertions ──
    states = store2.node_states(run_id)
    assert all(s == DONE for s in states.values()), f"not all DONE: {states}"
    assert store2.run_status(run_id) == "done"

    # No double-work: each node has exactly one 'done' event in the log.
    events = store2.replay_run(run_id)
    done_events = [e for e in events if e["type"] == "done"]
    done_names = [e["node"] for e in done_events]
    assert len(done_names) == 5, f"expected 5 done events, got {done_names}"
    assert len(set(done_names)) == 5, f"a node was done twice (double-work): {done_names}"

    # Recovery left a forensic trail.
    assert any(e["type"] == "recovered" for e in events), "no 'recovered' event logged"


def test_two_filelock_holders_are_mutually_exclusive() -> None:
    """Second acquire must time out while the first holder still owns the lock."""
    tmp = tempfile.mkdtemp(prefix="lock_test_")
    path = os.path.join(tmp, "mx.lock")

    a = FileLock(path)
    a.acquire()
    try:
        b = FileLock(path)
        with pytest.raises(LockTimeout):
            b.acquire(timeout=0.2, poll=0.01)
    finally:
        a.release()

    # Once released, a new holder can acquire immediately.
    c = FileLock(path)
    c.acquire(timeout=1.0)
    c.release()
