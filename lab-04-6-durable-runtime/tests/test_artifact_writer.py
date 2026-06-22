"""test_artifact_writer.py — concurrent artifact writes (deer-flow pattern #5).

The money test: N threads each do M read-modify-write increments on one shared
file. UNLOCKED, the interleaved RMW cycles lose updates (final < N*M). LOCKED via
`locked_update`, every increment lands (final == N*M). A third test proves the
`os.replace` publish is atomic (a reader never sees a torn file).
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from artifact_writer import atomic_write, locked_update  # noqa: E402

THREADS = 8
PER = 50
EXPECTED = THREADS * PER
GAP = 0.0005  # widen the read->write window so the unlocked race is real, not luck


def _run(path: str, lock_path: str, *, use_lock: bool) -> int:
    atomic_write(path, "0")

    def bump_unlocked() -> None:
        for _ in range(PER):
            old = int(Path(path).read_text())
            time.sleep(GAP)
            atomic_write(path, str(old + 1))

    def bump_locked() -> None:
        for _ in range(PER):
            locked_update(path, lambda o: str(int(o or "0") + 1), lock_path=lock_path)

    target = bump_locked if use_lock else bump_unlocked
    ts = [threading.Thread(target=target) for _ in range(THREADS)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return int(Path(path).read_text())


def test_locked_update_loses_no_increments(tmp_path) -> None:
    path = str(tmp_path / "counter.txt")
    lock = str(tmp_path / "counter.lock")
    assert _run(path, lock, use_lock=True) == EXPECTED


def test_unlocked_rmw_loses_updates(tmp_path) -> None:
    # demonstrates the bug the lock fixes: interleaved RMW drops increments
    path = str(tmp_path / "counter.txt")
    lock = str(tmp_path / "counter.lock")
    assert _run(path, lock, use_lock=False) < EXPECTED


def test_atomic_write_publish_is_never_torn(tmp_path) -> None:
    # a reader racing a writer sees only whole old/new values, never a partial
    path = str(tmp_path / "doc.txt")
    small, big = "x", "y" * 100_000
    atomic_write(path, small)
    seen: list[int] = []
    stop = threading.Event()

    def reader() -> None:
        while not stop.is_set():
            seen.append(len(Path(path).read_text()))

    r = threading.Thread(target=reader)
    r.start()
    for _ in range(200):
        atomic_write(path, big)
        atomic_write(path, small)
    stop.set()
    r.join()
    # every observed length is a COMPLETE value — never an in-between torn size
    assert set(seen) <= {len(small), len(big)}, f"torn read: {sorted(set(seen))}"
