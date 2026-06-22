"""artifact_writer.py — corruption-safe artifact writes (deer-flow pattern #5).

Phases 1–5 use `FileLock` to claim *nodes* atomically. The other use of a file
lock in an agent runtime is protecting the **artifacts** subagents write: when
two writers touch the same output file concurrently they race two distinct ways.

  * **torn read** — a reader observes a half-written file (writer is mid-`write`).
  * **lost update** — two read-modify-write cycles interleave; the second writer
    overwrites the first's change with stale data it read before the write.

The fixes are *different primitives*, and you usually need both:

  * `os.replace` (atomic rename on POSIX) gives **atomic publication** — a reader
    sees the whole old file or the whole new one, never a partial. Fixes torn
    reads with NO lock.
  * a cross-process `FileLock` around the *whole* read-modify-write fixes **lost
    updates** — only one writer is inside the RMW at a time.

A lock alone does not help a reader that opens the file mid-write; an atomic
rename alone does not stop two RMW cycles from clobbering each other. deer-flow's
sandbox file ops need both. Source: bytedance/deer-flow sandbox file operations.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Callable

from file_lock import FileLock


def atomic_write(path: str, data: str) -> None:
    """Publish `data` to `path` atomically: write a temp file, then rename.

    `os.replace` is atomic, so a concurrent reader of `path` sees the old
    contents or the new contents — never a half-written file. The temp file is
    created with `tempfile.mkstemp` in the *target's directory* (same filesystem,
    so the rename is atomic) with a UNIQUE name — pid-scoping is not enough, two
    threads of one process would collide on the same temp and one rename would
    race the other away.
    """
    p = Path(path)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=p.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(data)
        os.replace(tmp, p)  # atomic publish
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def locked_update(
    path: str,
    fn: Callable[[str | None], str],
    *,
    lock_path: str | None = None,
    timeout: float = 10.0,
) -> str:
    """Read-modify-write `path` under a cross-process lock (no lost updates).

    `fn(old)` receives the current contents (`None` if the file is absent) and
    returns the new contents. The read, the compute, and the atomic write all
    happen while holding `FileLock(lock_path)` — so two concurrent callers
    serialize instead of clobbering each other. Returns the written contents.
    """
    lock_path = lock_path or f"{path}.lock"
    lock = FileLock(lock_path)
    lock.acquire(timeout=timeout)
    try:
        old = Path(path).read_text() if Path(path).exists() else None
        new = fn(old)
        atomic_write(path, new)
        return new
    finally:
        lock.release()


def _demo() -> None:
    """Self-check — concurrent RMW: unlocked loses updates, locked does not."""
    import tempfile
    import threading
    import time

    def run(use_lock: bool, *, threads: int = 8, per: int = 50,
            gap: float = 0.0005) -> int:
        d = tempfile.mkdtemp()
        path = os.path.join(d, "counter.txt")
        atomic_write(path, "0")
        lock_path = os.path.join(d, "counter.lock")

        def bump_unlocked() -> None:
            for _ in range(per):
                old = int(Path(path).read_text())
                time.sleep(gap)  # widen the read->write window so the race is real
                atomic_write(path, str(old + 1))

        def bump_locked() -> None:
            for _ in range(per):
                locked_update(path, lambda o: str(int(o or "0") + 1),
                              lock_path=lock_path)

        target = bump_locked if use_lock else bump_unlocked
        ts = [threading.Thread(target=target) for _ in range(threads)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        return int(Path(path).read_text())

    expected = 8 * 50
    locked = run(True)
    unlocked = run(False)
    assert locked == expected, f"locked lost updates: {locked} != {expected}"
    assert unlocked < expected, f"unlocked should lose updates: {unlocked}"
    print(f"artifact_writer._demo OK — locked={locked}/{expected}, "
          f"unlocked={unlocked}/{expected} (lost {expected - unlocked})")


if __name__ == "__main__":
    _demo()
