"""worker_pool.py — N async workers draining the READY frontier of one run.

WHY asyncio over threads/processes here:
The work is I/O-bound (an oMLX HTTP call per LLM node, a sleep per tool node).
asyncio gives us cheap concurrency and a clean place to hang a graceful-drain
flag. The blocking bits — SQLite claim and the SYNC openai call — are pushed to a
thread pool via `asyncio.to_thread`, so the event loop never blocks and N workers
genuinely overlap their I/O. Concurrency is the *measured* lever of this lab:
peak_concurrency reported here is the actual observed overlap, not the configured
target.

WHY a graceful SIGTERM drain (not just kill):
A durable runtime should distinguish "crash" (kill -9 → recover_run resets
orphans) from "shutdown" (SIGTERM → finish in-flight nodes, claim no new work).
We install a SIGTERM handler that flips a stop flag: workers complete their
current node and exit cleanly, leaving zero orphaned RUNNING rows. The hard-kill
path is exercised separately by the durability test / bench recovery measurement.
"""
from __future__ import annotations

import asyncio
import signal
import time
from typing import Any, Callable

from graph_store import GraphStore

# How long a worker naps when it finds no READY node but the run isn't finished
# (another worker is mid-node and will promote downstream soon). Small enough to
# stay responsive, large enough not to spin the CPU on empty claims.
_IDLE_POLL_S = 0.02


class _PeakCounter:
    """Track concurrently-executing handlers and the high-water mark. Guarded by
    an asyncio.Lock because += on a shared int across awaits is not atomic."""

    def __init__(self) -> None:
        self.current = 0
        self.peak = 0
        self._lock = asyncio.Lock()

    async def enter(self) -> None:
        async with self._lock:
            self.current += 1
            self.peak = max(self.peak, self.current)

    async def exit(self) -> None:
        async with self._lock:
            self.current -= 1


async def run_graph(
    store: GraphStore,
    run_id: str,
    handler: Callable[[Any], dict[str, Any]],
    concurrency: int,
    cost_meter: Any | None = None,  # noqa: ARG001 — handler owns metering; kept for call-site symmetry
) -> dict[str, Any]:
    """Run `run_id` to completion with `concurrency` async workers.

    Each worker loops: claim a READY node (off-thread, since the claim takes the
    cross-process FileLock and hits SQLite); if None, exit when the run is no
    longer running else nap and retry; else execute the handler (off-thread —
    the LLM/tool handler is synchronous), then mark_done / mark_failed.

    Returns {"wall_clock_s", "peak_concurrency", "nodes_done"}."""
    peak = _PeakCounter()
    done_count = 0
    done_lock = asyncio.Lock()
    stop = asyncio.Event()  # graceful-drain flag flipped by SIGTERM

    # The GraphStore exposes ONE shared FileLock instance (store.lock). Its fd is
    # single-use: a release() nulls the fd, so two sibling threads entering that
    # same instance concurrently corrupt it (BCJ: 'argument must be an int').
    # The claim is a serialization point BY DESIGN (store orders by name, LIMIT
    # 1), so we gate it with an in-process asyncio.Lock. Cross-PROCESS safety
    # still comes from the FileLock; this only stops sibling THREADS in this
    # process from entering the one shared instance at once.
    claim_lock = asyncio.Lock()

    # Install SIGTERM → drain. Best-effort: only the main thread of the main
    # interpreter can set signal handlers, so guard for worker/child contexts.
    loop = asyncio.get_running_loop()
    try:
        loop.add_signal_handler(signal.SIGTERM, stop.set)
        _installed_signal = True
    except (NotImplementedError, ValueError, RuntimeError):
        _installed_signal = False

    async def worker(worker_id: str) -> None:
        nonlocal done_count
        while not stop.is_set():
            async with claim_lock:  # one thread inside the shared FileLock at a time
                node = await asyncio.to_thread(
                    store.claim_ready_node, run_id, worker_id
                )
            if node is None:
                # Nothing claimable. If the run is finished, we're done; else a
                # peer is mid-node and will unblock the frontier — nap + retry.
                if store.run_status(run_id) != "running":
                    return
                await asyncio.sleep(_IDLE_POLL_S)
                continue
            await peak.enter()
            try:
                result = await asyncio.to_thread(handler, node)
                await asyncio.to_thread(store.mark_done, run_id, node.name, result)
                async with done_lock:
                    done_count += 1
            except Exception as exc:  # handler blew up → durable retry/fail path
                await asyncio.to_thread(
                    store.mark_failed, run_id, node.name, repr(exc)
                )
            finally:
                await peak.exit()

    start = time.perf_counter()
    try:
        await asyncio.gather(*(worker(f"w{i}") for i in range(concurrency)))
    finally:
        if _installed_signal:
            loop.remove_signal_handler(signal.SIGTERM)
    wall = time.perf_counter() - start

    return {
        "wall_clock_s": wall,
        "peak_concurrency": peak.peak,
        "nodes_done": done_count,
    }
