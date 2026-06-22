# RESULTS — Durable Runtime: Four-Topology Throughput

## Methodology
- **Constant model:** `Qwen2.5-Coder-7B-Instruct-MLX-4bit` (held constant across all four topologies so
  dependency *structure* is the only independent variable — the W4.5 "hold mode
  constant" discipline).
- **Constant node count:** 5 per topology (only the edges differ).
- **Repeats:** 5 per topology; wall-clock is the mean. Token counts are
  summed REAL usage from oMLX `response.usage` (deterministic at temperature 0).
- **Recovery measurement:** recovery is a property of the `GraphStore`+scheduler
  layer, *shared identically by all four topologies* (a topology is only an edge
  set fed to the same store), so it is measured ONCE — not per row. A deterministic
  tool-node chain is drained by a child process, hard-killed with SIGKILL mid-run,
  then a fresh `GraphStore` calls `recover_run` and finishes the remaining nodes.
  Reported time is the recovery phase only (fresh-store → run done).
- **Hardware:** Apple M5 Pro, 48GB. **Date:** 2026-06-17.

## Results
| topology | mean wall-clock (s) | total tokens | peak concurrency |
|---|---|---|---|
| sequential | 0.994 | 195 | 1 |
| parallel | 0.903 | 195 | 4 |
| hierarchical | 0.921 | 195 | 3 |
| workflow | 0.928 | 195 | 1 |

**Recovery** (topology-agnostic — single probe on a dedicated linear tool-node chain, not tied to any row above): **1.227s** from fresh-store `recover_run` to run done.

## Interpretation (honest)
- **sequential** is the throughput floor: concurrency=1, every node waits on its
  predecessor, so wall-clock ≈ the sum of per-node latencies.
- **parallel / hierarchical** expose real overlap (peak concurrency 4 / 3); their
  wall-clock is governed by the critical path plus per-call latency, not the sum,
  so they finish faster than sequential despite identical node count and model.
- **workflow** is sequential-by-construction (gen→validate→finalize) and tracks
  the sequential floor.
- Token totals are within noise of each other across topologies — node count and
  prompt are constant, confirming we isolated *structure*, not work volume.
- **Recovery** is available to *every* topology identically — it lives in
  `recover_run` (flip orphaned RUNNING→READY) + `_promote_downstream` (resume by
  `deps_json`), one layer below topology shape. It completed the remaining nodes
  after a real SIGKILL with zero lost or double-done work (asserted in
  tests/test_durability.py via the event log). The measured time is dominated by
  the residual tool-node sleeps — the cost of *finishing*, not of recovering
  (recover_run itself is a single table update). Note: recovery *time* (unlike
  capability) would track the same critical-path logic as wall-clock above — a
  parallel remainder finishes faster than a sequential one — but we probe it once
  on a linear chain rather than per-topology, since the durability guarantee is
  what's under test, not its shape-dependent finish cost.

## End-to-end demo — `examples/example_graph.py` (live oMLX, 2026-06-17)

One full durable run through the public path: `topologies.parallel(llm=True)` → `create_graph` → `Scheduler.trigger_manually` → `run_graph` (4 async workers) → `cost_report`. Model `Qwen2.5-Coder-7B-Instruct-MLX-4bit` on oMLX `:8000`.

| field | value |
|---|---|
| graph / run | `g_b651c363be17` (5 nodes) / `r_bc350dfff2be` (trigger=manual) |
| final node states | n1..n5 all `done`; run status `done` |
| wall-clock | 1.366 s |
| peak concurrency | 4 (the parallel fan-out) |
| nodes done | 5 |
| tokens | in 185 + out 10 = **195 total** (5 nodes billed) |
| summed handler latency | `ms_total` 2784.79 ms |
| cloud-equivalent cost | $0.000188 |

Reading it: `peak_concurrency: 4` matches the fan-out's maximum parallelism, and `tokens_total: 195` matches the four-topology bench (same model + 5 nodes + prompt) — the demo and the bench agree. The single-run wall-clock (1.366 s) is higher than the parallel row's `repeats=5` mean (0.903 s) because a one-shot pays first-call warmup. The clearest overlap evidence is `ms_total` (2784.79 ms, the *sum* of all five node handler latencies) being ~2× the 1.366 s wall-clock — the four leaves ran concurrently, so summed work far exceeds elapsed time.

Per-node ledger (`cost.csv`):

| node | attempt | tokens_in | tokens_out | ms |
|---|---|---|---|---|
| n1 (root) | 1 | 37 | 2 | 649.7 |
| n2 | 1 | 37 | 2 | 178.6 |
| n3 | 1 | 37 | 2 | 645.0 |
| n4 | 1 | 37 | 2 | 644.2 |
| n5 | 1 | 37 | 2 | 667.3 |

Every node bills 37 in + 2 out = 39 tokens, identical model, `attempt 1` (no retries); the rows sum to the 185 / 10 / 2784.79 ms totals above. **The Amdahl signature is right in the numbers:** wall-clock 1366 ms ≈ `n1` (649.7 — the serial root, runs alone) + the slowest leaf (`n5` 667.3) = **1317 ms**, *not* the 2784.79 ms sum. The four leaves (n2–n5) overlapped, so elapsed time is `t(root) + max(t(leaf))`; the ~50 ms gap is scheduler/claim overhead. This is the fan-out throughput win measured at node granularity — the exact behavior the §4 walkthrough's timeline predicts.

## Subagent status contract + polling-timeout safety-net (deer-flow pattern #1, 2026-06-22)

Additive phase: `src/subagent.py`. A parent that spawns an async/background subagent cannot trust the child's self-reported `running` forever — a wedged child reports `running` indefinitely and the task's *own* timeout never fires. The fix is **two independent timers** (the task budget vs. the parent's poll-loop patience) plus a **closed status contract** (`parse_status`: `completed/failed/cancelled/timed_out/polling_timed_out`) parsed identically on both sides. `poll_until_terminal(poll, poll_timeout=…)` returns `polling_timed_out` once the child has claimed `running` for longer than `poll_timeout`.

| probe | poll_timeout | result | elapsed |
|---|---|---|---|
| stuck child (always `running`), real wall-clock | 0.3 s | `polling_timed_out` | **0.32 s** |
| stuck child, injected fake clock (zero real wait) | 1.0 s | `polling_timed_out` | ≥ 1.0 s (deterministic) |
| healthy child (terminal on 3rd poll) | 10 s | `completed` | safety-net never fires |

`tests/test_subagent.py` — **10 passed in 0.23 s** (parametrized contract parse + the three probes above + terminal-returns-immediately). The safety-net fires in ~`poll_timeout` of real time; the fake-clock test proves the same logic with no real waiting (advance a clock the injected `sleep` ticks). Source: bytedance/deer-flow `subagents/status_contract.py` + `task_tool.py` (EDP Pattern 38). The agentkit shared lib carries the productionised mirror at `agentkit/runtime/subagent.py` (measured 0.31 s — same ~`poll_timeout` behaviour, host noise).

## Concurrent artifact writes: lock the RMW + atomic publish (deer-flow pattern #5, 2026-06-22)

Additive phase: `src/artifact_writer.py`. A concurrent-write bug is *either* a lost update (interleaved read-modify-write) *or* a torn read (reader sees a half-written file) — different failures, different fixes. `locked_update` serializes the whole RMW under Phase 3's `FileLock`; `atomic_write` publishes via `tempfile.mkstemp` + `os.replace` so a reader never sees a partial.

Probe: 8 threads × 50 read-modify-write increments on one shared counter file, with a 0.5 ms gap widening the read→write window so the race is real (not luck). Run 3×:

| variant | run 1 | run 2 | run 3 | invariant |
|---|---|---|---|---|
| **locked** (`locked_update`) | 400/400 | 400/400 | 400/400 | exact every run |
| **unlocked** (bare RMW) | 52/400 | 51/400 | 51/400 | ≈ 349 lost; non-deterministic but reliably « 400 |

The unlocked count is a race, so it is *not* a fixed number — the **loss** is the lesson, and locked-is-always-exact is the proof the lock fixes it. The torn-read test races a reader against 200 big/small rewrites and asserts every observed file length is a *complete* value (`{1, 100000}`), never a torn in-between size. `tests/test_artifact_writer.py` — **3 passed in 0.22 s**. Source: bytedance/deer-flow sandbox file operations (EDP Pattern 42).

**Bug found by running (BCJ Entry 3):** the first `atomic_write` named its temp `f"{name}.{pid}.tmp"` — pid-scoped, so 8 threads of one process collided on one temp file and writers crashed with `FileNotFoundError` mid-rename, driving the unlocked count to 2/400 *by dying*, not by clean races. Fixed with `mkstemp` (unique per call) + `except BaseException: unlink`. A test that passed for the wrong reason until the crash was read.

## Fan-out cost aggregation + parent-level ceiling (deer-flow pattern #2, 2026-06-22)

Additive phase: `src/cost_ceiling.py`. Phase 4's `CostMeter` is a per-*node* ledger; it does not bound a *fan-out*. A per-child `max_tokens` is not a total cap — N children at the cap cost N× the cap. `FanoutBudget` aggregates every child's cost into one running sum at the parent; `run_fanout` aborts the whole fan-out (skips remaining children) the instant the sum crosses the ceiling.

Probe: 10 children, each cost 100 (the per-child "cap").

| config | children ran | total spent | aborted | note |
|---|---|---|---|---|
| no effective ceiling (10000) | 10/10 | 1000 | no | the N× blow-up a per-child cap leaves unbounded |
| **ceiling = 350** | **4/10** | **400** | **yes** | running sum 100,200,300,400 — child 4 crosses → abort, 6 skipped, **saved 600** |

The breaching child (the 4th) is counted in `ran`/`spent` — you can't un-spend it; the ceiling's value is aborting the 6 after it. `tests/test_cost_ceiling.py` — **4 passed in 0.01 s** (the two rows above + a per-child-cap-doesn't-bound-total contrast + `charge` raises with correct `spent`/`ceiling` and clamps `remaining` to 0). **Concurrency caveat:** the lab charges sequentially so the ceiling is exact; under Phase 2's parallel `worker_pool` the check races in-flight children, so the real bound is `ceiling + (in-flight × per-child)` — you can only cancel children that haven't started. Source: bytedance/deer-flow `subagents/token_collector.py` (EDP Pattern 39).
