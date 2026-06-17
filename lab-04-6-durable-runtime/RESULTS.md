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
