# lab-04-6 — Durable Agent-Graph Runtime

A small, durable runtime for agent **workflows-as-DAGs**. A process can die
anywhere (`kill -9`, OOM, Ctrl-C) and a run resumes from the last persisted node
with nothing lost and nothing done twice — because execution state lives in
SQLite, not in Python locals.

## Architecture (the seams)

| Module | Responsibility |
|---|---|
| `src/graph_store.py` | **(given, correct)** SQLite state machine: graphs/runs/nodes/executions; atomic claim, retry, recover, replay. |
| `src/file_lock.py` | **(given, correct)** cross-process advisory `fcntl.flock` claim lock. |
| `src/cost_meter.py` | retry-safe per-node cost ledger, `UNIQUE(run_id,node,attempt)` idempotency, cloud-equivalent USD. |
| `src/handlers.py` | node handlers: deterministic `tool_handler`, `make_llm_handler` (raw oMLX call, real `usage`). |
| `src/worker_pool.py` | `run_graph`: N async workers draining the READY frontier; peak-concurrency tracking; SIGTERM drain. |
| `src/topologies.py` | four 5-node DAG shapes (sequential / parallel / hierarchical / workflow). |
| `src/scheduler.py` | external triggers (cron / webhook / manual). **No self-prompting loop** — every run is externally fired. |

## Run it

All commands use the local venv (Python 3.14, `openai` 2.31) and call oMLX
directly on `:8000` (local-first, zero cloud).

```bash
PY=/Users/yuxinliu/.openharness-venv/bin/python3

# 1. end-to-end demo (one topology via scheduler + worker pool, live oMLX)
$PY examples/example_graph.py

# 2. durability tests (offline, deterministic — the kill-and-recover proof)
$PY -m pytest tests/test_durability.py -v

# 3. the headline bench: four-topology throughput + recovery (hits oMLX)
$PY bench_four_topology.py        # writes RESULTS.md
```

## What the bench measures

Holds **model constant** (`Qwen2.5-Coder-7B-Instruct-MLX-4bit`) and **node count
constant** (5) across all four topologies, so the only independent variable is
dependency *structure*. Reports mean wall-clock, summed real token usage, and
observed peak concurrency per topology, plus a partial-failure recovery time
(child process SIGKILLed mid-run, then `recover_run` + finish). See `RESULTS.md`.

## Durability invariant

Every status mutation and its event row commit in the **same** transaction
(`graph_store`), so the event log (`replay_run`) is an exact, double-work-free
audit trail. On restart, `recover_run` resets orphaned `RUNNING` nodes to `READY`
and the frontier is re-derived from the table — never from memory.
