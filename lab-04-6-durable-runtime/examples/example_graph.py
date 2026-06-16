"""example_graph.py — one durable run end-to-end, the smallest complete demo.

Run it:  /Users/yuxinliu/.openharness-venv/bin/python3 examples/example_graph.py

It wires every seam of the runtime together exactly once:
  topology builder → GraphStore.create_graph → Scheduler.trigger_manually
  (the EXTERNAL trigger) → worker_pool.run_graph (async workers) → cost report.

By default it runs the `parallel` topology with LIVE oMLX llm nodes so you see
real token counts. The point of the demo is to make the four-table durability
core observable: after the run, node_states shows every node DONE and the cost
report shows the cloud-equivalent spend the run WOULD have cost.
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile

# src/ on path; this lab imports its own modules by bare name (no packaging).
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

import openai  # noqa: E402

from cost_meter import CostMeter  # noqa: E402
from graph_store import GraphStore  # noqa: E402
from handlers import make_llm_handler  # noqa: E402
from scheduler import Scheduler  # noqa: E402
from topologies import BENCH_MODEL, parallel  # noqa: E402
from worker_pool import run_graph  # noqa: E402

OMLX_BASE = "http://localhost:8000/v1"


def main() -> None:
    tmp = tempfile.mkdtemp(prefix="durable_demo_")
    db = os.path.join(tmp, "runtime.db")
    cost_db = os.path.join(tmp, "cost.db")

    store = GraphStore(db)
    cm = CostMeter(cost_db)
    scheduler = Scheduler(store)
    client = openai.OpenAI(base_url=OMLX_BASE, api_key="EMPTY")

    # 1) author the DAG template (live llm nodes so usage is real)
    dag, concurrency = parallel(llm=True, model=BENCH_MODEL)
    graph_id = store.create_graph("demo-parallel", dag)
    print(f"created graph {graph_id}: {list(dag['nodes'])} (concurrency={concurrency})")

    # 2) fire it via an EXTERNAL trigger — the only way a run ever starts
    run_id = scheduler.trigger_manually(graph_id, {"by": "example"})
    print(f"triggered run {run_id} (trigger=manual)")

    # 3) drain it with async workers
    handler = make_llm_handler(client, BENCH_MODEL, cost_meter=cm)
    result = asyncio.run(run_graph(store, run_id, handler, concurrency, cost_meter=cm))

    # 4) observe durable state + cost
    print("\n── final node states ──")
    for name, status in sorted(store.node_states(run_id).items()):
        print(f"  {name:10s} {status}")
    print(f"\nrun status: {store.run_status(run_id)}")
    print("pool result:", result)
    print("cost report:", cm.cost_report(run_id))

    csv_path = os.path.join(tmp, "cost.csv")
    cm.export_csv(run_id, csv_path)
    print("cost csv:", csv_path)


if __name__ == "__main__":
    main()
