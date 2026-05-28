# code/hierarchical.py
from __future__ import annotations
import time
from concurrent.futures import ThreadPoolExecutor
from supervisor import plan_decompose, worker_run, synthesize, WorkerResult

def sub_supervisor(macro_question: str) -> WorkerResult:
    """Sub-lead: decompose macro into 2 sub-questions, run 2 workers, synthesize."""
    t0 = time.monotonic()
    sub_qs = plan_decompose(macro_question)[:2]   # cap at 2 sub-qs per macro
    with ThreadPoolExecutor(max_workers=2) as pool:
        leaf_results = list(pool.map(worker_run, sub_qs))
    sub_answer = synthesize(macro_question, leaf_results)
    return WorkerResult(
        sub_question=macro_question,
        answer=sub_answer,
        wall_seconds=time.monotonic() - t0,
    )

def hierarchical_run(question: str) -> dict:
    """Two-layer hierarchy: top lead → 2 sub-leads → 4 leaf workers."""
    t_total = time.monotonic()
    t_plan = time.monotonic()
    macros = plan_decompose(question)[:2]   # cap at 2 macros
    plan_wall = time.monotonic() - t_plan

    with ThreadPoolExecutor(max_workers=len(macros)) as pool:
        sub_results = list(pool.map(sub_supervisor, macros))

    t_syn = time.monotonic()
    answer = synthesize(question, sub_results)
    syn_wall = time.monotonic() - t_syn

    return {
        "answer": answer,
        "depth": 2,
        "agents_total": 1 + len(macros) + len(macros) * 2,   # top + sub-leads + leaves
        "plan_wall_s": round(plan_wall, 2),
        "sub_walls_s": [round(r.wall_seconds, 2) for r in sub_results],
        "max_sub_wall_s": round(max(r.wall_seconds for r in sub_results), 2),
        "synthesize_wall_s": round(syn_wall, 2),
        "total_wall_s": round(time.monotonic() - t_total, 2),
    }


if __name__ == "__main__":
    import json
    out = hierarchical_run(
        "Compare regulatory frameworks for AI across EU, US, and UK."
    )
    print(json.dumps(out, indent=2))
