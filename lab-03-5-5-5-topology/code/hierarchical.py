# code/hierarchical.py
from __future__ import annotations
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from supervisor import plan_decompose, worker_run, synthesize, WorkerResult


@dataclass
class SubLeadResult:
    """Sub-lead's full output: its macro question, the sub-questions it
    decomposed into, each leaf worker's answer + wall, the synthesized
    sub-answer, and the sub-lead's total wall."""
    macro_question: str
    sub_questions: list[str]
    leaf_results: list[WorkerResult]
    sub_answer: str
    wall_seconds: float


def sub_supervisor(macro_question: str) -> SubLeadResult:
    """Sub-lead: decompose macro into 2 sub-questions, run 2 leaf workers in
    parallel, synthesize. Returns full per-agent structure (NOT just timing)."""
    t0 = time.monotonic()
    sub_qs = plan_decompose(macro_question)[:2]   # cap at 2 sub-qs per macro
    with ThreadPoolExecutor(max_workers=2) as pool:
        leaf_results = list(pool.map(worker_run, sub_qs))
    sub_answer = synthesize(macro_question, leaf_results)
    return SubLeadResult(
        macro_question=macro_question,
        sub_questions=sub_qs,
        leaf_results=leaf_results,
        sub_answer=sub_answer,
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

    # Top-lead synthesizes by treating each sub-lead's output as a WorkerResult
    sub_as_workers = [
        WorkerResult(sub_question=s.macro_question, answer=s.sub_answer,
                     wall_seconds=s.wall_seconds)
        for s in sub_results
    ]
    t_syn = time.monotonic()
    # Bump max_tokens for TOP-lead synthesize — each sub-answer is itself
    # a synthesized output (table-formatted, multi-paragraph, ~600-1000
    # tokens), so input is 2-3k tokens. Measured 2026-05-28: 2048 budget
    # still produced empty TOP synthesize at 54s wall (CoT exhausted entire
    # budget). 4096 = 2048 CoT + 2048 answer headroom on reasoning models.
    answer = synthesize(question, sub_as_workers, max_tokens=4096)
    syn_wall = time.monotonic() - t_syn

    return {
        "answer": answer,
        "macros": macros,
        "sub_leads": [
            {
                "macro_question": s.macro_question,
                "sub_questions": s.sub_questions,
                "leaves": [
                    {"sub_question": lf.sub_question, "answer": lf.answer,
                     "wall_s": round(lf.wall_seconds, 2)}
                    for lf in s.leaf_results
                ],
                "sub_answer": s.sub_answer,
                "wall_s": round(s.wall_seconds, 2),
            }
            for s in sub_results
        ],
        "depth": 2,
        "agents_total": 1 + len(macros) + len(macros) * 2,   # top + sub-leads + leaves
        "plan_wall_s": round(plan_wall, 2),
        "sub_walls_s": [round(s.wall_seconds, 2) for s in sub_results],
        "max_sub_wall_s": round(max(s.wall_seconds for s in sub_results), 2),
        "synthesize_wall_s": round(syn_wall, 2),
        "total_wall_s": round(time.monotonic() - t_total, 2),
    }


def print_hierarchy_tree(out: dict) -> None:
    """Pretty-print 2-layer hierarchical run with per-agent visibility."""
    print(f"\n{'=' * 70}")
    print(f"HIERARCHICAL TOPOLOGY — depth {out['depth']}, "
          f"{out['agents_total']} agents, total {out['total_wall_s']}s")
    print(f"  plan {out['plan_wall_s']}s + max_sub {out['max_sub_wall_s']}s "
          f"+ synth {out['synthesize_wall_s']}s")
    print(f"{'=' * 70}\n")
    print(f"TOP LEAD (plan): decomposed into {len(out['sub_leads'])} macros\n")
    for i, sl in enumerate(out["sub_leads"], 1):
        print(f"┌─ Sub-lead {i} (wall {sl['wall_s']}s)")
        print(f"│  macro: {sl['macro_question']}")
        print(f"│  decomposed into {len(sl['leaves'])} leaf sub-questions:")
        for j, lf in enumerate(sl["leaves"], 1):
            print(f"│    Leaf {i}.{j} (wall {lf['wall_s']}s)")
            print(f"│    ├─ sub-question: {lf['sub_question']}")
            print(f"│    └─ answer:")
            for line in (lf["answer"] or "<empty>").splitlines():
                print(f"│       │ {line}")
        print(f"│")
        print(f"│  Sub-synthesis:")
        for line in (sl["sub_answer"] or "<empty>").splitlines():
            print(f"│    │ {line}")
        print(f"└─{'─' * 60}\n")
    print(f"{'─' * 70}")
    print(f"TOP LEAD (synthesize): combined {len(out['sub_leads'])} sub-answers:")
    print(f"{'─' * 70}")
    for line in (out["answer"] or "<EMPTY — BCJ Entry 8 trap, bump max_tokens>").splitlines():
        print(f"  {line}")
    print(f"\n{'=' * 70}\n")


if __name__ == "__main__":
    out = hierarchical_run(
        "Compare regulatory frameworks for AI across EU, US, and UK."
    )
    print_hierarchy_tree(out)
