# code/supervisor.py
from __future__ import annotations
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from llm import chat   # tiny chat(prompt, system=None) -> str helper


LEAD_DECOMPOSE_SYSTEM = """You are a research lead. Decompose the user's question into EXACTLY 3 narrow,
non-overlapping sub-questions. Each sub-question should be answerable independently with one focused search.

Return JSON only, no prose:
{"sub_questions": ["q1", "q2", "q3"]}"""

LEAD_SYNTHESIZE_SYSTEM = """You are a research lead synthesizing workers' answers into ONE coherent
answer to the original question. Cite which worker contributed which fact. Surface disagreement explicitly
if workers conflict — do NOT silently pick one side.

CRITICAL: synthesize ONLY what the workers provided. If the original question mentions a topic the workers
did NOT cover (e.g. question asks about EU/US/UK but workers covered EU/US only), state that gap EXPLICITLY
in one sentence and continue with what IS covered. Do NOT speculate or fill in missing material — that
causes reasoning models to loop (W3.5.5.5 BCJ Entry 6). Acknowledge gap, move on."""

WORKER_SYSTEM = """You are a research worker. Answer ONE sub-question with a 3-sentence factual response.
Stay within scope; don't expand to adjacent topics. If you don't know, say so explicitly."""


@dataclass
class WorkerResult:
    sub_question: str
    answer: str
    wall_seconds: float

def plan_decompose(question: str) -> list[str]:
    """Lead's first LLM call: decompose question into 3 sub-questions."""
    raw = chat(question, system=LEAD_DECOMPOSE_SYSTEM)
    # Strip optional ```json fence
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    parsed = json.loads(raw.strip())
    return parsed["sub_questions"]

def worker_run(sub_question: str) -> WorkerResult:
    """One worker: fresh context, narrow answer."""
    t0 = time.monotonic()
    answer = chat(sub_question, system=WORKER_SYSTEM)
    return WorkerResult(
        sub_question=sub_question,
        answer=answer,
        wall_seconds=time.monotonic() - t0,
    )

def synthesize(question: str, results: list[WorkerResult], max_tokens: int = 2048) -> str:
    """Lead's second LLM call: combine worker answers.

    `max_tokens` defaults to 2048 (1024 CoT + 1024 answer for supervisor).
    Hierarchical TOP-lead synthesize should pass HIGHER (e.g. 4096) because
    its input is 2 sub-syntheses (each already table-formatted, multi-
    paragraph, ~600-1000 tokens) PLUS the question — easily 2-3k input
    tokens. Reasoning models reasoning over that need proportionally more
    output budget. W3.5.8 BCJ Entry 8 trap (`finish_reason=length` + empty
    content) recurs at each lifecycle layer: supervisor at 1024 budget,
    hierarchical at 2048 budget — measured 2026-05-28 hierarchical run
    still emitted empty TOP synthesize at 2048; bumped to 4096 in caller.
    """
    bundle = "\n\n".join(
        f"Worker {i+1} on '{r.sub_question}':\n{r.answer}"
        for i, r in enumerate(results)
    )
    prompt = f"ORIGINAL QUESTION: {question}\n\nWORKER ANSWERS:\n{bundle}\n\nSYNTHESIZE."
    return chat(prompt, system=LEAD_SYNTHESIZE_SYSTEM, max_tokens=max_tokens)

def supervisor_run(question: str) -> dict:
    """End-to-end supervisor topology. Returns answer + timing breakdown."""
    t_total = time.monotonic()

    t_plan = time.monotonic()
    sub_qs = plan_decompose(question)
    plan_wall = time.monotonic() - t_plan

    with ThreadPoolExecutor(max_workers=len(sub_qs)) as pool:
        results = list(pool.map(worker_run, sub_qs))

    t_syn = time.monotonic()
    answer = synthesize(question, results)
    syn_wall = time.monotonic() - t_syn

    return {
        "answer": answer,
        "sub_questions": sub_qs,
        "workers": [
            {"sub_question": r.sub_question, "answer": r.answer,
             "wall_s": round(r.wall_seconds, 2)}
            for r in results
        ],
        "plan_wall_s": round(plan_wall, 2),
        "worker_walls_s": [round(r.wall_seconds, 2) for r in results],
        "max_worker_wall_s": round(max(r.wall_seconds for r in results), 2),
        "sum_worker_walls_s": round(sum(r.wall_seconds for r in results), 2),
        "synthesize_wall_s": round(syn_wall, 2),
        "total_wall_s": round(time.monotonic() - t_total, 2),
    }


def print_supervisor_tree(out: dict) -> None:
    """Pretty-print supervisor run with per-agent visibility."""
    print(f"\n{'=' * 70}")
    print(f"SUPERVISOR TOPOLOGY — total {out['total_wall_s']}s "
          f"(plan {out['plan_wall_s']}s + max_worker {out['max_worker_wall_s']}s "
          f"+ synth {out['synthesize_wall_s']}s)")
    print(f"{'=' * 70}\n")
    print(f"LEAD (plan): decomposed into {len(out['workers'])} sub-questions:")
    for i, w in enumerate(out["workers"], 1):
        print(f"\n  Worker {i} (wall {w['wall_s']}s)")
        print(f"  ├─ sub-question: {w['sub_question']}")
        print(f"  └─ answer:")
        for line in (w["answer"] or "<empty>").splitlines():
            print(f"     │ {line}")
    print(f"\n{'─' * 70}")
    print(f"LEAD (synthesize): combined {len(out['workers'])} worker answers:")
    print(f"{'─' * 70}")
    for line in (out["answer"] or "<EMPTY — BCJ Entry 8 trap, bump max_tokens>").splitlines():
        print(f"  {line}")
    print(f"\n{'=' * 70}")
    # Guard against ZeroDivisionError on instant workers (mock provider /
    # cached responses can produce max_worker_wall_s = 0.0)
    ratio = out['sum_worker_walls_s'] / max(out['max_worker_wall_s'], 0.01)
    print(f"PARALLELISM RECEIPT: sum_worker_walls_s = {out['sum_worker_walls_s']}s, "
          f"max_worker_wall_s = {out['max_worker_wall_s']}s, "
          f"ratio = {ratio:.2f}× "
          f"(close to N-1={len(out['workers'])-1} = full parallelism)")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    out = supervisor_run("What changed in multi-agent agent systems between 2023 and 2026?")
    print_supervisor_tree(out)
