# code/voting.py
from __future__ import annotations
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from llm import chat


SOLVER_PROMPTS = [
    # Three distinct prompt variants to maximize answer independence
    "You are a careful, step-by-step solver. Show your reasoning. End with: ANSWER: <final>",
    "You are an efficient solver. Skip reasoning; give just the answer. End with: ANSWER: <final>",
    "You are a creative solver. Consider unusual angles. End with: ANSWER: <final>",
]


@dataclass
class SolverResult:
    solver_id: int
    raw_response: str
    extracted_answer: str

def _extract_answer(raw: str) -> str:
    """Pull 'ANSWER: ...' line from a solver's raw response."""
    m = re.search(r"ANSWER:\s*(.+?)(?:\n|$)", raw, re.IGNORECASE)
    return (m.group(1) if m else raw).strip().rstrip(".").lower()

def solve_one(args: tuple[int, str, str]) -> SolverResult:
    """One solver runs the question with its prompt variant."""
    solver_id, question, system_prompt = args
    raw = chat(question, system=system_prompt)
    return SolverResult(
        solver_id=solver_id,
        raw_response=raw,
        extracted_answer=_extract_answer(raw),
    )

def aggregate_majority(results: list[SolverResult]) -> dict:
    """Majority vote on the extracted answers (normalized to lowercase, stripped)."""
    counts = Counter(r.extracted_answer for r in results)
    winner, count = counts.most_common(1)[0]
    return {
        "method": "majority",
        "answer": winner,
        "votes": dict(counts),
        "confidence": count / len(results),
    }

def aggregate_llm_judge(question: str, results: list[SolverResult]) -> dict:
    """LLM reads all 3 answers + picks the best."""
    bundle = "\n\n".join(
        f"SOLVER {r.solver_id}:\n{r.raw_response}"
        for r in results
    )
    prompt = (
        f"QUESTION: {question}\n\n{bundle}\n\n"
        f"Which solver's ANSWER is most accurate? Reply with EXACTLY:\n"
        f"BEST: <solver_id>\n"
        f"REASON: <one sentence>"
    )
    reply = chat(prompt)
    m = re.search(r"BEST:\s*(\d+)", reply)
    if m:
        best_id = int(m.group(1))
        winner = next((r for r in results if r.solver_id == best_id), results[0])
        return {
            "method": "llm-judge",
            "answer": winner.extracted_answer,
            "judge_reply": reply,
            "winning_solver": best_id,
        }
    # Defensive fallback: majority
    return aggregate_majority(results)

def voting_run(
    question: str,
    aggregator: str = "majority",
) -> dict:
    """Run 3 solvers in parallel + aggregate."""
    tasks = [(i, question, prompt) for i, prompt in enumerate(SOLVER_PROMPTS)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(solve_one, tasks))

    if aggregator == "majority":
        agg = aggregate_majority(results)
    elif aggregator == "llm-judge":
        agg = aggregate_llm_judge(question, results)
    else:
        raise ValueError(f"unknown aggregator: {aggregator}")

    return {
        "question": question,
        "aggregator": aggregator,
        "solver_answers": [r.extracted_answer for r in results],
        "aggregate": agg,
    }


if __name__ == "__main__":
    import json
    questions = [
        "What is 137 * 23?",
        "Is the Eiffel Tower in Paris? Yes or No.",
        "What year was Python first released?",
    ]
    for q in questions:
        print(f"\nQ: {q}")
        print("Majority:", voting_run(q, "majority")["aggregate"])
        print("LLM-judge:", voting_run(q, "llm-judge")["aggregate"])
