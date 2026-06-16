"""
src/05_compare.py — run every pattern on the canonical task, judge correctness,
and print the comparison matrix.

Each impl exposes  run(task) -> AgentResult  (see src/schema.py). This harness:
  1. runs them all on src.schema.CANONICAL_TASK,
  2. scores correctness two ways:
       - exact: does the expected number (126000) appear in the answer? (cheap,
         deterministic ground-truth check)
       - judge: an LLM-as-judge PASS/FAIL on the SAME local oMLX model,
  3. prints a markdown matrix with the columns RESULTS.md uses:
       Pattern | Correct (exact/judge) | LLM calls | Prompt tok | Completion tok | Total tok | Latency (ms)

Run:
    cd lab-05-pattern-zoo
    set -a; source .env; set +a
    /Users/yuxinliu/code/agent-prep/.venv/bin/python -m src.05_compare
(module name starts with a digit, so use -m with the package path, or run via the
runpy shim at the bottom of this file with `python src/05_compare.py`).
"""

from __future__ import annotations

import importlib
import re
import sys
import time

from src.llm_client import call_llm
from src.schema import CANONICAL_EXPECTED_ANSWER, CANONICAL_TASK, AgentResult

# (row label, module path) — adding a pattern is one line here.
IMPLS = [
    ("ReAct", "src.impl_react"),
    ("Plan-and-Solve", "src.impl_plan_solve"),
    ("CodeAct", "src.impl_codeact"),
    ("ReWOO", "src.impl_rewoo"),
]

_EXPECTED_NUM = float(CANONICAL_EXPECTED_ANSWER)


def _exact_correct(answer: str) -> bool:
    """Ground-truth check: does the expected value appear as a number in the
    answer? Tolerates thousands separators (126,000) and trailing .0 (126000.0)."""
    cleaned = answer.replace(",", "")
    for tok in re.findall(r"-?\d+(?:\.\d+)?", cleaned):
        try:
            if abs(float(tok) - _EXPECTED_NUM) < 1e-6:
                return True
        except ValueError:
            continue
    return False


_JUDGE_SYSTEM = """You are a strict grader. You are given a TASK, the EXPECTED final
number, and an AGENT ANSWER. Reply with exactly one word: PASS if the agent's answer
contains the correct final number, FAIL otherwise. No explanation."""


def _judge_correct(answer: str) -> tuple[bool, int, int]:
    """LLM-as-judge PASS/FAIL on the local oMLX model. Returns (passed, ptok, ctok)."""
    resp = call_llm(
        [
            {"role": "system", "content": _JUDGE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"TASK:\n{CANONICAL_TASK}\n\n"
                    f"EXPECTED final number: {CANONICAL_EXPECTED_ANSWER}\n\n"
                    f"AGENT ANSWER:\n{answer}\n\nGrade:"
                ),
            },
        ],
        tools=None,
        max_tokens=8,
    )
    verdict = resp.content.strip().upper()
    return verdict.startswith("PASS"), resp.prompt_tokens, resp.completion_tokens


def run_all() -> list[tuple[AgentResult, bool, bool]]:
    """Run every impl; return (result, exact_ok, judge_ok) per pattern."""
    rows: list[tuple[AgentResult, bool, bool]] = []
    for label, modpath in IMPLS:
        print(f"--- running {label} ({modpath}) ---", file=sys.stderr)
        mod = importlib.import_module(modpath)
        try:
            result = mod.run(CANONICAL_TASK)
        except Exception as e:  # a real failure must be reported, never faked
            print(f"!!! {label} FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            result = AgentResult(
                pattern=label, answer=f"[RUN FAILED: {type(e).__name__}: {e}]",
                llm_calls=0, prompt_tokens=0, completion_tokens=0, latency_ms=0,
            )
        exact_ok = _exact_correct(result.answer)
        judge_ok, _, _ = _judge_correct(result.answer)
        print(
            f"    answer={result.answer[:80]!r} exact={exact_ok} judge={judge_ok} "
            f"calls={result.llm_calls} tok={result.total_tokens} ms={result.latency_ms}",
            file=sys.stderr,
        )
        rows.append((result, exact_ok, judge_ok))
    return rows


def print_matrix(rows: list[tuple[AgentResult, bool, bool]]) -> None:
    """Emit the markdown comparison matrix (the RESULTS.md format)."""
    print("\n## Pattern comparison — canonical task\n")
    print(f"Task: {CANONICAL_TASK}")
    print(f"\nExpected answer: **{CANONICAL_EXPECTED_ANSWER}**\n")
    print("| Pattern | Correct (exact) | Correct (judge) | LLM calls | Prompt tok | Completion tok | Total tok | Latency (ms) |")
    print("|---|---|---|---|---|---|---|---|")
    for result, exact_ok, judge_ok in rows:
        print(
            f"| {result.pattern} "
            f"| {'PASS' if exact_ok else 'FAIL'} "
            f"| {'PASS' if judge_ok else 'FAIL'} "
            f"| {result.llm_calls} "
            f"| {result.prompt_tokens} "
            f"| {result.completion_tokens} "
            f"| {result.total_tokens} "
            f"| {result.latency_ms} |"
        )


def main() -> None:
    t0 = time.perf_counter()
    rows = run_all()
    print_matrix(rows)
    print(f"\n_Total wall-clock for the full comparison: "
          f"{int((time.perf_counter() - t0) * 1000)} ms_")


if __name__ == "__main__":
    main()
