"""
src/impl_rewoo.py — ReWOO (Reasoning WithOut Observation) for the pattern zoo.

ReWOO (Xu et al., 2023) splits the agent into three roles, and crucially the
PLANNER never sees a tool observation:

  1. PLANNER  — ONE LLM call. Emits the ENTIRE plan of tool calls up front, using
                #E1, #E2 ... variable placeholders where a later step references
                an earlier step's result. The planner reasons purely from the
                task — it does NOT call tools and never sees their output.
  2. WORKER   — pure execution, NO LLM. Walks the plan top to bottom, substitutes
                already-computed #E values into each step's arguments, runs the
                tool, and stores the result as that step's #E. Observations live
                only in this evidence table; they are never sent back to a model.
  3. SOLVER   — ONE LLM call. Given the task + the filled evidence table, writes
                the final answer.

Total LLM calls = 2 (planner + solver), INDEPENDENT of the number of tool steps.
Contrast: ReAct / Plan-and-Solve re-feed every observation into the model, so each
tool step costs an LLM round-trip and the prompt grows with the trajectory. ReWOO
pays the trajectory cost in the Worker (free, no tokens), so the token-savings
claim is structural — and measured here via real response.usage.

Difference from src/impl_plan_solve.py: Plan-Solve REPLANS with observations (the
solver re-feeds each tool result and adapts step by step). ReWOO does not — the
plan is fixed before any tool runs, and no observation ever re-enters the LLM.

Entry point: run(task: str) -> AgentResult
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from src.llm_client import call_llm
from src.schema import CANONICAL_TASK, AgentResult
from src.tools import TOOLS, dispatch, tools_doc

PATTERN = "ReWOO"

# A plan line:  #E1 = tool_name[json-args]
#   e.g.  #E1 = kb_lookup[{"key": "Springfield"}]
#         #E3 = add[{"a": "#E1", "b": "#E2"}]
_PLAN_LINE = re.compile(r"#E(\d+)\s*=\s*(\w+)\s*\[(.*?)\]\s*$", re.MULTILINE)
_EVIDENCE_REF = re.compile(r"#E\d+")

_PLANNER_SYSTEM = f"""You are the ReWOO Planner. Produce a COMPLETE plan of tool
calls to solve the task, all at once, WITHOUT calling any tools and WITHOUT seeing
any results. Use variable placeholders #E1, #E2, ... for each step's result, and
reference earlier results by their placeholder in later steps.

Tools available:
{tools_doc()}

Output format — one step per line, nothing else, no prose:
#E1 = tool_name[{{"arg": value}}]
#E2 = tool_name[{{"arg": "#E1"}}]
...

Rules:
- Arguments are a JSON object inside the [ ].
- To pass a previous step's result as an argument, use its placeholder string,
  e.g. {{"a": "#E1", "b": "#E2"}} — the worker substitutes the real value.
- Plan EVERY step needed to reach the final number. Do not stop early.
- Emit ONLY the #E lines."""

_SOLVER_SYSTEM = """You are the ReWOO Solver. You are given the original task and an
evidence table mapping each plan variable (#E1, #E2, ...) to the value the worker
computed. Using ONLY this evidence, state the final answer. Give the final number
clearly."""


def _parse_plan(plan_text: str) -> list[tuple[str, str, str]]:
    """Parse plan text into ordered (var, tool, raw_args) tuples."""
    steps = []
    for m in _PLAN_LINE.finditer(plan_text):
        var = f"#E{m.group(1)}"
        tool = m.group(2)
        raw_args = m.group(3).strip()
        steps.append((var, tool, raw_args))
    return steps


def _substitute(raw_args: str, evidence: dict[str, str]) -> dict[str, Any]:
    """Resolve #E refs in a step's JSON args from the evidence table.

    Replaces every "#En" token (whether it appears as a bare value or inside a
    string) with the computed value before json.loads. Numeric evidence is
    substituted unquoted so JSON stays valid (e.g. {"a": "#E1"} -> {"a": 30000.0}).
    """
    def repl(match: re.Match) -> str:
        ref = match.group(0)
        val = evidence.get(ref, ref)
        return str(val)

    # Replace "#En" (quoted placeholder) with the raw value, then bare #En too.
    s = re.sub(r'"(#E\d+)"', lambda m: repl(re.match(r"#E\d+", m.group(1))), raw_args)
    s = _EVIDENCE_REF.sub(repl, s)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # Fall back: best-effort single-key parse so the worker can still surface
        # a real error observation rather than crashing.
        return {"__raw__": s}


def run(task: str = CANONICAL_TASK) -> AgentResult:
    """Planner (1 LLM) -> Worker (0 LLM) -> Solver (1 LLM)."""
    t0 = time.perf_counter()
    llm_calls = 0
    prompt_tokens = 0
    completion_tokens = 0
    trace: list[str] = []

    # --- Phase 1: PLANNER (one LLM call, no tools) ---
    plan_resp = call_llm(
        [
            {"role": "system", "content": _PLANNER_SYSTEM},
            {"role": "user", "content": task},
        ],
        tools=None,
        max_tokens=512,
    )
    llm_calls += 1
    prompt_tokens += plan_resp.prompt_tokens
    completion_tokens += plan_resp.completion_tokens
    plan_text = plan_resp.content
    trace.append(f"plan:\n{plan_text}")

    steps = _parse_plan(plan_text)
    trace.append(f"parsed {len(steps)} steps")

    # --- Phase 2: WORKER (NO LLM — pure execution) ---
    evidence: dict[str, str] = {}
    for var, tool, raw_args in steps:
        args = _substitute(raw_args, evidence)
        if "__raw__" in args:
            result = f"ERROR: could not parse args for {tool!r}: {args['__raw__']!r}"
        else:
            result = dispatch(tool, args)
        evidence[var] = result
        trace.append(f"worker {var} = {tool}({args}) -> {result}")

    evidence_table = "\n".join(f"{v} = {val}" for v, val in evidence.items()) or "(empty)"

    # --- Phase 3: SOLVER (one LLM call, no tools) ---
    solve_resp = call_llm(
        [
            {"role": "system", "content": _SOLVER_SYSTEM},
            {"role": "user", "content": f"Task: {task}\n\nEvidence:\n{evidence_table}"},
        ],
        tools=None,
    )
    llm_calls += 1
    prompt_tokens += solve_resp.prompt_tokens
    completion_tokens += solve_resp.completion_tokens
    trace.append(f"final: {solve_resp.content!r}")

    return AgentResult(
        pattern=PATTERN,
        answer=solve_resp.content,
        llm_calls=llm_calls,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=int((time.perf_counter() - t0) * 1000),
        trace=tuple(trace),
    )


if __name__ == "__main__":
    print(run())
