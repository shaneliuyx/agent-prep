"""
src/impl_plan_solve.py — Plan-and-Solve for the pattern zoo.

Plan-and-Solve = an explicit PLAN phase ("devise a plan, break the task into
ordered subtasks") followed by a SOLVE phase that executes the plan one step at a
time. The defining trait FOR THIS ZOO: the solver re-feeds each observation and is
allowed to adapt — it sees the result of step k before issuing step k+1, exactly
like ReAct but front-loaded with an explicit plan. This is the foil for ReWOO:
ReWOO plans ALL tool calls up front and NEVER re-feeds an observation into the
reasoning loop. Keeping Plan-Solve observation-driven makes that contrast real.

Entry point: run(task: str) -> AgentResult
"""

from __future__ import annotations

import json
import time

from src.llm_client import call_llm
from src.schema import CANONICAL_TASK, AgentResult
from src.tools import TOOL_SCHEMAS, dispatch

PATTERN = "Plan-and-Solve"
MAX_STEPS = 10

_PLAN_SYSTEM = """You are a planner. Read the user's task and write a short, ordered
plan: number each step, one action per line. Mention which tool each step uses.
Do NOT execute anything yet — just produce the plan as plain text."""

_SOLVE_SYSTEM = """You are executing a plan step by step using tools.

You have a plan and the results of any steps already done. Issue the NEXT tool
call needed to make progress. Use the observations from earlier steps as inputs to
later steps. When the plan is complete and you have the final number, respond in
plain text with the final answer and NO tool call."""


def run(task: str = CANONICAL_TASK) -> AgentResult:
    """Plan once, then solve step-by-step (re-feeding observations)."""
    t0 = time.perf_counter()
    llm_calls = 0
    prompt_tokens = 0
    completion_tokens = 0
    trace: list[str] = []

    # --- Phase 1: PLAN (one LLM call, no tools) ---
    plan_resp = call_llm(
        [
            {"role": "system", "content": _PLAN_SYSTEM},
            {"role": "user", "content": task},
        ],
        tools=None,
    )
    llm_calls += 1
    prompt_tokens += plan_resp.prompt_tokens
    completion_tokens += plan_resp.completion_tokens
    plan = plan_resp.content
    trace.append(f"plan:\n{plan}")

    # --- Phase 2: SOLVE (observation-driven loop, WITH tools) ---
    messages: list[dict] = [
        {"role": "system", "content": _SOLVE_SYSTEM},
        {"role": "user", "content": f"Task: {task}\n\nPlan:\n{plan}"},
    ]

    for _ in range(MAX_STEPS):
        resp = call_llm(messages, tools=TOOL_SCHEMAS)
        llm_calls += 1
        prompt_tokens += resp.prompt_tokens
        completion_tokens += resp.completion_tokens

        if not resp.tool_calls:
            trace.append(f"final: {resp.content!r}")
            return AgentResult(
                pattern=PATTERN,
                answer=resp.content,
                llm_calls=llm_calls,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=int((time.perf_counter() - t0) * 1000),
                trace=tuple(trace),
            )

        messages.append({
            "role": "assistant",
            "content": resp.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in resp.tool_calls
            ],
        })
        for tc in resp.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
                result = f"ERROR: arguments for {name!r} were not valid JSON: {tc.function.arguments!r}"
            else:
                result = dispatch(name, args)
            trace.append(f"act {name}({args}) -> {result}")
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": name,
                "content": result,
            })

    return AgentResult(
        pattern=PATTERN,
        answer=f"[Plan-and-Solve stopped: reached MAX_STEPS={MAX_STEPS} without a final answer]",
        llm_calls=llm_calls,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=int((time.perf_counter() - t0) * 1000),
        trace=tuple(trace),
    )


if __name__ == "__main__":
    print(run())
