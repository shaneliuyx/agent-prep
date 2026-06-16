"""
src/impl_react.py — ReAct reference for the pattern zoo.

ReAct = interleave Reasoning + Acting: one structured tool call per LLM turn, the
observation is fed BACK into the next LLM call. So an N-step task costs ~N+1 LLM
round-trips. This is the baseline the other patterns are measured against.

Distilled from lab-04-react-from-scratch/src/react.py (same oMLX plumbing, same
OpenAI function-calling tool loop) and wrapped in the zoo's common entry point:

    run(task: str) -> AgentResult
"""

from __future__ import annotations

import json
import time

from src.llm_client import call_llm
from src.schema import CANONICAL_TASK, AgentResult
from src.tools import TOOL_SCHEMAS, dispatch

PATTERN = "ReAct"
MAX_ITER = 10

SYSTEM_PROMPT = """You are a capable assistant with access to tools.

Work step by step. When you need a fact or an arithmetic result, call the
appropriate tool. After you have everything you need, respond in plain text with
the final answer and NO tool call — that ends the loop.

Rules:
- Call tools one at a time.
- Use the tool results (observations) to decide the next call.
- When you give the final answer, state the final number clearly.
"""


def run(task: str = CANONICAL_TASK) -> AgentResult:
    """Run the ReAct loop on `task`; return the measured AgentResult."""
    t0 = time.perf_counter()
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]
    llm_calls = 0
    prompt_tokens = 0
    completion_tokens = 0
    trace: list[str] = []

    for _ in range(MAX_ITER):
        resp = call_llm(messages, tools=TOOL_SCHEMAS)
        llm_calls += 1
        prompt_tokens += resp.prompt_tokens
        completion_tokens += resp.completion_tokens

        # No tool call -> final answer; loop ends.
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

        # Record the assistant turn (with its tool_calls) for the next context.
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

        # Execute each call; append the observation as a tool message.
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

    # MAX_ITER exhausted without a final answer.
    return AgentResult(
        pattern=PATTERN,
        answer=f"[ReAct stopped: reached MAX_ITER={MAX_ITER} without a final answer]",
        llm_calls=llm_calls,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=int((time.perf_counter() - t0) * 1000),
        trace=tuple(trace),
    )


if __name__ == "__main__":
    r = run()
    print(r)
