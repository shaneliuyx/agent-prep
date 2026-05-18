"""Phase 2 — instrumented ReAct loop demonstrating span tree.

Minimal ReAct over local oMLX (no tools beyond a fake calculator)
just to show the span structure flowing to Langfuse. The chapter's
Phase 1 instrumentation primitive is the load-bearing piece; this
script proves it works end-to-end.
"""
from __future__ import annotations

import os
import sys
from openai import OpenAI

from src.tracing import (
    init_tracing, llm_call_span, annotate_usage, traced,
)


SYSTEM = (
    "You are a math assistant. If the user asks an arithmetic question, "
    "respond with the operation expressed as JSON: "
    '{"op": "+|-|*|/", "a": <num>, "b": <num>}. '
    "Otherwise, answer normally."
)


_client = OpenAI(
    base_url=os.getenv("OMLX_BASE_URL", "http://localhost:8000/v1"),
    api_key=os.getenv("OMLX_API_KEY", "not-set"),
)
_MODEL = os.getenv("MODEL_HAIKU", "MLX-Qwen3.5-9B-GLM5.1-Distill-v1-8bit")


@traced("agent_turn")
def agent_turn(user_msg: str) -> str:
    with llm_call_span(role="loop", model=_MODEL) as span:
        resp = _client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        content = (resp.choices[0].message.content or "").strip()
        if resp.usage:
            annotate_usage(span, _MODEL,
                           tokens_in=resp.usage.prompt_tokens,
                           tokens_out=resp.usage.completion_tokens)
    return content


@traced("calculator_tool")
def calculator(op: str, a: float, b: float) -> float:
    if op == "+": return a + b
    if op == "-": return a - b
    if op == "*": return a * b
    if op == "/": return a / b
    raise ValueError(f"unknown op: {op}")


if __name__ == "__main__":
    init_tracing()
    query = sys.argv[1] if len(sys.argv) > 1 else "What's 7 + 5?"
    print(f">>> {query}")
    response = agent_turn(query)
    print(f"<<< {response}")
