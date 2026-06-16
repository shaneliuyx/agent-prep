"""handlers.py — node handlers: the pluggable "what a node actually does".

WHY handlers are a separate seam from the runtime:
graph_store + worker_pool own *scheduling and durability* (claim, retry, recover).
They are deliberately ignorant of what a node computes. A handler is the function
that turns a claimed `Node` into a result dict `{"tokens", "ms", "text"?}`. This
split is the whole point of the architecture: you can swap an LLM call for a tool
call, or a real model for a deterministic sleep, WITHOUT touching the durable
core. The durability test (tests/test_durability.py) leans on exactly this: it
runs the runtime with a deterministic `tool_handler` so a kill-and-recover test
has zero LLM nondeterminism.

WHY we call the raw OpenAI client and not shared/llm.chat:
shared/llm.chat returns only the text and DISCARDS `response.usage`. This lab's
headline number is *real* token counts summed across a topology, so we must read
`usage.prompt_tokens` / `usage.completion_tokens` off the raw response ourselves.
"""
from __future__ import annotations

import time
from typing import Any, Callable

from graph_store import Node

# Tiny generation budget: this lab measures topology/throughput, not output
# quality. Short prompts + small max_tokens keep oMLX from thrashing and keep
# each node's wall-clock dominated by structure, not by token generation.
_MAX_TOKENS = 64


def tool_handler(node: Node) -> dict[str, Any]:
    """A deterministic, LLM-free node: sleep for `payload['sleep_s']` (default
    0.2s) then return zero-token usage. Deterministic by design so durability /
    recovery tests have no model nondeterminism to fight."""
    sleep_s = float(node.payload.get("sleep_s", 0.2))
    start = time.perf_counter()
    time.sleep(sleep_s)
    ms = (time.perf_counter() - start) * 1000.0
    return {"tokens": 0, "ms": ms, "text": ""}


def make_llm_handler(
    client: Any,
    model: str,
    cost_meter: Any | None = None,
) -> Callable[[Node], dict[str, Any]]:
    """Build a handler that calls oMLX for one node and captures REAL usage.

    `client` is a raw `openai.OpenAI` pointed at oMLX :8000 (so we get
    `response.usage`). If `cost_meter` is passed, the call is wrapped in
    `cost_meter.meter(...)` keyed by (run_id, node, attempt) so retries don't
    double-bill. Returns {"tokens": total, "ms": wall, "text": ...}."""

    def handler(node: Node) -> dict[str, Any]:
        prompt = node.payload.get("prompt", "Reply with the single word: ok.")
        node_model = node.payload.get("model", model)

        def _call() -> dict[str, Any]:
            start = time.perf_counter()
            resp = client.chat.completions.create(
                model=node_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=_MAX_TOKENS,
            )
            ms = (time.perf_counter() - start) * 1000.0
            usage = resp.usage
            t_in = getattr(usage, "prompt_tokens", 0) or 0
            t_out = getattr(usage, "completion_tokens", 0) or 0
            text = resp.choices[0].message.content or ""
            return {"_t_in": t_in, "_t_out": t_out, "ms": ms, "text": text}

        if cost_meter is not None:
            with cost_meter.meter(node.run_id, node.name, node.attempts,
                                  node_model) as m:
                out = _call()
                m.record(out["_t_in"], out["_t_out"])
        else:
            out = _call()

        return {
            "tokens": out["_t_in"] + out["_t_out"],
            "ms": out["ms"],
            "text": out["text"],
        }

    return handler
