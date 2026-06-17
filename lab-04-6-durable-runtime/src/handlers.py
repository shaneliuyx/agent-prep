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

WHY token counts are real:
this lab's headline number is *real* token counts summed across a topology. We
get them from `shared/llm.chat_usage`, which returns `(text, usage)` in one call
(usage = prompt/completion/total tokens off `response.usage`) — so the handler
never has to hand-roll a raw client just to keep the usage the cost story needs.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any, Callable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "shared"))

from graph_store import Node  # noqa: E402
from llm import chat_usage  # noqa: E402

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

    `client` is an `openai.OpenAI` pointed at oMLX :8000; the call goes through
    `shared/llm.chat_usage`, which returns `(text, usage)` so we get real token
    counts without re-reading `response.usage` by hand. If `cost_meter` is passed,
    the call is wrapped in `cost_meter.meter(...)` keyed by (run_id, node, attempt)
    so a crash-replayed (node, attempt) isn't re-billed; genuine retries get a fresh
    attempt and bill once each. Returns {"tokens": total, "ms": wall, "text": ...}."""

    def handler(node: Node) -> dict[str, Any]:
        prompt = node.payload.get("prompt", "Reply with the single word: ok.")
        node_model = node.payload.get("model", model)

        def _call() -> dict[str, Any]:
            start = time.perf_counter()
            text, usage = chat_usage(client, prompt, node_model,
                                     temperature=0.0, max_tokens=_MAX_TOKENS)
            ms = (time.perf_counter() - start) * 1000.0
            return {
                "_t_in": usage["prompt_tokens"],
                "_t_out": usage["completion_tokens"],
                "ms": ms,
                "text": text,
            }

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
