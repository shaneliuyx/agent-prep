"""
src/models.py — model routing for the ReAct lab.

Loads the vMLX fleet config from the environment, exposes a per-role lookup,
and provides a lazy-load post-loop composer.

Why this lives in its own module:
  src/react.py is the loop. The loop should not know which model is "best at
  reasoning" or "best at tool calling" — that is a fleet-tuning concern. By
  isolating role -> (url, model) mapping here, swapping a tier (e.g. when the
  vMLX engine is upgraded and Sonnet's latency changes) is a one-file edit
  and does not touch the loop logic.

Routing decisions are anchored to the empirical findings in
data/fleet_probe_*.json — see the README before re-tuning ROLE_MAP.

Roles (used by src/react.py and Week 5 pattern zoo):
  loop      : default ReAct loop driver. Hot path; tool calling required.
  tool_arg  : synthesize tool arguments. Same model as loop unless tuned.
  classify  : cheap pre-loop intent triage / observability sidecar.
  reason    : math / multi-step reasoning sub-step.
  compose   : post-loop final answer composition (no tools needed).
  finisher  : lazy-loaded long-form / uncensored output (post-loop only).

Probe-driven mapping as of 2026-05-04 (vMLX engine v?, M5 Pro 48 GB):
  loop, tool_arg, classify  -> Distill 9B  (only model with 1.00 across 5 runs)
  reason, compose           -> Gemma-26B   (reason+instr 1.00 when warm)
  finisher                  -> gemma-31B-heretic (lazy; tool=0.00 so post-loop only)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from openai import OpenAI

# ---------------------------------------------------------------------------
# Fleet endpoint defaults. Override via env when running tier experiments.
# ---------------------------------------------------------------------------
_HAIKU_URL = os.getenv("VMLX_URL_HAIKU", "http://127.0.0.1:8004/v1")
_SONNET_URL = os.getenv("VMLX_URL_SONNET", "http://127.0.0.1:8003/v1")
_OPUS_LAZY_URL = os.getenv("VMLX_URL_OPUS_LAZY", "http://127.0.0.1:8000/v1")

_HAIKU_MODEL = os.getenv("MODEL_HAIKU", "MLX-Qwen3.5-9B-GLM5.1-Distill-v1-8bit")
_SONNET_MODEL = os.getenv("MODEL_SONNET", "gemma-4-26B-A4B-it-heretic-4bit")
_OPUS_LAZY_MODEL = os.getenv("MODEL_OPUS_LAZY", "gemma-4-31B-uncensored-heretic-mlx-4bit")

API_KEY = os.getenv("VMLX_API_KEY", "not-needed")  # vMLX ignores; SDK requires non-empty


Role = Literal["loop", "tool_arg", "classify", "reason", "compose", "finisher"]


@dataclass(frozen=True)
class Endpoint:
    url: str
    model: str
    timeout_s: float


ROLE_MAP: dict[str, Endpoint] = {
    "loop":     Endpoint(_HAIKU_URL,     _HAIKU_MODEL,     timeout_s=30),
    "tool_arg": Endpoint(_HAIKU_URL,     _HAIKU_MODEL,     timeout_s=30),
    "classify": Endpoint(_HAIKU_URL,     _HAIKU_MODEL,     timeout_s=10),
    "reason":   Endpoint(_SONNET_URL,    _SONNET_MODEL,    timeout_s=45),
    "compose":  Endpoint(_SONNET_URL,    _SONNET_MODEL,    timeout_s=45),
    "finisher": Endpoint(_OPUS_LAZY_URL, _OPUS_LAZY_MODEL, timeout_s=90),
}


# Cache one client per (url, timeout) — OpenAI() construction is non-trivial.
_CLIENT_CACHE: dict[tuple[str, float], OpenAI] = {}


def get_client(role: Role) -> tuple[OpenAI, str]:
    """Return (client, model_name) for a role. Reuses cached clients."""
    ep = ROLE_MAP[role]
    key = (ep.url, ep.timeout_s)
    client = _CLIENT_CACHE.get(key)
    if client is None:
        client = OpenAI(
            base_url=ep.url,
            api_key=API_KEY,
            timeout=ep.timeout_s,
            max_retries=0,
        )
        _CLIENT_CACHE[key] = client
    return client, ep.model


def compose_final_answer(raw_answer: str, user_query: str,
                         system: str | None = None) -> str:
    """Lazy-spin gemma-31B-heretic for high-quality post-loop polishing.

    Only invoke after `agent_run()` returns. Cold-start tax ~5-15 s on the
    first call; subsequent calls are ~1.7 s median. Tool calling is
    intentionally NOT requested — this model scored 0.00 on the tool probe
    and would emit plain text instead of structured calls.

    Returns the polished answer, or `raw_answer` unchanged on any error.
    """
    client, model = get_client("finisher")
    sys_prompt = system or (
        "Polish the agent's draft answer for a human reader. Keep all factual "
        "content. Improve clarity, fix grammar, drop scratchpad noise."
    )
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"Query: {user_query}\nDraft: {raw_answer}"},
            ],
            temperature=0.3,
            max_tokens=2048,
        )
        return r.choices[0].message.content or raw_answer
    except Exception:  # broad: lazy-loader is opt-in polish, never block return
        return raw_answer


__all__ = ["Role", "Endpoint", "ROLE_MAP", "get_client",
           "compose_final_answer", "API_KEY"]
