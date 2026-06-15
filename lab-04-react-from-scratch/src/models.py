"""
src/models.py — model routing for the ReAct lab.

Loads the oMLX fleet config from the environment, exposes a per-role lookup,
and provides a lazy-load post-loop composer.

Why this lives in its own module:
  src/react.py is the loop. The loop should not know which model is "best at
  reasoning" or "best at tool calling" — that is a fleet-tuning concern. By
  isolating role -> (url, model) mapping here, swapping a tier (e.g. when the
  oMLX engine is upgraded and a model's latency changes) is a one-file edit
  and does not touch the loop logic.

Routing decisions are anchored to the empirical findings in
data/fleet_probe_*.json — see the README before re-tuning ROLE_MAP.

Roles (used by src/react.py and Week 5 pattern zoo):
  loop      : default ReAct loop driver. Hot path; tool calling required.
  tool_arg  : synthesize tool arguments. Same model as loop unless tuned.
  classify  : cheap pre-loop intent triage / observability sidecar.
  reason    : math / multi-step reasoning sub-step.
  compose   : post-loop final answer composition (no tools needed).
  finisher  : lazy-loaded long-form output (post-loop polish).
  hard_loop : in-loop fallback for a larger reasoning attempt that still
              needs tool calling.

Engine: oMLX exposes one OpenAI-compatible endpoint on :8000/v1 and routes
internally by the `model:` field — it IS the gateway (no per-model ports).
Models are addressed by bare served id (GET /v1/models), no `models/` prefix.

Probe-driven mapping as of 2026-06-15 (oMLX :8000, M5 Pro 48 GB) — see
data/fleet_probe_20260615_omlx.json for raw scores:
  loop, tool_arg, reason, compose, finisher -> Gemma-26B
      (the ONLY model scoring 1.00 across tool+json+reason+instr)
  classify -> Qwen2.5-Coder-7B (fast 4 GB, ~241 ms, reason=1.00) — cheap
      non-tool triage; format is loose and tools need the client parser,
      neither of which matters for plain classification
  hard_loop -> Qwen3.5-27B-Claude-Distilled
      (the only OTHER loadable tool-capable model; larger reasoning attempt)

Why one workhorse instead of a fleet of specialists (measured, not aspired):
  - Gemma-26B (sonnet): tool=json=reason=instr=1.00, ping ~353 ms. Reliable.
  - 35B-A3B / 27B (reasoning-distilled): tool=1.00 BUT json/reason/instr≈0 —
    they emit <think> blocks that break strict-format output. Safe only for
    raw tool-call emission, not for formatted text. Kept as MODEL_HAIKU (fast
    tool-only, ping ~149 ms) but NOT on any format-sensitive role.
  - Gemma-31B-uncensored-heretic: cannot load — oMLX memory_guard rejects it
    (projected 47.6 GB > 37.44 GB ceiling) once other models are warm. The
    48 GB box holds one heavy model hot at a time.

NOTE on tool calling: oMLX parses structured tool_calls per MODEL FAMILY
(Gemma / Qwen3 / gpt-oss yes; Qwen2.5-Coder NO — it leaks <tools>/<function>
text on both the OpenAI and Anthropic surfaces). If you route a non-parsed
family here, src/react.py must apply a client-side text fallback parser (see
scripts/probe_fleet.py::extract_text_tool_calls) before reading tool_calls.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from openai import OpenAI

# ---------------------------------------------------------------------------
# Fleet endpoint default. Override via env when running tier experiments.
# ---------------------------------------------------------------------------
# oMLX OpenAI-compatible surface: all models behind one endpoint, selected by
# the `model:` field (no per-model ports — oMLX routes internally).
# Override with OMLX_URL to point at a different host (e.g. LAN).
_OMLX_URL = os.getenv("OMLX_URL", "http://127.0.0.1:8000/v1")

# Bare served ids (GET /v1/models) — no `models/` prefix on oMLX.
_SONNET_MODEL = os.getenv("MODEL_SONNET", "gemma-4-26B-A4B-it-heretic-4bit")
_HAIKU_MODEL = os.getenv("MODEL_HAIKU", "MLX-Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled-4bit")
_OPUS_MODEL = os.getenv("MODEL_OPUS", "Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit")
_FAST_MODEL = os.getenv("MODEL_FAST", "Qwen2.5-Coder-7B-Instruct-MLX-4bit")

API_KEY = os.getenv("OMLX_API_KEY", "not-needed")  # oMLX ignores; SDK requires non-empty

# Probed tiers (env-overridable). Used by §5 + Week-5 tier-comparison
# experiments. `haiku` is the fast tool-only option (ping ~149 ms, tool=1.00)
# deliberately kept OUT of ROLE_MAP: its <think> blocks fail format-sensitive
# roles, so it is opt-in for raw tool-call emission only.
TIER_MODELS: dict[str, str] = {
    "sonnet": _SONNET_MODEL,
    "haiku": _HAIKU_MODEL,
    "opus": _OPUS_MODEL,
    "fast": _FAST_MODEL,
}


Role = Literal["loop", "tool_arg", "classify", "reason", "compose",
               "finisher", "hard_loop"]


@dataclass(frozen=True)
class Endpoint:
    url: str
    model: str
    timeout_s: float


ROLE_MAP: dict[str, Endpoint] = {
    # Gemma-26B is the measured workhorse: the only model with tool, json,
    # reason, AND instr all at 1.00 (2026-06-15 probe). Every reliable role
    # routes here. The reasoning-distilled models pass tool but fail strict
    # format, so they cannot safely own compose/classify/reason.
    "loop":      Endpoint(_OMLX_URL, _SONNET_MODEL, timeout_s=30),
    "tool_arg":  Endpoint(_OMLX_URL, _SONNET_MODEL, timeout_s=30),
    # classify is cheap, high-frequency, non-tool triage → the fast 7B tier
    # (4 GB, ~241 ms, reason=1.00). It does NOT call tools, so the Qwen2.5
    # no-server-parser limitation is irrelevant here; if you ever route a
    # tool-bearing role to _FAST_MODEL, the caller MUST apply the client-side
    # text parser (scripts/probe_fleet.py::extract_text_tool_calls) first.
    "classify":  Endpoint(_OMLX_URL, _FAST_MODEL, timeout_s=15),
    "reason":    Endpoint(_OMLX_URL, _SONNET_MODEL, timeout_s=45),
    "compose":   Endpoint(_OMLX_URL, _SONNET_MODEL, timeout_s=45),
    "finisher":  Endpoint(_OMLX_URL, _SONNET_MODEL, timeout_s=60),
    # Larger reasoning attempt that still needs tool calling. Qwen3.5-27B is
    # the only OTHER loadable tool-capable model (tool=1.00). Its format
    # scores are poor (think-blocks), so use it for tool-driven reasoning,
    # not for formatted output. The 31B heretic that would have served this
    # role OOMs on the 37.44 GB oMLX memory ceiling.
    "hard_loop": Endpoint(_OMLX_URL, _OPUS_MODEL, timeout_s=90),
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
    """Spin the `finisher` model for high-quality post-loop polishing.

    Routes via `ROLE_MAP["finisher"]` — currently Gemma-26B (the reliable
    workhorse). Tool calling is intentionally NOT requested — this is plain-
    text-in, plain-text-out.

    Returns the polished answer, or `raw_answer` unchanged on any error
    (broad except is deliberate: optional polish must never block the
    agent's return path).
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


__all__ = ["Role", "Endpoint", "ROLE_MAP", "TIER_MODELS", "get_client",
           "compose_final_answer", "API_KEY"]
