"""Single-classifier router. Qwen3.5-4B on :8000 emits JSON
{tier, mode, confidence} via a strict system prompt + few-shot exemplars.
JSON-parse with graceful fallback to (sonnet, react, 0.5) on malformed
output — never crash the pipeline on a classifier hiccup; degrade to a
safe middle tier.

The few-shot exemplars (one per (tier, mode) cell, drawn from the train
split) are the load-bearing accuracy lever: they lift this 4B classifier
from ~61% to ~83% tier accuracy. A same-model vote on top of them did NOT
help (see RESULTS.md / chapter Phase 3) — independent errors are what make
a vote pay off, and two prompts on one model share its blind spots. So the
shipped router is deliberately the simple few-shot classifier.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from openai import OpenAI

from src.fleet_config import FLEET


Tier = Literal["haiku", "sonnet", "opus"]
Mode = Literal["minimal", "react", "deliberate"]


@dataclass(frozen=True)
class RouterVerdict:
    tier: Tier
    mode: Mode
    confidence: float  # 0.0-1.0, self-reported by classifier


ROUTER_PROMPT = """You route LLM queries to the right model+control-flow.

Output ONE JSON object on a single line:
  {"tier": "haiku" | "sonnet" | "opus",
   "mode": "minimal" | "react" | "deliberate",
   "confidence": 0.0-1.0}

TIER:
  haiku    — fast MoE (35B-A3B, 3B active), ~152ms idle. Use for: arithmetic,
             factual recall, simple summarisation, single-fact lookup, short rewrites.
  sonnet   — workhorse (Gemma-26B), ~317ms idle. Use for: code debug/refactor (single
             file), concept explanation, structured-output generation, light planning.
  opus     — large (27B reasoning-distill), ~726ms idle. Use for: multi-step
             architecture, deep explanation requiring synthesis, multi-component
             planning, ambiguous-spec reasoning.

MODE:
  minimal    — single LLM call, no tool use, no scratchpad. Use for: factual,
               arithmetic, summarisation, one-shot rewrites.
  react      — ReAct loop with tool calls and scratchpad. Use for: code debug,
               multi-step math, anything needing observation-action cycles.
  deliberate — plan-then-execute split (plan with one call, execute with another).
               Use for: architecture, multi-component planning, deep explanation.

CONFIDENCE: your self-assessed certainty. Drop below 0.7 if the prompt is
ambiguous; downstream will escalate or trigger a vote when conf < 0.7.

Return ONLY the JSON object. No prose, no markdown fence, no preamble."""


@lru_cache(maxsize=1)
def _fewshot_messages() -> tuple[dict, ...]:
    """One labelled exemplar per (tier, mode) cell, as user/assistant turns.

    Drawn from the TRAIN split only — same default seed/frac as the accuracy
    test's `train_eval_split(rows)`, so the partition is identical and the eval
    rows the test scores on are NEVER shown to the classifier (no leakage).
    Imported lazily to keep module import side-effect-free; cached so the probe
    file is read and split once per process, not on every classify() call.
    """
    from src.probes import load_probes, train_eval_split

    train, _ = train_eval_split(load_probes())
    seen: set[tuple[str, str]] = set()
    msgs: list[dict] = []
    for r in train:
        key = (r["expected_tier"], r["expected_mode"])
        if key in seen:
            continue
        seen.add(key)
        msgs.append({"role": "user", "content": r["prompt"]})
        msgs.append({"role": "assistant", "content": json.dumps(
            {"tier": r["expected_tier"], "mode": r["expected_mode"], "confidence": 0.9}
        )})
    return tuple(msgs)


def _classify_once(
    client: OpenAI, model: str, system_prompt: str, user_msg: str, temperature: float
) -> RouterVerdict | None:
    """One classifier call + parse. Returns a valid verdict, or None on any
    failure (caller decides how to degrade — never crash on a classifier hiccup)."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                *_fewshot_messages(),  # one labelled exemplar per (tier, mode) cell
                {"role": "user", "content": user_msg},
            ],
            temperature=temperature,
            max_tokens=120,
        )
        raw = (resp.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        parsed = json.loads(raw)
        tier = parsed.get("tier")
        mode = parsed.get("mode")
        conf = float(parsed.get("confidence", 0.5))
        if tier in ("haiku", "sonnet", "opus") and mode in ("minimal", "react", "deliberate"):
            return RouterVerdict(tier=tier, mode=mode, confidence=conf)
    except Exception:  # noqa: BLE001
        pass
    return None


def classify(prompt: str, scratchpad: str = "") -> RouterVerdict:
    """Route the prompt to one (tier, mode) cell via the few-shot classifier.

    A single deterministic call (temperature 0) against the 4B classifier tier.
    Degrades to (sonnet, react, 0.5) on any classifier failure — a safe middle
    tier is better than crashing the dispatch pipeline.
    """
    ep = FLEET["classifier"]
    client = OpenAI(base_url=ep.base_url, api_key=os.getenv("OMLX_API_KEY"))

    user_msg = prompt
    if scratchpad:
        user_msg = f"{prompt}\n\nRecent scratchpad context:\n{scratchpad[-2000:]}"

    verdict = _classify_once(client, ep.model, ROUTER_PROMPT, user_msg, 0.0)
    if verdict is not None:
        return verdict
    return RouterVerdict(tier="sonnet", mode="react", confidence=0.5)
