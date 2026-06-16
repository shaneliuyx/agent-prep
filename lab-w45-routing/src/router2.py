"""2-tier difficulty router — the workable Phase 4 solution.

Phase 4 measured that 3-way tier (haiku/sonnet/opus) plateaus at ~83% because the
sonnet<->opus boundary has only 78% inter-annotator agreement (RESULTS.md). Collapsing
to {haiku, heavy} removes the contested boundary -> 95.65% tier (measured re-score).
This module is the shipped fix: same cheap 4B, same few-shot mechanism, two difficulty
classes. The 3-tier `router.py` is kept as the recorded negative-result history.

Mode (minimal/react/deliberate) is unchanged — the merge is on the difficulty axis only.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from openai import OpenAI

from src.fleet_config import FLEET

Tier2 = Literal["haiku", "heavy"]
Mode = Literal["minimal", "react", "deliberate"]


@dataclass(frozen=True)
class Verdict2:
    tier: Tier2
    mode: Mode
    confidence: float


def merge_tier(t: str) -> Tier2:
    """sonnet + opus collapse to 'heavy' — the boundary humans don't agree on."""
    return "haiku" if t == "haiku" else "heavy"


ROUTER2_PROMPT = """You route LLM queries on two axes: difficulty TIER and control-flow MODE.

Output ONE JSON object on a single line:
  {"tier": "haiku" | "heavy", "mode": "minimal" | "react" | "deliberate", "confidence": 0.0-1.0}

TIER (difficulty):
  haiku — trivial: arithmetic, factual recall, single-fact lookup, short summary/rewrite.
  heavy — anything needing real reasoning: code (debug/refactor/write), concept explanation,
          architecture, multi-step planning, deep cross-file debugging, synthesis.

MODE (control flow):
  minimal    — one LLM call, no tools/scratchpad (factual, arithmetic, one-shot).
  react      — Think-Act-Observe loop with tools (code debug, multi-step compute).
  deliberate — plan-then-execute split (architecture, multi-component planning).

Return ONLY the JSON object. No prose, no markdown fence."""


@lru_cache(maxsize=1)
def _fewshot2() -> tuple[dict, ...]:
    """One exemplar per (tier2, mode) cell from the TRAIN split, tiers merged to 2-way.
    Lazy + cached; no eval leakage (same split as the accuracy test)."""
    from src.probes import load_probes, train_eval_split

    train, _ = train_eval_split(load_probes())
    seen: set[tuple[str, str]] = set()
    msgs: list[dict] = []
    for r in train:
        key = (merge_tier(r["expected_tier"]), r["expected_mode"])
        if key in seen:
            continue
        seen.add(key)
        msgs.append({"role": "user", "content": r["prompt"]})
        msgs.append({"role": "assistant", "content": json.dumps(
            {"tier": key[0], "mode": key[1], "confidence": 0.9}
        )})
    return tuple(msgs)


def classify2(prompt: str, scratchpad: str = "") -> Verdict2:
    """Route to one (tier2, mode) cell via the few-shot 4B. Degrades to (heavy, react, 0.5)
    on failure — bias to the safe heavier tier rather than crash dispatch."""
    ep = FLEET["classifier"]
    client = OpenAI(base_url=ep.base_url, api_key=os.getenv("OMLX_API_KEY"))

    user_msg = prompt
    if scratchpad:
        user_msg = f"{prompt}\n\nRecent scratchpad context:\n{scratchpad[-2000:]}"

    try:
        resp = client.chat.completions.create(
            model=ep.model,
            messages=[
                {"role": "system", "content": ROUTER2_PROMPT},
                *_fewshot2(),
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
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
        if tier in ("haiku", "heavy") and mode in ("minimal", "react", "deliberate"):
            return Verdict2(tier=tier, mode=mode, confidence=conf)
    except Exception:  # noqa: BLE001
        pass
    return Verdict2(tier="heavy", mode="react", confidence=0.5)
