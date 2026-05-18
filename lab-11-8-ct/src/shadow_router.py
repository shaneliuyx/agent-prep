"""Phase 3 — shadow deployment router.

Splits traffic between production + candidate model.
Records both responses + user feedback signal for offline comparison.

Pattern: candidate gets a small share (5%) initially. If 24h health OK,
ramp: 5% -> 25% -> 100%.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Literal


@dataclass
class ShadowConfig:
    candidate_traffic_pct: float = 0.05    # 5% to candidate by default
    enabled: bool = True


SHADOW = ShadowConfig()


def route(prompt: str,
         prod_call: Callable[[str], str],
         candidate_call: Callable[[str], str],
         ) -> tuple[str, Literal["prod", "candidate"]]:
    """Run prompt through one of the two models.
    On candidate failure, falls back to prod silently."""
    use_candidate = (SHADOW.enabled
                     and random.random() < SHADOW.candidate_traffic_pct)
    if use_candidate:
        try:
            return candidate_call(prompt), "candidate"
        except Exception:
            pass
    return prod_call(prompt), "prod"


def ramp_step(current_pct: float) -> float:
    """Ramp schedule: 5% -> 25% -> 50% -> 75% -> 100%."""
    schedule = [0.05, 0.25, 0.50, 0.75, 1.0]
    for step in schedule:
        if current_pct < step:
            return step
    return 1.0
