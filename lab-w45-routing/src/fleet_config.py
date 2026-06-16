"""W4.5 fleet config — extends W4's oMLX setup with a classifier tier.

W4 runs ONE oMLX endpoint on :8000, model-routed by the `model:` field
(no per-tier ports). W4.5 keeps that design: every tier here shares the
same base_url (:8000) and is distinguished ONLY by its model id.

Two design choices worth flagging:
1. The classifier is a SEPARATE model id from the executor tiers. The
   router never picks the classifier as an executor — invariant: the
   classifier ALWAYS runs first, then dispatches to one of the 3 executor
   tiers. A distinct model id (not a distinct port) encodes that invariant.
2. Model ids are kept here (single source of truth) and consumed by both
   router.py and tier_dispatch.py. Hand-edit the model ids here when the
   oMLX catalog changes (`curl :8000/v1/models`); do NOT scatter id strings
   across the codebase. NOTE: one heavy model is hot at a time — oMLX
   cold-loads (~10-30 s) on a tier switch and 507s if two heavy models
   would co-resident on a 48 GB box.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class FleetEndpoint:
    name: str
    tier: str  # "classifier" | "haiku" | "sonnet" | "opus"
    model: str  # oMLX model id (see `curl :8000/v1/models`)
    base_url: str  # full OpenAI-compatible base URL


FLEET: dict[str, FleetEndpoint] = {
    "classifier": FleetEndpoint(
        name="classifier",
        tier="classifier",
        model="Qwen3.5-4B-MLX-4bit",                  # W4's `fast` role: 4 GB, structured tools, ~235 ms
        base_url="http://127.0.0.1:8000/v1",
    ),
    "haiku": FleetEndpoint(
        name="haiku",
        tier="haiku",
        model="MLX-Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled-4bit",  # W4 `haiku`: ~152 ms, tool=1.00
        base_url="http://127.0.0.1:8000/v1",
    ),
    "sonnet": FleetEndpoint(
        name="sonnet",
        tier="sonnet",
        model="gemma-4-26B-A4B-it-heretic-4bit",      # W4 workhorse: only model 1.00 on tool+json+reason+instr
        base_url="http://127.0.0.1:8000/v1",
    ),
    "opus": FleetEndpoint(
        name="opus",
        tier="opus",
        model="Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit",  # W4 `opus_qwen`: larger reasoning, format-terse
        base_url="http://127.0.0.1:8000/v1",
    ),
}