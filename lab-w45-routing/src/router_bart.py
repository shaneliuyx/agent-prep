"""Second classifier — HuggingFace zero-shot BART-MNLI.

Runs locally on CPU/MPS via transformers pipeline. Maps each (tier, mode)
to a candidate label (e.g. "needs a small fast model for a one-shot answer")
and uses BART-MNLI's entailment scores to pick the highest-probability cell.

The two classifiers run in parallel via asyncio.gather — BART's CPU
inference (~150-300ms) overlaps with Qwen's GPU inference (~100ms idle).
"""
from __future__ import annotations

import functools
from src.router import RouterVerdict


# Map each (tier, mode) cell to a natural-language hypothesis for BART-MNLI.
LABELS = {
    ("haiku", "minimal"):    "This prompt needs a small fast model for a one-shot factual or arithmetic answer.",
    ("haiku", "react"):      "This prompt needs a small model running a Think-Act-Observe loop for simple multi-step reasoning.",
    ("haiku", "deliberate"): "This prompt needs a small model with explicit planning for a structured short task.",
    ("sonnet", "minimal"):   "This prompt needs a medium model for a single-shot code or concept explanation.",
    ("sonnet", "react"):     "This prompt needs a medium model with a Think-Act-Observe loop for code debugging or multi-step reasoning.",
    ("sonnet", "deliberate"): "This prompt needs a medium model with explicit planning for a moderately complex task.",
    ("opus", "minimal"):     "This prompt needs a large model for a single-shot deep reasoning answer.",
    ("opus", "react"):       "This prompt needs a large model with a Think-Act-Observe loop for deep multi-step reasoning.",
    ("opus", "deliberate"):  "This prompt needs a large model with explicit planning for a complex architectural or design task.",
}


@functools.lru_cache(maxsize=1)
def _bart_pipeline():
    """Lazy import + load BART-MNLI once per process. ~3s warm-up."""
    from transformers import pipeline
    return pipeline(
        task="zero-shot-classification",
        model="facebook/bart-large-mnli",
        device="cpu",  # MPS works too; CPU is more portable for the lab
    )


def classify_bart(prompt: str) -> RouterVerdict:
    """Pick the highest-entailment (tier, mode) cell via BART-MNLI."""
    candidate_labels = list(LABELS.values())
    out = _bart_pipeline()(prompt[:1000], candidate_labels)
    top_label = out["labels"][0]
    top_score = out["scores"][0]
    # Reverse-map top label → (tier, mode)
    for key, label in LABELS.items():
        if label == top_label:
            return RouterVerdict(tier=key[0], mode=key[1], confidence=float(top_score))
    # Fallback if reverse-lookup fails (shouldn't happen)
    return RouterVerdict(tier="sonnet", mode="react", confidence=0.5)