"""Vote-layer measurement: voted accuracy >= single-classifier accuracy on
the eval split, and disagreement rate is observable + bounded.
"""
import asyncio

import pytest

from src.probes import load_probes, train_eval_split
from src.router import classify
from src.router_vote import router_vote


@pytest.mark.integration
@pytest.mark.xfail(
    reason="BART-MNLI vote regresses vs single classifier (RESULTS.md Phase 4) — an "
    "independent but incompetent second voter. Kept as the measured negative result.",
    strict=False,
)
def test_voted_accuracy_at_least_matches_single():
    rows = load_probes()
    _, eval_ = train_eval_split(rows)

    single_correct = sum(1 for r in eval_ if classify(r["prompt"]).tier == r["expected_tier"])
    voted_correct = sum(
        1 for r in eval_
        if asyncio.run(router_vote(r["prompt"])).tier == r["expected_tier"]
    )
    assert voted_correct >= single_correct, (
        f"voted ({voted_correct}/{len(eval_)}) < single ({single_correct}/{len(eval_)}) "
        "— vote layer regressed accuracy; check tie-break logic"
    )


@pytest.mark.integration
@pytest.mark.xfail(
    reason="BART-MNLI disagreement was 83% > 50% bound (RESULTS.md Phase 4). Kept as the "
    "measured signal that a topic model is the wrong second voter.",
    strict=False,
)
def test_disagreement_rate_observable_and_bounded():
    rows = load_probes()
    _, eval_ = train_eval_split(rows)
    disagreements = 0
    for r in eval_:
        qwen_v = classify(r["prompt"])
        voted = asyncio.run(router_vote(r["prompt"]))
        if voted != qwen_v:
            disagreements += 1
    rate = disagreements / len(eval_)
    assert rate <= 0.50, f"disagreement rate {rate:.0%} > 50% — taxonomy is too fuzzy"