"""Offline unit tests for quality_gate scoring signals.

These tests do NOT require guild or EverCore — they exercise the pure
scoring functions (length, specificity, lexical-style novelty fallback)
with `tm=None`. Integration with a live EverCore search is covered in
the §3.3 measurement run, not here.
"""
from src.quality_gate import (
    DEFAULT_THRESHOLD,
    DEFAULT_WEIGHTS,
    _length_score,
    _specificity_score,
    quality_score,
    should_promote,
)


def test_length_score_peaks_around_15_words():
    fifteen = " ".join(["word"] * 15)
    five = " ".join(["word"] * 5)
    fifty = " ".join(["word"] * 50)
    assert _length_score(fifteen) == 1.0
    assert _length_score(five) < _length_score(fifteen)
    assert _length_score(fifty) == 0.2


def test_length_score_zero_below_5_words():
    assert _length_score("too short here") == 0.0


def test_specificity_score_counts_numbers_and_proper_nouns():
    high = "Terraform deploys APIs to AWS at 200ms p99 latency"
    low = "we did stuff and it worked"
    assert _specificity_score(high) > 0.5
    assert _specificity_score(low) < 0.3


def test_specificity_saturates_at_three_hits():
    many = "200ms 30min 5GB Terraform AWS Kubernetes 99% p50"
    assert _specificity_score(many) == 1.0


def test_quality_score_high_for_concrete_factual_summary():
    summary = "Production API deployments use Terraform IaC with VPC peering and 5-min apply budget."
    score = quality_score(summary, tm=None)
    assert score > DEFAULT_THRESHOLD


def test_quality_score_low_for_vague_summary():
    summary = "tried things and stuff worked maybe"
    score = quality_score(summary, tm=None)
    assert score < DEFAULT_THRESHOLD


def test_should_promote_threshold_dial():
    summary = "API uses Terraform IaC pattern for deploys"
    # Default threshold accepts this concrete summary.
    assert should_promote(summary, threshold=0.3, tm=None) is True
    # Aggressive threshold rejects it (novelty defaults to 1.0 offline,
    # length+specificity weighted average still below 0.9).
    assert should_promote(summary, threshold=0.95, tm=None) is False


def test_weights_can_be_overridden():
    summary = " ".join(["word"] * 15)  # length=1.0, specificity=0.0, novelty=1.0
    # Default weights → 0.3*1 + 0.4*0 + 0.3*1 = 0.6
    assert abs(quality_score(summary, tm=None) - 0.6) < 0.01
    # Push all weight onto specificity → 0.0
    only_spec = {"length": 0.0, "specificity": 1.0, "novelty": 0.0}
    assert quality_score(summary, tm=None, weights=only_spec) == 0.0


def test_default_weights_sum_to_one():
    assert abs(sum(DEFAULT_WEIGHTS.values()) - 1.0) < 1e-9
