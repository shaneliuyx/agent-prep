"""Offline tests for the tracing primitive — no Langfuse, no live LLM.

Validates the cost formula + rate-card normalization without network.
"""
from src.tracing import compute_cost, RATE_CARD, _normalize_model


def test_normalize_strips_models_prefix():
    """Gateway-style `models/X` should normalize to `X` for RATE_CARD lookup."""
    assert _normalize_model("models/gpt-oss-20b-MXFP4-Q8") == "gpt-oss-20b-MXFP4-Q8"
    assert _normalize_model("gpt-oss-20b-MXFP4-Q8") == "gpt-oss-20b-MXFP4-Q8"


def test_compute_cost_matches_formula():
    """$C = t_{in} \\cdot p_{in} + t_{out} \\cdot p_{out}$ per 1M-token rates."""
    # gpt-oss-20b: in=0.50, out=2.00 per 1M
    cost = compute_cost("gpt-oss-20b-MXFP4-Q8", tokens_in=1_000_000, tokens_out=1_000_000)
    assert abs(cost - (0.50 + 2.00)) < 1e-6


def test_compute_cost_handles_gateway_prefix():
    """Should match the same model with or without `models/` prefix."""
    a = compute_cost("gpt-oss-20b-MXFP4-Q8", 100, 200)
    b = compute_cost("models/gpt-oss-20b-MXFP4-Q8", 100, 200)
    assert abs(a - b) < 1e-9


def test_compute_cost_unknown_model_returns_zero():
    """Defensive — production rule: missing rate card should NOT crash."""
    assert compute_cost("nonexistent-model", 1000, 1000) == 0.0


def test_rate_card_has_5_models():
    """Lab fleet coverage — should match W4 §1.5 role map + W7.7 quant fleet."""
    assert len(RATE_CARD) == 5
