"""Offline tests for PSI + retrain trigger."""
import numpy as np
from src.drift import psi, should_trigger_retrain


def test_psi_identical_distributions_near_zero():
    """PSI(P, P) ≈ 0 — same distribution should yield ~0."""
    rng = np.random.default_rng(42)
    p = rng.normal(0, 1, 1000)
    assert psi(p, p) < 0.01


def test_psi_shifted_distribution_above_threshold():
    """Shift mean significantly -> PSI > 0.25 (significant drift)."""
    rng = np.random.default_rng(42)
    ref = rng.normal(0, 1, 1000)
    cur = rng.normal(2.0, 1, 1000)  # shifted by 2 sigma
    assert psi(ref, cur) > 0.25


def test_psi_moderate_shift_in_moderate_zone():
    """Modest shift should land in moderate zone (0.1–0.25)."""
    rng = np.random.default_rng(7)
    ref = rng.normal(0, 1, 1000)
    cur = rng.normal(0.5, 1, 1000)  # half-sigma shift
    p = psi(ref, cur)
    assert 0.05 < p < 0.5  # wide band — distribution-dependent


def test_should_trigger_retrain_3_consecutive():
    """Fire only on 3 consecutive significant readings."""
    history = [{"verdict": "stable"}] * 5 + [{"verdict": "significant"}] * 3
    assert should_trigger_retrain(history) is True


def test_should_trigger_retrain_not_consecutive():
    """Don't fire if last 3 aren't all significant."""
    history = [{"verdict": "significant"}, {"verdict": "stable"},
               {"verdict": "significant"}]
    assert should_trigger_retrain(history) is False


def test_should_trigger_retrain_insufficient_history():
    """Need at least N days of data."""
    history = [{"verdict": "significant"}]
    assert should_trigger_retrain(history, days_consecutive=3) is False
