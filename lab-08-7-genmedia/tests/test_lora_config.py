"""Offline tests for LoRA config + diffusion forward math.
No diffusers/torch required for these tests — they test the constants
+ closed-form math only."""
from src.lora_config import STANDARD_LORA_CONFIG, SUBJECT_LORA_CONFIG
from src.train_lora import forward_step_x_t, get_alpha_bar_schedule


def test_standard_lora_config_rank_16():
    """W8.7 §Concept 3 rule: rank 16 is the sweet spot."""
    assert STANDARD_LORA_CONFIG["r"] == 16
    assert STANDARD_LORA_CONFIG["lora_alpha"] == 32  # 2x rank


def test_standard_lora_targets_attention_only():
    """Cross-attention is where text-conditioning meets image features."""
    targets = STANDARD_LORA_CONFIG["target_modules"]
    assert "to_q" in targets
    assert "to_k" in targets
    assert "to_v" in targets
    # Should NOT target FFN
    assert "ffn" not in targets
    assert "mlp" not in targets


def test_subject_lora_lower_rank_to_reduce_forgetting():
    """SUBJECT_LORA uses lower rank to reduce catastrophic forgetting."""
    assert SUBJECT_LORA_CONFIG["r"] < STANDARD_LORA_CONFIG["r"]


def test_alpha_bar_schedule_monotonic_decreasing():
    """$\\bar{\\alpha}_t$ should decrease monotonically from ~1 to ~0."""
    schedule = get_alpha_bar_schedule(n_timesteps=100)
    assert schedule[0] > 0.99
    assert schedule[-1] < 0.5
    for i in range(1, len(schedule)):
        assert schedule[i] <= schedule[i-1]


def test_forward_step_at_t0_returns_x0_approx():
    """At t=0, alpha_bar≈1, so x_t ≈ x_0 (epsilon contribution negligible)."""
    x_0 = 1.0
    eps = 0.5
    alpha_bar_at_t0 = 0.999
    x_t = forward_step_x_t(x_0, t=0, eps=eps, alpha_bar_t=alpha_bar_at_t0)
    assert abs(x_t - x_0) < 0.05


def test_forward_step_at_t_max_returns_eps_approx():
    """At t=T, alpha_bar≈0, so x_t ≈ epsilon (pure noise)."""
    x_0 = 1.0
    eps = 0.5
    alpha_bar_at_T = 0.001
    x_t = forward_step_x_t(x_0, t=999, eps=eps, alpha_bar_t=alpha_bar_at_T)
    # x_t ≈ 0 * x_0 + sqrt(1) * eps = eps
    assert abs(x_t - eps) < 0.1
