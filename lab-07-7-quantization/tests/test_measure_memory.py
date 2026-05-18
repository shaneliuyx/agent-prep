"""Smoke tests for Phase 1 — verify the math + table rendering work
without requiring an actual model load."""
from src.measure_memory import MODELS, render_table


def test_models_have_4_entries():
    """Fleet config matches W7.7 chapter §1.1."""
    assert len(MODELS) == 4


def test_theoretical_math_matches_chapter():
    """$M_{\\text{weights}} = N_{\\text{params}} \\times \\text{bytes per param}$ holds for known cases."""
    # gpt-oss-20b-MXFP4-Q8 -> 20e9 * 0.5 = 10 GB
    for model_id, n_params, bytes_per_param in MODELS:
        theoretical_gb = n_params * bytes_per_param / 1e9
        if "20b" in model_id.lower():
            assert abs(theoretical_gb - 10.0) < 0.1
        elif "9b" in model_id.lower():
            assert abs(theoretical_gb - 9.0) < 0.1


def test_render_table_handles_missing_disk():
    """Disk gb can be None when model not in HF cache."""
    rows = [{
        "model": "test",
        "params_b": 1.0,
        "bytes_per_param": 2.0,
        "theoretical_gb": 2.0,
        "disk_gb": None,
        "rss_delta_gb": 1.5,
    }]
    table = render_table(rows)
    assert "?" in table  # placeholder for missing disk
    assert "test" in table
