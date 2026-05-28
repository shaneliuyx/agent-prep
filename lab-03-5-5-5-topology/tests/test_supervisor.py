# tests/test_supervisor.py
import json
import os
import pytest
from supervisor import supervisor_run

@pytest.mark.integration
def test_supervisor_parallel_wins():
    """total ≈ plan + max(workers) + synth, NOT plan + sum(workers) + synth.
    Requires real LLM latency; mock returns instantly so wall-times are 0.
    Conftest's autouse fixture leaves LLM_PROVIDER alone for tests with the
    `integration` marker; the user's exported provider (openai / anthropic-
    proxy) takes effect here."""
    if os.getenv("LLM_PROVIDER", "mock") == "mock":
        pytest.skip("set LLM_PROVIDER=openai or anthropic-proxy to run")
    out = supervisor_run("What is photosynthesis?")
    parallel = out["plan_wall_s"] + out["max_worker_wall_s"] + out["synthesize_wall_s"]
    sequential = out["plan_wall_s"] + out["sum_worker_walls_s"] + out["synthesize_wall_s"]
    assert out["total_wall_s"] < (parallel + sequential) / 2

def test_supervisor_decomposition_is_three():
    """plan_decompose returns exactly 3 sub-questions per contract."""
    assert len(supervisor_run("Compare REST vs GraphQL.")["worker_walls_s"]) == 3

def test_supervisor_synthesizes_answer():
    """End-to-end: answer is a non-trivially long string."""
    out = supervisor_run("What is OAuth?")
    assert isinstance(out["answer"], str) and len(out["answer"]) > 50
