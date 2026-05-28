# tests/test_hierarchical.py
import os
import pytest
from hierarchical import hierarchical_run

def test_hierarchy_depth_and_agent_count():
    """2-macro × 2-sub-q hierarchy → 7 agents (1 top + 2 sub + 4 leaves)."""
    out = hierarchical_run("Compare HTTP/2 vs HTTP/3 vs HTTP/3 over QUIC.")
    assert out["depth"] == 2 and out["agents_total"] == 7

@pytest.mark.integration
def test_hierarchy_parallel_at_sub_level():
    """Sub-leads run in parallel: total ≈ plan + max(sub) + synth.
    Requires real LLM latency. Marked `integration` so conftest's autouse
    fixture preserves the user's exported LLM_PROVIDER."""
    if os.getenv("LLM_PROVIDER", "mock") == "mock":
        pytest.skip("set LLM_PROVIDER=openai or anthropic-proxy to run")
    out = hierarchical_run("OAuth 2.0 vs OAuth 2.1 differences?")
    parallel = out["plan_wall_s"] + out["max_sub_wall_s"] + out["synthesize_wall_s"]
    sequential = out["plan_wall_s"] + sum(out["sub_walls_s"]) + out["synthesize_wall_s"]
    assert out["total_wall_s"] < (parallel + sequential) / 2
