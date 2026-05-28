# tests/test_handoffs.py
from handoffs import swarm_run

def test_refund_intent_routes_to_refund():
    """Refund-flavored message should hand off to refund agent."""
    out = swarm_run("I want a refund please.")
    assert "refund" in out["handoff_trace"]

def test_sales_intent_routes_to_sales():
    """Plan-question message should hand off to sales agent."""
    out = swarm_run("What plans do you offer?")
    assert "sales" in out["handoff_trace"]

def test_handoff_count_bounded():
    """No path should require more than 2 handoffs (triage → specialist)."""
    out = swarm_run("Tell me about your enterprise plan.")
    assert out["handoff_count"] <= 2

def test_trace_starts_with_triage():
    """Every conversation starts at triage."""
    out = swarm_run("Anything works")
    assert out["handoff_trace"][0] == "triage"
