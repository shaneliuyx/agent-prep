"""Tests for token_accounting — sticky-estimated flag is the load-bearing
behavior to lock in."""
from agent_loop_tools import TokenAccounting, UsageReport


def test_initial_state_is_zero_and_exact():
    a = TokenAccounting()
    assert a.total_input_tokens == 0
    assert a.total_output_tokens == 0
    assert a.tokens_estimated is False
    assert a.iteration_count == 0


def test_add_exact_usage_keeps_estimated_false():
    a = TokenAccounting()
    a.add(UsageReport(input_tokens=100, output_tokens=50, estimated=False))
    a.add(UsageReport(input_tokens=200, output_tokens=75, estimated=False))
    assert a.total_input_tokens == 300
    assert a.total_output_tokens == 125
    assert a.total_tokens == 425
    assert a.tokens_estimated is False
    assert a.iteration_count == 2


def test_estimated_flag_is_sticky():
    """Once ANY iteration reports estimated, the WHOLE run is estimated.
    This is the load-bearing honest-reporting invariant from gnhf's
    OrchestratorState — exact totals after a single estimated iteration
    would silently lie about precision."""
    a = TokenAccounting()
    a.add(UsageReport(input_tokens=100, output_tokens=50, estimated=False))
    a.add(UsageReport(input_tokens=200, output_tokens=75, estimated=True))   # one bad iter
    a.add(UsageReport(input_tokens=300, output_tokens=100, estimated=False))  # back to exact
    assert a.tokens_estimated is True   # sticky: never resets


def test_summary_line_marks_estimated_with_tilde_prefix():
    a = TokenAccounting()
    a.add(UsageReport(input_tokens=100, output_tokens=50, estimated=True))
    line = a.summary_line()
    assert "~100" in line
    assert "~50" in line
    assert "estimated" in line


def test_summary_line_no_tilde_when_all_exact():
    a = TokenAccounting()
    a.add(UsageReport(input_tokens=100, output_tokens=50, estimated=False))
    line = a.summary_line()
    assert "~" not in line
    assert "estimated" not in line
    assert "100" in line


def test_merge_preserves_sticky_estimated():
    """Parallel iteration aggregation via __add__: estimated propagates."""
    a = TokenAccounting()
    a.add(UsageReport(input_tokens=100, output_tokens=50, estimated=False))
    b = TokenAccounting()
    b.add(UsageReport(input_tokens=200, output_tokens=75, estimated=True))
    merged = a + b
    assert merged.total_input_tokens == 300
    assert merged.total_output_tokens == 125
    assert merged.tokens_estimated is True   # propagated from b
    assert merged.iteration_count == 2
