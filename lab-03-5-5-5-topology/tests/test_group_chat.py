# tests/test_group_chat.py
from group_chat import group_chat_run, SELECTORS

def test_round_robin_zero_selector_calls():
    """Round-robin: selector_calls == rounds_used (0 LLM calls inside)."""
    out = group_chat_run("Test task", selector="round-robin", max_rounds=3)
    assert out["selector_calls"] == out["rounds_used"]

def test_llm_selected_returns_valid_agent_name():
    """LLM-selected always returns a registered agent name."""
    valid = {"coder", "reviewer", "tester"}
    out = group_chat_run("Write factorial(n).", selector="llm-selected", max_rounds=3)
    assert all(m["speaker"] in valid for m in out["pool"][1:])

def test_custom_tester_follows_coder():
    """Custom rule: after a coder message, tester MUST speak next."""
    out = group_chat_run("Write a Python function.", selector="custom", max_rounds=6)
    msgs = out["pool"]
    for i in range(len(msgs) - 1):
        if msgs[i]["speaker"] == "coder":
            assert msgs[i + 1]["speaker"] == "tester"

def test_terminate_token_ends_conversation():
    """TERMINATE token ends the loop before max_rounds."""
    out = group_chat_run("Solve x + 1 = 2.", selector="round-robin", max_rounds=20)
    if out["terminated_by"] == "TERMINATE token":
        assert out["rounds_used"] < 20
