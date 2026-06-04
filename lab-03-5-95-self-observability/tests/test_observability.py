"""Unit tests for the OBSERVABILITY tier (no LLM — fast, deterministic)."""
import sqlite3

import observability as obs
import pytest


def _db(tmp_path):
    return obs.connect(str(tmp_path / "t.db"))


def test_append_and_query_by_tool(tmp_path):
    conn = _db(tmp_path)
    for i in range(5):
        obs.log_observation(conn, agent_run_id="r1", step_idx=i,
                            tool_name="grep" if i % 2 else "rg",
                            args={"q": "x"}, outcome={"ok": True},
                            outcome_status="ok", latency_ms=1.0)
    assert len(obs.observations_by_tool(conn, "grep")) == 2
    assert len(obs.observations_by_tool(conn, "rg")) == 3
    assert len(obs.recent_observations(conn, limit=3)) == 3
    assert len(obs.observations_by_run(conn, "r1")) == 5


def test_append_only_pk(tmp_path):
    """Append-only discipline: a duplicate (run_id, step_idx) is surfaced, not
    silently overwritten."""
    conn = _db(tmp_path)
    obs.log_observation(conn, agent_run_id="r", step_idx=0, tool_name="grep",
                        args={}, outcome={}, outcome_status="ok", latency_ms=1.0)
    with pytest.raises(sqlite3.IntegrityError):
        obs.log_observation(conn, agent_run_id="r", step_idx=0, tool_name="rg",
                            args={}, outcome={}, outcome_status="ok", latency_ms=1.0)


def test_pii_scrubbed_at_write(tmp_path):
    """Secrets/paths in args are redacted before persisting (BCJ — write-boundary scrub)."""
    conn = _db(tmp_path)
    obs.log_observation(conn, agent_run_id="r", step_idx=0, tool_name="curl",
                        args={"url": "api", "key": "sk-abcdef0123456789abcdef",
                              "path": "/Users/alice/secret.txt", "email": "a@b.com"},
                        outcome={}, outcome_status="ok", latency_ms=1.0)
    row = obs.observations_by_run(conn, "r")[0]
    assert "sk-abcdef" not in row["args_json"]
    assert "/Users/alice" not in row["args_json"]
    assert "a@b.com" not in row["args_json"]
    assert "<API_KEY>" in row["args_json"] and "/Users/<USER>" in row["args_json"]


def test_presidio_scrubs_named_entities(tmp_path):
    """Presidio upgrade: NER catches PII a fixed regex CANNOT — person names,
    locations, phones, credit cards, IPs. Skips when the regex fallback is active
    (Presidio / spaCy model not installed)."""
    import pii_scrub
    if pii_scrub.backend() != "presidio":
        pytest.skip("Presidio backend unavailable — regex fallback can't catch named entities")
    conn = _db(tmp_path)
    obs.log_observation(conn, agent_run_id="r", step_idx=0, tool_name="notify",
                        args={"note": "ticket from Dr. Sarah Johnson in Seattle, card 4111 1111 1111 1111"},
                        outcome={}, outcome_status="ok", latency_ms=1.0)
    blob = obs.observations_by_run(conn, "r")[0]["args_json"]
    assert "Sarah Johnson" not in blob and "<PERSON>" in blob
    assert "4111 1111 1111 1111" not in blob and "<CREDIT_CARD>" in blob


def test_raw_args_optout_keeps_secrets(tmp_path):
    conn = _db(tmp_path)
    obs.log_observation(conn, agent_run_id="r", step_idx=0, tool_name="curl",
                        args={"key": "sk-abcdef0123456789abcdef"}, outcome={},
                        outcome_status="ok", latency_ms=1.0, raw_args=True)
    assert "sk-abcdef0123456789abcdef" in obs.observations_by_run(conn, "r")[0]["args_json"]


def test_tool_query_uses_index(tmp_path):
    """The by-tool query must hit ix_obs_tool_ts, not scan."""
    conn = _db(tmp_path)
    obs.log_observation(conn, agent_run_id="r", step_idx=0, tool_name="grep",
                        args={}, outcome={}, outcome_status="ok", latency_ms=1.0)
    plan = conn.execute(
        "EXPLAIN QUERY PLAN SELECT * FROM observability WHERE tool_name='grep' ORDER BY ts DESC"
    ).fetchall()
    assert any("ix_obs_tool_ts" in str(tuple(r)) for r in plan), [tuple(r) for r in plan]


def test_user_signal_stamp(tmp_path):
    conn = _db(tmp_path)
    obs.log_observation(conn, agent_run_id="r", step_idx=0, tool_name="grep",
                        args={}, outcome={}, outcome_status="ok", latency_ms=1.0)
    obs.stamp_user_signal(conn, "r", 0, "thumbs_down")
    assert obs.observations_by_run(conn, "r")[0]["user_signal"] == "thumbs_down"
