"""15-question recall benchmark for the W3.5 cross-session memory lab.

Each test is one (seed, probe, assertion) triple. The seed pre-populates
memory with N facts via remember_turn(); a fresh "session" then calls
recall() with a question that requires one or more of those facts.

Target: 12 of 15 passing. Failing cases go into the Bad-Case Journal.

Run via pytest from the project root:
    .venv/bin/python -m pytest tests/test_recall.py -v

Or directly (the sys.path bootstrap below makes `python tests/test_recall.py`
also work, in case someone hits this from outside pytest):
    .venv/bin/python tests/test_recall.py

Caveats:
- Tests depend on LLM extraction quality (gpt-oss-20b for MODEL_HAIKU).
  Some tests will be flaky on extraction edge cases; that's expected
  signal for the BCJ.
- Tests share Qdrant + SQLite state; isolated via unique user_ids per
  test (`make_user()` generates random suffixes).
"""
import sys
from pathlib import Path

# Bootstrap: add project root to sys.path so `from src.memory import ...`
# works whether this file is run via pytest (cwd=project root) or
# directly (cwd=tests/). Idempotent — re-running adds nothing duplicate.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import sqlite3
import uuid

import pytest

from src.memory_mem0 import format_memory_block, recall, remember_turn
# (SQLITE is hand-rolled-only; tests using it directly need adaptation OR skipping)

def make_user() -> str:
    return "bench_" + str(uuid.uuid4())[:8]


def new_session() -> str:
    return str(uuid.uuid4())


# ──────────────────────────────────────────────────────────────────────
# 1-4: basic semantic recall (one fact, one key)
# ──────────────────────────────────────────────────────────────────────

def test_01_recall_location() -> None:
    u, s = make_user(), new_session()
    remember_turn(u, s, "I live in Osaka.", "Got it — Osaka.")
    mem = recall(u, "any good ramen nearby?")
    assert any("osaka" in f["value"].lower() for f in mem["semantic_facts"])


def test_02_recall_diet() -> None:
    u, s = make_user(), new_session()
    remember_turn(u, s, "I'm vegan.", "Got it — vegan.")
    mem = recall(u, "lunch ideas?")
    assert any("vegan" in f["value"].lower() for f in mem["semantic_facts"])


def test_03_recall_job() -> None:
    u, s = make_user(), new_session()
    remember_turn(u, s, "I work as a cloud infrastructure engineer.", "Noted.")
    mem = recall(u, "any career advice?")
    facts = mem["semantic_facts"]
    assert any(
        "engineer" in f["value"].lower() or "cloud" in f["value"].lower()
        for f in facts
    )


def test_04_recall_hobby() -> None:
    u, s = make_user(), new_session()
    remember_turn(u, s, "I ride my bicycle to work every day.", "Got it.")
    mem = recall(u, "fitness goals?")
    facts = mem["semantic_facts"]
    assert any(
        any(t in f["value"].lower() for t in ("bicycle", "bike", "cycling"))
        for f in facts
    )


# ──────────────────────────────────────────────────────────────────────
# 5: contradiction update — archive-on-overwrite (SCD-2)
# ──────────────────────────────────────────────────────────────────────

def test_05_contradiction_update_latest_wins() -> None:
    u, s = make_user(), new_session()
    remember_turn(u, s, "I live in Osaka.", "Got it — Osaka.")
    remember_turn(u, s, "Actually I moved to Tokyo.", "Got it — Tokyo.")
    mem = recall(u, "where do I live now?")
    locs = [
        f["value"].lower()
        for f in mem["semantic_facts"]
        if f["key"] in ("location", "city", "residence")
    ]
    assert any("tokyo" in v for v in locs)
    assert not any("osaka" in v for v in locs)


# ──────────────────────────────────────────────────────────────────────
# 6: multi-fact composition — two facts in one turn, both extracted
# ──────────────────────────────────────────────────────────────────────

def test_06_multi_fact_composition() -> None:
    u, s = make_user(), new_session()
    remember_turn(
        u, s,
        "I'm vegan and I live in Taipei.",
        "Noted — vegan in Taipei.",
    )
    mem = recall(u, "lunch nearby?")
    vals = " ".join(f["value"].lower() for f in mem["semantic_facts"])
    assert "vegan" in vals
    assert "taipei" in vals


# ──────────────────────────────────────────────────────────────────────
# 7: cross-session recall — same user, different sessions
# ──────────────────────────────────────────────────────────────────────

def test_07_cross_session_recall() -> None:
    """Cross-session recall is the architectural DEFAULT: semantic facts
    in SQLite have no session_id column; episodic memories in Qdrant
    store session_id in payload but recall() filters only on user_id.

    Test strategy: write in one session (s1), recall without passing
    any session context, verify BOTH stores' contents surface.
    The semantic store proves the fact persists; the episodic store
    proves recall ignores session_id (the load-bearing cross-session
    property — without it, episodic memories would be session-scoped).
    """
    u = make_user()
    s1 = new_session()
    remember_turn(u, s1, "I'm allergic to peanuts.", "Got it.")

    # Recall takes (user_id, query) — no session_id. By design, the
    # facts written under s1 should appear in any subsequent recall
    # for the same user, regardless of which session does the query.
    mem = recall(u, "any food I should avoid?")

    # (a) Semantic store: the fact persists across session close.
    assert any(
        "peanut" in f["value"].lower() for f in mem["semantic_facts"]
    ), "semantic fact written in s1 should be recallable cross-session"

    # (b) Episodic store: the s1 episode surfaces too. Episodes carry
    # session_id in their Qdrant payload — if recall accidentally
    # filtered on session, this assertion would fail (no episode for
    # the "current session" since recall didn't receive one).
    episodic_text = " ".join(mem["relevant_episodes"]).lower()
    assert "peanut" in episodic_text or "allerg" in episodic_text, (
        "episodic memory from s1 should surface; if it doesn't, "
        "recall is implicitly session-scoping (cross-session is broken)"
    )


# ──────────────────────────────────────────────────────────────────────
# 8: user isolation — alice's facts don't leak to bob
# ──────────────────────────────────────────────────────────────────────

def test_08_user_isolation() -> None:
    alice, bob = make_user(), make_user()
    s = new_session()
    remember_turn(alice, s, "I live in Tokyo.", "Got it.")
    mem_bob = recall(bob, "where do I live?")
    # Bob has no facts; if any are returned, isolation is broken
    assert not any(
        "tokyo" in f["value"].lower() for f in mem_bob["semantic_facts"]
    )


# ──────────────────────────────────────────────────────────────────────
# 9: empty user — fresh user has no facts
# ──────────────────────────────────────────────────────────────────────

def test_09_empty_user_returns_empty() -> None:
    u = make_user()
    mem = recall(u, "what do you know about me?")
    assert mem["semantic_facts"] == []
    assert mem["relevant_episodes"] == []


# ──────────────────────────────────────────────────────────────────────
# 10: episodic surfaces on relevant query (semantic similarity > 0.35)
# ──────────────────────────────────────────────────────────────────────

def test_10_episodic_surfaces_on_relevant_query() -> None:
    u, s = make_user(), new_session()
    remember_turn(
        u, s,
        "I asked about setting up LangGraph for a customer-support agent.",
        "Sure — here are the LangGraph patterns.",
    )
    mem = recall(u, "tell me more about agent frameworks")
    text = " ".join(mem["relevant_episodes"]).lower()
    assert "langgraph" in text or "agent" in text


# ──────────────────────────────────────────────────────────────────────
# 11: episodic does NOT surface on irrelevant query (threshold filtering)
# ──────────────────────────────────────────────────────────────────────

def test_11_episodic_does_not_surface_on_irrelevant_query() -> None:
    u, s = make_user(), new_session()
    remember_turn(
        u, s,
        "I asked about Python list comprehensions.",
        "Sure — here's how list comprehensions work.",
    )
    mem = recall(u, "what's the weather like in Antarctica?")
    text = " ".join(mem["relevant_episodes"]).lower()
    assert "comprehension" not in text and "python" not in text


# ──────────────────────────────────────────────────────────────────────
# 12: multiple contradictions — only the latest value remains
# ──────────────────────────────────────────────────────────────────────

def test_12_multiple_contradictions_latest_wins() -> None:
    u, s = make_user(), new_session()
    remember_turn(u, s, "I live in Osaka.", "Got it.")
    remember_turn(u, s, "Actually I moved to Tokyo.", "Got it.")
    remember_turn(u, s, "Sorry, I meant Kyoto.", "Got it.")
    mem = recall(u, "where do I live?")
    locs = [
        f["value"].lower()
        for f in mem["semantic_facts"]
        if f["key"] in ("location", "city", "residence")
    ]
    assert any("kyoto" in v for v in locs)
    assert not any(("osaka" in v) or ("tokyo" in v) for v in locs)


# ──────────────────────────────────────────────────────────────────────
# 13: synonym query — different words, same concept
# ──────────────────────────────────────────────────────────────────────

def test_13_synonym_query() -> None:
    u, s = make_user(), new_session()
    remember_turn(u, s, "I currently reside in Seoul.", "Got it.")
    mem = recall(u, "my city of residence")
    # Semantic facts OR episodic — either should catch it
    has_seoul = any(
        "seoul" in f["value"].lower() for f in mem["semantic_facts"]
    ) or any("seoul" in e.lower() for e in mem["relevant_episodes"])
    assert has_seoul


# ──────────────────────────────────────────────────────────────────────
# 14: format_memory_block — non-empty when facts exist
# ──────────────────────────────────────────────────────────────────────

def test_14_format_memory_block_with_facts_nonempty() -> None:
    mem = {
        "semantic_facts": [{"key": "location", "value": "Taipei"}],
        "relevant_episodes": ["user asked about cycling routes"],
    }
    block = format_memory_block(mem)
    assert "Taipei" in block
    assert "cycling" in block.lower()
    assert "Known facts" in block


# ──────────────────────────────────────────────────────────────────────
# 15: archived rows not returned — direct DB inject + verify filtering
# ──────────────────────────────────────────────────────────────────────
import os

@pytest.mark.skipif(
    os.getenv("MEMORY_BACKEND") == "mem0",
    reason="test_15 inserts directly into SQLite; mem0 uses its own backend"
)
def test_15_archived_facts_not_returned() -> None:
    u = make_user()
    conn = sqlite3.connect(SQLITE, timeout=30)
    conn.execute(
        "INSERT INTO user_facts (user_id, key, value, archived) "
        "VALUES (?, ?, ?, 1)",
        (u, "location", "Osaka"),
    )
    conn.commit()
    conn.close()
    mem = recall(u, "where do I live?")
    # Archived row must NOT surface in recall results
    assert not any(
        "osaka" in f["value"].lower() for f in mem["semantic_facts"]
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
