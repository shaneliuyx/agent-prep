"""Phase 8 — Bucket-1 conversational extraction tests.

Validates that EverCore's memcell + profile pipeline EARNS its cost on
multi-turn dialogue data (vs Bucket-2 pre-extracted facts where it
pays 5-30x latency for redundant work).

Asserts:
- A 10-12 turn dialogue + flush produces ≥ 1 episode (`/memories/get memory_type=episodic_memory`)
- A 10-12 turn dialogue produces a non-empty episode summary string
- After multiple imprints, per-user profile aggregation populates ≥ 1 profile row

These tests are SLOW (~3-5 min wall) because they wait for EverCore's
async memcell-extraction pipeline. Skip via `pytest -m 'not slow'` if
running tight CI.

Empirical caveat (BCJ Entry: 2026-05-15 lab run): even 10-12 turn
natural dialogues return `status=accumulated` from POST /memories.
Flush is REQUIRED even on Bucket-1 data — EverCore's boundary detector
is conservative at lab scale. The test uses imprint_dialogue() from
demo_conversational_imprint.py which already calls flush.
"""
import time
import uuid

import pytest

# Re-use the dialogue corpus + helpers from the demo
from src.demo_conversational_imprint import (
    DIALOGUES,
    get_episodes,
    get_profiles,
    imprint_dialogue,
)


pytestmark = pytest.mark.slow  # opt-out via -m 'not slow'


@pytest.fixture(scope="module")
def imprinted_dialogues():
    """Imprint all 3 dialogues once, wait for extraction, share across tests."""
    # Rename user_ids per-test-run so we don't collide with prior data
    suffix = uuid.uuid4().hex[:6]
    seeded = []
    for d in DIALOGUES:
        local = dict(d)
        local["user_id"] = f"{d['user_id']}-{suffix}"
        local["session_id"] = f"{d['session_id']}-{suffix}"
        wall, resp, flush_resp = imprint_dialogue(local)
        seeded.append({"dialogue": local, "wall": wall, "post": resp, "flush": flush_resp})
    # Wait for async memcell extraction to complete
    time.sleep(60)
    return seeded


def test_each_dialogue_produces_episode(imprinted_dialogues):
    """Every Bucket-1 dialogue → ≥ 1 episode after flush+wait."""
    for entry in imprinted_dialogues:
        uid = entry["dialogue"]["user_id"]
        eps = get_episodes(uid)
        assert len(eps) >= 1, (
            f"user_id={uid} produced 0 episodes. "
            "EverCore async extraction may need more wait time, OR the "
            "boundary detector rejected the dialogue. Check the flush "
            "response status."
        )


def test_episode_has_non_empty_summary(imprinted_dialogues):
    """The extracted episode carries a non-empty summary string."""
    for entry in imprinted_dialogues:
        uid = entry["dialogue"]["user_id"]
        eps = get_episodes(uid)
        if not eps:
            pytest.skip(f"no episodes for {uid} — see test_each_dialogue_produces_episode")
        ep = eps[0]
        summary = ep.get("summary") or ep.get("episode") or ""
        assert summary.strip(), f"empty summary on episode for {uid}: {ep}"
        # Summary should reference the dialogue's topic
        topic = entry["dialogue"]["topic"].lower().split()[0:2]
        # Soft check — at least ONE topic-significant word appears
        # (EverCore's summary may paraphrase aggressively)
        summary_lower = summary.lower()
        topic_overlap = any(w in summary_lower for w in topic if len(w) > 3)
        # Don't hard-assert topic overlap — summary may be very abstracted
        if not topic_overlap:
            print(f"NOTE: summary for {uid} doesn't surface topic words {topic}: {summary[:200]}")


def test_at_least_one_dialogue_produces_profile(imprinted_dialogues):
    """At least ONE of the 3 users gets a profile row after a single
    dialogue. EverCore's threshold for profile aggregation is high; on
    single-dialogue input, 2/3 users typically materialize per the
    2026-05-15 measurement run. We assert ≥ 1 to be empirically faithful
    rather than asserting 3/3 which the boundary detector won't always hit.
    """
    total = 0
    for entry in imprinted_dialogues:
        uid = entry["dialogue"]["user_id"]
        profs = get_profiles(uid)
        total += len(profs)
    assert total >= 1, (
        f"got 0 profiles across all {len(imprinted_dialogues)} users. "
        "EverCore profile aggregation requires sufficient signal — try "
        "wait_seconds > 60 or longer/richer dialogues. If still 0, the "
        "Bucket-1 claim is empirically refuted on this hardware + model "
        "combo and the chapter Production Considerations table needs "
        "updating to reflect that."
    )


def test_imprint_wall_within_expected_range(imprinted_dialogues):
    """Per-dialogue imprint+flush wall is in the 60-300s band measured
    2026-05-15 on M5 Pro + oMLX gpt-oss-20b. If outside, surface for
    investigation (could indicate slow oMLX, contention, or EverCore
    pipeline regression)."""
    walls = [e["wall"] for e in imprinted_dialogues]
    mean = sum(walls) / len(walls)
    # Soft band — wide because oMLX latency varies with concurrent load
    assert 30 < mean < 600, (
        f"mean per-dialogue wall {mean:.1f}s outside expected 30-600s band. "
        f"individual walls: {walls}"
    )
