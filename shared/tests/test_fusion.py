"""Unit test for rrf_fuse — math + tie-handling + dedup.

No qdrant, no model. Pure Python — runs in <100ms.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "shared"))

from rag_hybrid.fusion import rrf_fuse, RRF_DEFAULT_K


class P:
    """Minimal stand-in for Qdrant's ScoredPoint — only ``.id`` is used by RRF."""
    def __init__(self, pid: str, payload: dict | None = None) -> None:
        self.id = pid
        self.payload = payload or {}

    def __repr__(self) -> str:
        return f"P({self.id!r})"


def test_basic_two_rankings() -> None:
    # a appears first in r1 (rank 0) and second in r2 (rank 1). Same for b but flipped.
    # By the math, a and b tie. Tie broken by insertion order — r1 contributes a first.
    r1 = [P("a"), P("b"), P("c")]
    r2 = [P("b"), P("a"), P("d")]
    out = rrf_fuse([r1, r2])
    ids = [p.id for p in out]
    assert ids[:2] == ["a", "b"], f"top-2 got {ids}"
    assert set(ids) == {"a", "b", "c", "d"}, f"dedup leak: {ids}"
    assert len(out) == 4, f"expected 4 unique, got {len(out)}"


def test_payload_first_occurrence_wins() -> None:
    # Same id appears in both rankings with different payloads — the *first*
    # occurrence's payload should be the one returned. This is the contract that
    # lets callers trust they get the dense-search payload (which usually has
    # better metadata) when the dense ranking is listed first.
    r1 = [P("x", {"src": "dense"})]
    r2 = [P("x", {"src": "sparse"})]
    out = rrf_fuse([r1, r2])
    assert len(out) == 1
    assert out[0].payload == {"src": "dense"}, f"payload drift: {out[0].payload}"


def test_score_math() -> None:
    # 3 rankings, doc "z" appears at ranks 0/2/4. Verify aggregate score.
    rs = [
        [P("z"), P("a")],
        [P("a"), P("b"), P("z")],
        [P("a"), P("b"), P("c"), P("d"), P("z")],
    ]
    out = rrf_fuse(rs)
    # Compute scores by hand
    scores = {}
    for r in rs:
        for rank, p in enumerate(r):
            scores[p.id] = scores.get(p.id, 0.0) + 1.0 / (RRF_DEFAULT_K + rank + 1)
    expected_order = sorted(scores, key=lambda pid: -scores[pid])
    got_order = [p.id for p in out]
    assert got_order == expected_order, f"order drift: got {got_order}, expected {expected_order}"


def test_empty_inputs() -> None:
    # Both empty rankings — should not crash, returns []
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], []]) == []


def test_single_ranking_passthrough() -> None:
    # One ranking → output preserves order (no fusion to do).
    r = [P("a"), P("b"), P("c")]
    out = rrf_fuse([r])
    assert [p.id for p in out] == ["a", "b", "c"]


def main() -> int:
    failures = []
    for name, fn in [
        ("test_basic_two_rankings", test_basic_two_rankings),
        ("test_payload_first_occurrence_wins", test_payload_first_occurrence_wins),
        ("test_score_math", test_score_math),
        ("test_empty_inputs", test_empty_inputs),
        ("test_single_ranking_passthrough", test_single_ranking_passthrough),
    ]:
        try:
            fn()
            print(f"PASS: {name}")
        except AssertionError as e:
            failures.append(f"{name}: {e}")
            print(f"FAIL: {name}: {e}")
    if failures:
        return 1
    print(f"OK — {5} fusion tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
