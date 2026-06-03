# tests/test_aggregate_merge.py — regression guard for the per-run merge fix
"""Proves scripts/aggregate.merge_runs reconciles per-run files latest-per-cell:
a later run replaces only the cells it ran, never clobbering sibling backends or
sibling questions. This is the invariant the single-file `unlink`-and-overwrite
driver violated (it lost full-run raw 3× in one session)."""
from __future__ import annotations

import json

from scripts.aggregate import aggregate, merge_runs


def _write(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def test_later_run_replaces_only_its_own_cells(tmp_path):
    # Run 1 (oldest): full run, two questions, two backends each.
    run1 = tmp_path / "run_full_111.jsonl"
    _write(run1, [
        {"question_id": "q1", "question_type": "knowledge-update", "gold": "A",
         "atomic_fact": {"correct": False, "predicted": "old-af"},
         "mem0": {"correct": False, "predicted": "old-m0"}},
        {"question_id": "q2", "question_type": "multi-session", "gold": "B",
         "atomic_fact": {"correct": True, "predicted": "keep-af"},
         "mem0": {"correct": True, "predicted": "keep-m0"}},
    ])
    # Run 2 (newest): re-ran ONLY atomic_fact for ONLY q1 (the KU fix).
    run2 = tmp_path / "run_atomic_fact_q1_222.jsonl"
    run2.write_text(json.dumps(
        {"question_id": "q1", "question_type": "knowledge-update", "gold": "A",
         "atomic_fact": {"correct": True, "predicted": "new-af"}}) + "\n")
    # mtime order: run1 older than run2.
    import os
    os.utime(run1, (1000, 1000))
    os.utime(run2, (2000, 2000))

    merged = merge_runs([run1, run2])

    # q1 atomic_fact = refreshed; q1 mem0 = preserved (run2 never touched it).
    assert merged["q1"]["atomic_fact"]["predicted"] == "new-af"
    assert merged["q1"]["atomic_fact"]["correct"] is True
    assert merged["q1"]["mem0"]["predicted"] == "old-m0"  # NOT clobbered
    # q2 entirely preserved — run2 never mentioned it.
    assert merged["q2"]["atomic_fact"]["predicted"] == "keep-af"
    assert merged["q2"]["mem0"]["predicted"] == "keep-m0"


def test_accuracy_excludes_broken_gold(tmp_path):
    run = tmp_path / "run_full_111.jsonl"
    _write(run, [
        {"question_id": "0a995998", "question_type": "multi-session", "gold": "3",
         "atomic_fact": {"correct": False}},  # broken-gold → must be excluded
        {"question_id": "good", "question_type": "multi-session", "gold": "4",
         "atomic_fact": {"correct": True}},
    ])
    merged = merge_runs([run])
    agg = aggregate(list(merged.values()), ["atomic_fact"])
    bools = agg["axis_backend"]["multi-session"]["atomic_fact"]
    # Only the good question counts → 1 sample, all correct.
    assert bools == [True]


def test_pending_counts_correct_none(tmp_path):
    run = tmp_path / "run_full_111.jsonl"
    _write(run, [
        {"question_id": "q1", "question_type": "multi-session", "gold": "A",
         "mem0": {"correct": None}},  # judge deferred under 503 cooldown
        {"question_id": "q2", "question_type": "multi-session", "gold": "B",
         "mem0": {"correct": True}},
    ])
    agg = aggregate(list(merge_runs([run]).values()), ["mem0"])
    assert agg["pending"]["mem0"] == 1
    assert agg["axis_backend"]["multi-session"]["mem0"] == [True]
