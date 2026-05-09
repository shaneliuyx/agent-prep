import json
from pathlib import Path

import pytest

from tree_index._hashing import tree_hash
from tree_index.summary_index import SummaryIndex


def test_tree_hash_stable_across_whitespace(tmp_path: Path) -> None:
    """Same content + different formatting -> same hash."""
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text('{"node_id":"0001","title":"X"}')
    b.write_text('{\n  "node_id": "0001",\n  "title": "X"\n}')
    assert tree_hash(a) == tree_hash(b)


def test_tree_hash_changes_on_content_change(tmp_path: Path) -> None:
    """Different content -> different hash."""
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text('{"node_id":"0001","title":"X"}')
    b.write_text('{"node_id":"0001","title":"Y"}')
    assert tree_hash(a) != tree_hash(b)


def test_tree_hash_independent_of_key_order(tmp_path: Path) -> None:
    """Same content + different key order -> same hash."""
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text('{"node_id":"0001","title":"X"}')
    b.write_text('{"title":"X","node_id":"0001"}')
    assert tree_hash(a) == tree_hash(b)


def _write_fixture(tmp_path: Path, tree_hash_value: str) -> tuple[Path, Path]:
    """Create a tree.json + summary_index.json pair for testing."""
    tree = tmp_path / "tree.json"
    tree.write_text('{"node_id":"0001","title":"X"}')
    idx = tmp_path / "summary_index.json"
    idx.write_text(json.dumps({
        "build_meta": {
            "tree_hash": tree_hash_value,
            "k": 2, "embedding_model": "BGE-M3",
            "created": "2026-05-09T01:00:00Z",
        },
        "clusters": [
            {"cluster_id": "C1", "title": "T1", "summary": "S1",
             "tags": ["a", "b"], "member_node_ids": ["0001"],
             "primary_pages": [[1, 2]]},
        ],
    }))
    return tree, idx


def test_summary_index_loads_when_hash_matches(tmp_path: Path) -> None:
    # Write tree first with placeholder, compute its hash, then rewrite
    # the index with the correct hash so the binding validates.
    tree, idx = _write_fixture(tmp_path, "placeholder")
    correct_hash = tree_hash(tree)
    tree, idx = _write_fixture(tmp_path, correct_hash)
    si = SummaryIndex(idx, tree)
    assert len(si.clusters) == 1
    assert si.clusters[0]["cluster_id"] == "C1"


def test_summary_index_raises_on_stale_hash(tmp_path: Path) -> None:
    tree, idx = _write_fixture(tmp_path, "abc123_wrong")
    with pytest.raises(RuntimeError, match="stale"):
        SummaryIndex(idx, tree)


def test_summary_index_missing_file_raises_with_helpful_message(
    tmp_path: Path,
) -> None:
    tree = tmp_path / "tree.json"
    tree.write_text('{"node_id":"0001"}')
    with pytest.raises(FileNotFoundError, match="build_summary_index"):
        SummaryIndex(tmp_path / "missing.json", tree)
