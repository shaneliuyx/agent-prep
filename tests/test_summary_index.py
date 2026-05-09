import json
from pathlib import Path
from tree_index._hashing import tree_hash


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
