import json
from pathlib import Path

import numpy as np
import pytest

from build_summary_index import extract_summary_nodes, kmeans_cluster


@pytest.fixture
def fixture_tree(tmp_path: Path) -> Path:
    """Build a 6-node primary tree fixture for clustering."""
    tree = {
        "node_id": "root", "title": "Root", "nodes": [
            {"node_id": f"00{i+1:02d}",
             "title": f"Section {i}",
             "summary": f"Topic {chr(ord('A') + i % 3)}",
             "tags": [f"tag_{i}"],
             "start_page": i * 2 + 1, "end_page": i * 2 + 2}
            for i in range(6)
        ],
    }
    p = tmp_path / "tree.json"
    p.write_text(json.dumps(tree))
    return p


def test_extract_summary_nodes_returns_all_summary_nodes(fixture_tree: Path) -> None:
    tree = json.loads(fixture_tree.read_text())
    leaves = extract_summary_nodes(tree)
    assert len(leaves) == 6
    assert all("node_id" in n and "summary" in n for n in leaves)


def test_kmeans_groups_summaries_deterministically() -> None:
    # 6 vectors in 2 clear groups
    embeddings = np.array([
        [1.0, 0.0], [1.0, 0.1], [1.0, -0.1],
        [0.0, 1.0], [0.1, 1.0], [-0.1, 1.0],
    ])
    labels_a = kmeans_cluster(embeddings, k=2, random_state=42)
    labels_b = kmeans_cluster(embeddings, k=2, random_state=42)
    assert (labels_a == labels_b).all()
    assert len(set(labels_a)) == 2


def test_kmeans_raises_when_k_exceeds_n() -> None:
    """k > len(embeddings) should fail-fast with actionable message."""
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="k=5"):
        kmeans_cluster(embeddings, k=5, random_state=42)


def test_atomic_write_removes_stale_partial(tmp_path: Path) -> None:
    """write_atomic must clean up any pre-existing .partial after success."""
    from build_summary_index import write_atomic
    out = tmp_path / "summary_index.json"
    stale = tmp_path / "summary_index.json.partial"
    stale.write_text('{"clusters_completed": []}')   # simulate stale journal
    payload = {"clusters": [{"cluster_id": "C1"}]}

    write_atomic(out, payload)

    assert out.exists()
    assert json.loads(out.read_text()) == payload
    assert not stale.exists(), \
        "write_atomic must remove stale .partial after successful commit"


def test_journal_partial_persists_completed_clusters(tmp_path: Path) -> None:
    from build_summary_index import journal_partial, load_partial
    out = tmp_path / "summary_index.json"
    journal_partial(out, [
        {"cluster_id": "C1", "title": "T1", "summary": "S1",
         "tags": [], "member_node_ids": ["0001"], "primary_pages": [[1, 2]]},
    ])
    completed = load_partial(out)
    assert {c["cluster_id"] for c in completed} == {"C1"}


def test_resume_flow_cleans_partial_on_commit(tmp_path: Path) -> None:
    """Full lifecycle: journal twice, load returns latest, then commit
    clears the partial."""
    from build_summary_index import (
        write_atomic, journal_partial, load_partial, _partial_path,
    )
    out = tmp_path / "summary_index.json"
    clusters = [{"cluster_id": "C1"}, {"cluster_id": "C2"}]
    journal_partial(out, clusters[:1])
    journal_partial(out, clusters)
    assert load_partial(out) == clusters
    write_atomic(out, {"clusters": clusters})
    assert out.exists()
    assert not _partial_path(out).exists(), \
        "write_atomic must clean .partial after committing the final artifact"
