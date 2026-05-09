import json
import sys
from pathlib import Path

import numpy as np
import pytest


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


def test_extract_leaves_returns_all_summary_nodes(fixture_tree: Path) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                          / "lab-02-7-pageindex" / "src"))
    from build_summary_index import extract_leaves
    tree = json.loads(fixture_tree.read_text())
    leaves = extract_leaves(tree)
    assert len(leaves) == 6
    assert all("node_id" in n and "summary" in n for n in leaves)


def test_kmeans_groups_summaries_deterministically() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                          / "lab-02-7-pageindex" / "src"))
    from build_summary_index import kmeans_cluster
    # 6 vectors in 2 clear groups
    embeddings = np.array([
        [1.0, 0.0], [1.0, 0.1], [1.0, -0.1],
        [0.0, 1.0], [0.1, 1.0], [-0.1, 1.0],
    ])
    labels_a = kmeans_cluster(embeddings, k=2, random_state=42)
    labels_b = kmeans_cluster(embeddings, k=2, random_state=42)
    assert (labels_a == labels_b).all()
    assert len(set(labels_a)) == 2
