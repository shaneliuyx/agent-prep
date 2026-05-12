import json
from pathlib import Path

import numpy as np
import pytest

from build_summary_index import (
    auto_k_silhouette,
    build_page_vector_index,
    extract_summary_nodes,
    kmeans_cluster,
    resolve_k,
)


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


def test_summarize_cluster_returns_required_fields() -> None:
    from build_summary_index import summarize_cluster

    member_summaries = [
        "Buffett discusses Coca-Cola, American Express ownership.",
        "Berkshire holds 27.8% of Occidental Petroleum.",
    ]

    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    class M:
                        content = json.dumps({
                            "title": "Non-controlled investments",
                            "summary": ("Discusses Coca-Cola, American Express, "
                                        "and 27.8% Occidental Petroleum stakes."),
                            "tags": ["Coca-Cola", "American Express",
                                     "Occidental", "27.8%"],
                        })
                    class C:
                        message = M()
                    class R:
                        choices = [C()]
                    return R()

    out = summarize_cluster(FakeClient(), "test-model", member_summaries)
    assert out["title"]
    assert len(out["summary"]) >= 30
    assert len(out["tags"]) >= 3


def test_summarize_cluster_llm_error_returns_empty_fields() -> None:
    """LLM raises → empty-fields dict, never propagates exception."""
    from build_summary_index import summarize_cluster

    class ErrorClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    raise RuntimeError("rate limit")

    out = summarize_cluster(ErrorClient(), "test-model", ["summary text"])
    assert out == {"title": "", "summary": "", "tags": []}


def test_summarize_cluster_bad_json_returns_empty_fields() -> None:
    """Non-JSON content → empty-fields dict (parse failure swallowed)."""
    from build_summary_index import summarize_cluster

    class BadJsonClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    class M:
                        content = "not json at all"
                    class C:
                        message = M()
                    class R:
                        choices = [C()]
                    return R()

    out = summarize_cluster(BadJsonClient(), "test-model", ["summary text"])
    assert out == {"title": "", "summary": "", "tags": []}


def test_summarize_cluster_tags_with_non_strings_filtered() -> None:
    """Tags containing non-strings (None, ints) get filtered to str-only."""
    from build_summary_index import summarize_cluster

    class MixedTagsClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    class M:
                        content = json.dumps({
                            "title": "T",
                            "summary": "S" * 30,
                            "tags": ["valid", 1, None, "Coca-Cola", 27.8],
                        })
                    class C:
                        message = M()
                    class R:
                        choices = [C()]
                    return R()

    out = summarize_cluster(MixedTagsClient(), "test-model", ["x"])
    assert out["tags"] == ["valid", "Coca-Cola"]


def test_summarize_cluster_list_response_preserved_as_tags() -> None:
    """DWQ sometimes returns JSON list (schema-disagreement) — preserve content."""
    from build_summary_index import summarize_cluster

    class ListResponseClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    class M:
                        content = json.dumps([
                            "Charlie Munger", "Berkshire Hathaway",
                            "Our Not-So-Secret Weapon", "$96 billion",
                        ])
                    class C:
                        message = M()
                    class R:
                        choices = [C()]
                    return R()

    out = summarize_cluster(ListResponseClient(), "test-model", ["x"])
    assert out["title"] == ""
    assert out["summary"] == ""
    assert out["tags"] == [
        "Charlie Munger", "Berkshire Hathaway",
        "Our Not-So-Secret Weapon", "$96 billion",
    ]


# ===== auto_k_silhouette tests (Phase 7 follow-up — Auto-K selection) =====

def _three_cluster_synthetic(n_per_cluster: int = 8, dim: int = 16,
                             rng_seed: int = 7) -> np.ndarray:
    """Synthetic 3-cluster data: well-separated isotropic Gaussians around
    3 distinct centers. Silhouette score should clearly peak at k=3."""
    rng = np.random.default_rng(rng_seed)
    centers = np.array([
        [1.0] * dim,                             # cluster A
        [-1.0] * dim,                            # cluster B
        [1.0 if i % 2 == 0 else -1.0 for i in range(dim)],  # cluster C alternating
    ], dtype=np.float32)
    parts = [centers[i] + rng.normal(0, 0.1, (n_per_cluster, dim)).astype(np.float32)
             for i in range(3)]
    return np.vstack(parts)


def test_auto_k_silhouette_picks_correct_k_on_synthetic_3_cluster() -> None:
    embeddings = _three_cluster_synthetic()
    chosen_k, scores = auto_k_silhouette(embeddings, k_range=(2, 6))
    # Strict assertion: 3 well-separated clusters → silhouette peaks at k=3
    assert chosen_k == 3, f"expected k=3 on synthetic, got {chosen_k}; scores={scores}"
    assert isinstance(scores, list) and len(scores) == 5, \
        f"expected 5 (k=2..6) score entries, got {len(scores)}"
    assert all(isinstance(s, tuple) and len(s) == 2 for s in scores), \
        "scores entries must be (k, score) tuples"


def test_auto_k_silhouette_truncates_when_k_exceeds_n() -> None:
    """k_range upper bound > n_embeddings - 1 must be silently truncated,
    not crash. silhouette_score requires at least 2 clusters AND k < n."""
    embeddings = _three_cluster_synthetic(n_per_cluster=2)  # n=6 only
    chosen_k, scores = auto_k_silhouette(embeddings, k_range=(2, 12))
    assert chosen_k <= 5, f"chosen_k must respect n-1 ceiling for n=6, got {chosen_k}"
    assert all(k <= 5 for k, _ in scores), \
        f"scores curve must not include k > n-1; got {[k for k, _ in scores]}"


def test_auto_k_silhouette_tiebreak_prefers_smaller_k() -> None:
    """When k=4 score is within min_improvement of k=3, prefer k=3 (Occam)."""
    embeddings = _three_cluster_synthetic()
    chosen_k, _ = auto_k_silhouette(embeddings, k_range=(3, 5),
                                     min_improvement=0.5)
    assert chosen_k == 3, \
        f"with min_improvement=0.5, k=3 should win unconditionally, got {chosen_k}"


def test_auto_k_silhouette_returns_curve_for_diagnostics() -> None:
    """Full (k, score) curve must be persistable for build_meta forensics."""
    embeddings = _three_cluster_synthetic()
    _, scores = auto_k_silhouette(embeddings, k_range=(2, 5))
    # All scores must be in [-1, 1] (silhouette range)
    for k, s in scores:
        assert -1.0 <= s <= 1.0, f"silhouette score out of range: k={k} s={s}"
        assert k >= 2, f"silhouette undefined for k<2; got {k}"


# ===== resolve_k tests (CLI flag → final k decision) =====

def test_resolve_k_returns_requested_when_auto_off() -> None:
    """auto_k=False: pass through requested k, no curve computed."""
    embeddings = _three_cluster_synthetic()
    k, curve = resolve_k(requested_k=8, auto_k=False, embeddings=embeddings)
    assert k == 8
    assert curve is None


def test_resolve_k_runs_silhouette_when_auto_on() -> None:
    """auto_k=True: pick k via silhouette, return curve for build_meta."""
    embeddings = _three_cluster_synthetic()
    k, curve = resolve_k(requested_k=8, auto_k=True, embeddings=embeddings)
    assert k == 3, f"silhouette should pick k=3 on synthetic 3-cluster, got {k}"
    assert curve is not None and len(curve) > 0
    assert all(isinstance(t, tuple) and len(t) == 2 for t in curve)


def test_resolve_k_caps_auto_at_requested_k() -> None:
    """When auto_k=True, requested_k acts as upper bound on the silhouette
    sweep. Prevents auto-K from exploring k larger than the user's max."""
    embeddings = _three_cluster_synthetic()
    k, curve = resolve_k(requested_k=4, auto_k=True, embeddings=embeddings)
    assert k <= 4
    assert curve is not None
    assert all(k_val <= 4 for k_val, _ in curve)


# ===== build_page_vector_index tests (chunk-level fallback build artifact) =====

def test_build_page_vector_index_writes_dense_artifact(tmp_path: Path) -> None:
    """Hybrid encoder writes dense .npy. File is shape (n_pages, dim)."""
    pages = ["page one text", "page two text", "page three text"]

    def encoder(texts: list[str]) -> dict:
        dense = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
                         dtype=np.float32)[: len(texts)]
        sparse = [{ord(t[0]): 1.0} for t in texts]
        return {"dense": dense, "sparse": sparse}

    out = tmp_path / "page_vectors.npy"
    build_page_vector_index(pages_text=pages, output_path=out, encoder=encoder)

    assert out.exists()
    loaded = np.load(out)
    assert loaded.shape == (3, 2), f"expected (3, 2), got {loaded.shape}"


def test_build_page_vector_index_writes_sparse_when_provided(tmp_path: Path) -> None:
    """Sparse output → .sparse.json sibling file with serialized weights."""
    pages = ["alpha", "beta"]

    def encoder(texts: list[str]) -> dict:
        return {
            "dense": np.eye(2, dtype=np.float32),
            "sparse": [{1: 5.5, 2: 3.0}, {7: 9.0}],
        }

    out = tmp_path / "page_vectors.npy"
    build_page_vector_index(pages_text=pages, output_path=out, encoder=encoder)

    sparse_file = tmp_path / "page_vectors.sparse.json"
    assert sparse_file.exists()
    raw = json.loads(sparse_file.read_text())
    assert len(raw) == 2
    # Keys serialized as strings; values preserved
    assert raw[0]["1"] == 5.5
    assert raw[1]["7"] == 9.0


def test_build_page_vector_index_dense_only_mode(tmp_path: Path) -> None:
    """When encoder returns only dense (no 'sparse' key), no sparse file
    written. PageVectorIndex.load() then runs in dense-only mode."""
    pages = ["a", "b"]

    def encoder(texts: list[str]) -> dict:
        return {"dense": np.eye(2, dtype=np.float32)}

    out = tmp_path / "page_vectors.npy"
    build_page_vector_index(pages_text=pages, output_path=out, encoder=encoder)

    assert out.exists()
    sparse_file = tmp_path / "page_vectors.sparse.json"
    assert not sparse_file.exists(), \
        "no sparse file should be written when encoder omits sparse"


def test_build_page_vector_index_creates_parent_dir(tmp_path: Path) -> None:
    """Output path with non-existent parent directory must auto-mkdir."""
    pages = ["x"]
    encoder = lambda t: {"dense": np.array([[1.0, 0.0]], dtype=np.float32)}
    nested = tmp_path / "nested" / "subdir" / "page_vectors.npy"
    build_page_vector_index(pages_text=pages, output_path=nested, encoder=encoder)
    assert nested.exists()
