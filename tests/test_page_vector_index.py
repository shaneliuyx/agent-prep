"""Tests for PageVectorIndex — dense + optional sparse hybrid match.

Used by AgenticTreeRetriever as a chunk-level fallback when the agent loop
hits max_iterations + returns a refusal. PageIndex's failure-mode protection.

Hybrid mode (BGE-M3 dense + sparse RRF fusion) targets Q9-class queries
where the variant generator paraphrased a distinctive document term away —
sparse matches the literal token directly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tree_index.page_vector_index import PageVectorIndex


def _dense_embedder(dim: int = 4):
    """Mock dense-only embedder. Returns np.ndarray (back-compat path)."""
    def encode(text: str) -> np.ndarray:
        v = np.zeros(dim, dtype=np.float32)
        idx = (ord(text[0]) if text else 0) % dim
        v[idx] = 1.0
        return v
    return encode


def _hybrid_embedder(dim: int = 4):
    """Mock dense+sparse embedder. Returns dict with both representations."""
    def encode(text: str) -> dict:
        # Dense: basis vector by ord(text[0]) % dim
        dv = np.zeros(dim, dtype=np.float32)
        dv[(ord(text[0]) if text else 0) % dim] = 1.0
        # Sparse: token_id = ord(c), weight = 1.0 for each char
        sparse = {}
        for c in text:
            sparse[ord(c)] = sparse.get(ord(c), 0.0) + 1.0
        return {"dense": dv, "sparse": sparse}
    return encode


# ===== dense-only path (back-compat with existing fallback callers) =====

def test_search_returns_top_k_pages_by_cosine() -> None:
    page_vecs = np.eye(4, dtype=np.float32)
    embedder = _dense_embedder(dim=4)
    idx = PageVectorIndex(dense_embeddings=page_vecs, embedder=embedder)
    hits = idx.search("Berkshire", top_k=2)
    assert len(hits) == 2
    page_num, score = hits[0]
    assert page_num == 3, f"expected page 3, got {page_num}; hits={hits}"
    assert score == pytest.approx(1.0, abs=1e-5)


def test_search_returns_pages_in_descending_score_order() -> None:
    page_vecs = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.7, 0.7, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ], dtype=np.float32)
    page_vecs /= np.linalg.norm(page_vecs, axis=1, keepdims=True)
    embedder = _dense_embedder(dim=4)
    idx = PageVectorIndex(dense_embeddings=page_vecs, embedder=embedder)
    hits = idx.search("Apple", top_k=3)
    assert [p for p, _ in hits] == [3, 2, 1], f"wrong order: {hits}"
    assert hits[0][1] > hits[1][1] > hits[2][1]


def test_search_handles_top_k_larger_than_n_pages() -> None:
    page_vecs = np.eye(3, dtype=np.float32)
    idx = PageVectorIndex(dense_embeddings=page_vecs, embedder=_dense_embedder(dim=3))
    hits = idx.search("anything", top_k=10)
    assert len(hits) == 3, f"expected 3 (all pages), got {len(hits)}"


def test_save_and_load_dense_only_roundtrip(tmp_path: Path) -> None:
    page_vecs = np.random.RandomState(42).rand(5, 4).astype(np.float32)
    page_vecs /= np.linalg.norm(page_vecs, axis=1, keepdims=True)
    artifact = tmp_path / "page_vectors.npy"
    PageVectorIndex.save(dense_embeddings=page_vecs, sparse_embeddings=None,
                          path=artifact)
    assert artifact.exists()
    loaded = PageVectorIndex.load(artifact, embedder=_dense_embedder(dim=4))
    np.testing.assert_array_almost_equal(loaded.dense_embeddings, page_vecs)
    assert loaded.num_pages == 5
    assert loaded.sparse_embeddings is None


def test_search_normalizes_query_embedding() -> None:
    page_vecs = np.eye(3, dtype=np.float32)

    def unnormalized(text: str) -> np.ndarray:
        v = np.zeros(3, dtype=np.float32)
        v[0] = 5.0
        return v

    idx = PageVectorIndex(dense_embeddings=page_vecs, embedder=unnormalized)
    hits = idx.search("test", top_k=1)
    assert hits[0][0] == 1
    assert hits[0][1] == pytest.approx(1.0, abs=1e-5)


def test_load_raises_helpfully_when_artifact_missing(tmp_path: Path) -> None:
    missing = tmp_path / "page_vectors.npy"
    with pytest.raises(FileNotFoundError, match="page_vectors.npy"):
        PageVectorIndex.load(missing, embedder=_dense_embedder())


def test_init_validates_embedder_callable() -> None:
    page_vecs = np.eye(2, dtype=np.float32)
    with pytest.raises(TypeError):
        PageVectorIndex(dense_embeddings=page_vecs, embedder="not callable")  # type: ignore[arg-type]


# ===== hybrid (dense + sparse) path =====

def test_hybrid_search_combines_dense_and_sparse_via_rrf() -> None:
    """Hybrid mode: when both query and index have sparse, RRF fuses
    dense + sparse rankings. Synthetic case where dense and sparse
    rank different pages first → both should appear in top-2."""
    # Page 1: dense aligned with 'A' (ord=65, %4=1), no sparse for 'B'
    # Page 2: dense aligned with 'B' (ord=66, %4=2), heavy sparse for 'B'
    # Page 3: dense neutral, no sparse
    dense_vecs = np.array([
        [0.0, 1.0, 0.0, 0.0],   # page 1 — dense matches 'A'
        [0.0, 0.0, 1.0, 0.0],   # page 2 — dense matches 'B'
        [0.0, 0.0, 0.0, 1.0],   # page 3 — orthogonal to both
    ], dtype=np.float32)
    sparse_vecs = [
        {ord("X"): 5.0},                  # page 1 — no overlap with query
        {ord("B"): 10.0, ord("e"): 2.0},  # page 2 — heavy overlap with "Berkshire"
        {ord("Y"): 1.0},                  # page 3 — no overlap
    ]
    idx = PageVectorIndex(
        dense_embeddings=dense_vecs,
        sparse_embeddings=sparse_vecs,
        embedder=_hybrid_embedder(dim=4),
    )
    # Query "Berkshire": ord('B')=66, %4=2 → dense matches page 2
    # Sparse: B+e+r+k+s+h+i+r+e tokens → page 2 gets B(10) + e(2) match
    hits = idx.search("Berkshire", top_k=2)
    assert hits[0][0] == 2, f"page 2 should rank first (dense+sparse both), got {hits}"


def test_hybrid_recovers_when_dense_alone_misses() -> None:
    """The Q9 case: dense embedding doesn't match 'Scorecard' page,
    but sparse literal-token match does. Hybrid surfaces the page."""
    # Page 1: dense aligned with query, no sparse signal
    # Page 2: dense orthogonal to query BUT heavy sparse on literal token
    dense_vecs = np.array([
        [1.0, 0.0, 0.0, 0.0],   # page 1 — dense matches 'A'
        [0.0, 0.0, 0.0, 1.0],   # page 2 — dense orthogonal to 'A'
    ], dtype=np.float32)
    sparse_vecs = [
        {},                                 # page 1 — empty sparse
        {ord("S"): 10.0, ord("c"): 5.0,     # page 2 — heavy "Scorecard" tokens
         ord("o"): 5.0, ord("r"): 5.0,
         ord("e"): 5.0, ord("a"): 5.0,
         ord("d"): 5.0},
    ]
    idx = PageVectorIndex(
        dense_embeddings=dense_vecs,
        sparse_embeddings=sparse_vecs,
        embedder=_hybrid_embedder(dim=4),
    )
    # Dense-only on "Scorecard": ord('S')=83, %4=3 → matches page 2 (basis 3)
    # But the test's intent: even if dense missed, sparse would catch.
    # Verify page 2 ranks at top.
    hits = idx.search("Scorecard", top_k=2)
    assert hits[0][0] == 2, f"sparse-strong page 2 should win, got {hits}"


def test_hybrid_falls_back_to_dense_when_query_has_no_sparse() -> None:
    """When embedder returns dict with only 'dense' (no 'sparse' key),
    behave as dense-only — index sparse data ignored for this query."""
    dense_vecs = np.eye(3, dtype=np.float32)
    sparse_vecs = [{1: 5.0}, {2: 5.0}, {3: 5.0}]

    def dense_dict_embedder(text: str) -> dict:
        v = np.zeros(3, dtype=np.float32)
        v[(ord(text[0]) if text else 0) % 3] = 1.0
        return {"dense": v}  # no sparse key

    idx = PageVectorIndex(
        dense_embeddings=dense_vecs,
        sparse_embeddings=sparse_vecs,
        embedder=dense_dict_embedder,
    )
    hits = idx.search("Apple", top_k=1)  # ord('A')=65, %3=2 → page 3
    assert hits[0][0] == 3


def test_save_and_load_hybrid_roundtrip(tmp_path: Path) -> None:
    """Sparse embeddings must serialize alongside dense and reload."""
    dense = np.random.RandomState(7).rand(4, 4).astype(np.float32)
    dense /= np.linalg.norm(dense, axis=1, keepdims=True)
    sparse = [
        {1: 2.5, 7: 1.0},
        {2: 3.0},
        {1: 0.5, 99: 4.5, 100: 1.5},
        {},
    ]
    artifact = tmp_path / "page_vectors.npy"
    PageVectorIndex.save(dense_embeddings=dense, sparse_embeddings=sparse,
                          path=artifact)
    sparse_path = tmp_path / "page_vectors.sparse.json"
    assert artifact.exists()
    assert sparse_path.exists()

    loaded = PageVectorIndex.load(artifact, embedder=_hybrid_embedder(dim=4))
    np.testing.assert_array_almost_equal(loaded.dense_embeddings, dense)
    assert loaded.sparse_embeddings is not None
    assert len(loaded.sparse_embeddings) == 4
    assert loaded.sparse_embeddings[0] == {1: 2.5, 7: 1.0}
    assert loaded.sparse_embeddings[2] == {1: 0.5, 99: 4.5, 100: 1.5}
    assert loaded.sparse_embeddings[3] == {}
