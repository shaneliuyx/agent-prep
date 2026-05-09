"""PageVectorIndex — dense + optional sparse hybrid match over per-page text.

Used by AgenticTreeRetriever as a chunk-level fallback when the agent loop
hits max_iterations and returns a refusal. Recovers from cases where the
structural navigation missed an answer that exists on a specific page.

Two modes:
- **Dense-only**: cosine on BGE-M3 (or any) dense embeddings.
- **Hybrid (dense + sparse RRF)**: dense cosine + sparse dot-product +
  reciprocal rank fusion (k=60). Targets Q9-class regressions where the
  variant generator paraphrased a distinctive document term away — sparse
  matches the literal token directly.

BGE-M3 is the intended embedder. Its `encode(text, return_dense=True,
return_sparse=True)` returns both dense (1024-dim) + sparse (dict[token_id,
weight]). Single forward pass produces both representations — sparse is
essentially free if dense is already being computed.

Storage:
- Dense: `.npy` (numpy binary).
- Sparse: `.sparse.json` sibling file (only written when sparse provided).
- 200-page corpus × 1024-dim float32 dense ≈ 800 KB; sparse adds ~200 KB.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np


# Type alias for the embedder. Dense-only callers can return np.ndarray;
# hybrid callers return dict with at least 'dense' key, optionally 'sparse'.
Embedder = Callable[[str], "np.ndarray | dict[str, Any]"]

RRF_K = 60
OVER_RECALL_FACTOR = 3   # rank top_k * 3 per modality before fusion


class PageVectorIndex:
    """Cosine + sparse-dot-product hybrid match over per-page embeddings.

    Construction takes (num_pages, dim) dense matrix + optional list of
    sparse dicts (one per page). Dense rows are L2-normalized at
    construction time so query-time `dense @ q_dense` is cosine after
    query is also L2-normalized. Sparse uses raw dot product on shared
    tokens.

    Search returns top-K (page_num, score) sorted desc, with 1-indexed
    page numbers (matches PDF page convention). Ties broken by lower
    page number.
    """

    def __init__(
        self,
        dense_embeddings: np.ndarray,
        embedder: Embedder,
        sparse_embeddings: list[dict[int, float]] | None = None,
        rrf_k: int = RRF_K,
    ) -> None:
        if not callable(embedder):
            raise TypeError(
                f"embedder must be callable, got {type(embedder).__name__}"
            )
        emb = np.asarray(dense_embeddings, dtype=np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        self.dense_embeddings: np.ndarray = emb / np.maximum(norms, 1e-8)
        self.num_pages: int = self.dense_embeddings.shape[0]
        self.sparse_embeddings: list[dict[int, float]] | None = sparse_embeddings
        if (sparse_embeddings is not None
                and len(sparse_embeddings) != self.num_pages):
            raise ValueError(
                f"sparse_embeddings length ({len(sparse_embeddings)}) does "
                f"not match num_pages ({self.num_pages})"
            )
        self._embedder = embedder
        self._rrf_k = rrf_k

    def search(self, query: str, top_k: int = 3) -> list[tuple[int, float]]:
        """Return top-K (page_num, score) sorted desc.

        Mode selection:
        - If embedder returns np.ndarray → dense-only.
        - If embedder returns dict with 'dense' key only → dense-only
          (sparse data on the index ignored for this query).
        - If embedder returns dict with 'dense' AND 'sparse' AND the
          index has sparse_embeddings → hybrid via RRF fusion.
        """
        out = self._embedder(query)
        if isinstance(out, np.ndarray):
            q_dense = out
            q_sparse: dict[int, float] | None = None
        elif isinstance(out, dict):
            q_dense = np.asarray(out["dense"], dtype=np.float32)
            q_sparse = out.get("sparse")
        else:
            raise TypeError(
                f"embedder must return np.ndarray or dict; got {type(out).__name__}"
            )

        # Dense ranking (always computed)
        dense_ranking = self._rank_dense(q_dense)

        if q_sparse is None or self.sparse_embeddings is None:
            # Dense-only: take top_k directly
            k = min(top_k, len(dense_ranking))
            return [(int(idx) + 1, float(score))
                    for idx, score in dense_ranking[:k]]

        # Hybrid: rank sparse, then RRF-fuse with dense
        sparse_ranking = self._rank_sparse(q_sparse)
        return self._rrf_fuse(dense_ranking, sparse_ranking, top_k)

    def _rank_dense(self, q_dense: np.ndarray) -> list[tuple[int, float]]:
        """Cosine rank (descending). Returns [(page_idx, score), ...]
        for ALL pages — caller slices to top_k or fuses with sparse."""
        n = float(np.linalg.norm(q_dense))
        if n < 1e-8:
            return []
        q = q_dense / n
        scores = self.dense_embeddings @ q
        page_idx = np.arange(len(scores))
        # lexsort: primary -scores ascending = descending; secondary page_idx
        # ascending = lower page wins ties. Last key = primary in lexsort.
        order = np.lexsort((page_idx, -scores))
        return [(int(i), float(scores[i])) for i in order]

    def _rank_sparse(self, q_sparse: dict[int, float]) -> list[tuple[int, float]]:
        """Sparse dot-product rank. Returns [(page_idx, score), ...] desc."""
        assert self.sparse_embeddings is not None
        scores: list[float] = [
            sum(w_q * page_sparse.get(tok, 0.0) for tok, w_q in q_sparse.items())
            for page_sparse in self.sparse_embeddings
        ]
        scores_arr = np.array(scores, dtype=np.float32)
        page_idx = np.arange(len(scores_arr))
        order = np.lexsort((page_idx, -scores_arr))
        return [(int(i), float(scores_arr[i])) for i in order]

    def _rrf_fuse(
        self,
        dense_ranking: list[tuple[int, float]],
        sparse_ranking: list[tuple[int, float]],
        top_k: int,
    ) -> list[tuple[int, float]]:
        """Reciprocal Rank Fusion over dense + sparse rankings.

        Score = sum over modalities of 1 / (rrf_k + rank). Same formula
        as TREC 2009 RRF + the EntityIndex multi-query expansion in
        agentic.py. Over-recalls top_k * OVER_RECALL_FACTOR per modality
        to give RRF a meaningful pool to fuse.
        """
        pool = top_k * OVER_RECALL_FACTOR
        combined: dict[int, float] = {}
        for rank, (idx, _) in enumerate(dense_ranking[:pool]):
            combined[idx] = combined.get(idx, 0.0) + 1.0 / (self._rrf_k + rank)
        for rank, (idx, _) in enumerate(sparse_ranking[:pool]):
            combined[idx] = combined.get(idx, 0.0) + 1.0 / (self._rrf_k + rank)
        ranked = sorted(
            combined.items(),
            key=lambda kv: (-kv[1], kv[0]),  # score desc, then page asc tie-break
        )
        k = min(top_k, len(ranked))
        return [(idx + 1, float(score)) for idx, score in ranked[:k]]

    @staticmethod
    def _sparse_path(dense_path: Path) -> Path:
        return dense_path.with_suffix(".sparse.json")

    @staticmethod
    def save(
        dense_embeddings: np.ndarray,
        sparse_embeddings: list[dict[int, float]] | None,
        path: Path,
    ) -> None:
        """Persist dense to `path` (.npy) and sparse to `path.sparse.json`
        (when provided). Caller embeds + saves at index-build time.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.asarray(dense_embeddings, dtype=np.float32))
        if sparse_embeddings is not None:
            sparse_path = PageVectorIndex._sparse_path(path)
            # JSON requires string keys; convert int → str on write,
            # restore on load.
            serializable = [
                {str(tok): float(w) for tok, w in d.items()}
                for d in sparse_embeddings
            ]
            sparse_path.write_text(json.dumps(serializable), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, embedder: Embedder) -> "PageVectorIndex":
        """Load dense from `.npy` + sparse from `.sparse.json` (when
        present) and return a ready PageVectorIndex.

        Raises FileNotFoundError with a rebuild hint if the dense
        artifact is missing.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"page_vectors.npy not found at {path}. "
                f"Rebuild via: python lab-02-7-pageindex/src/build_summary_index.py"
            )
        dense = np.load(path)
        sparse: list[dict[int, float]] | None = None
        sparse_path = cls._sparse_path(path)
        if sparse_path.exists():
            raw = json.loads(sparse_path.read_text(encoding="utf-8"))
            sparse = [{int(k): float(v) for k, v in d.items()} for d in raw]
        return cls(
            dense_embeddings=dense,
            sparse_embeddings=sparse,
            embedder=embedder,
        )
