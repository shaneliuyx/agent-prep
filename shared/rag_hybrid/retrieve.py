"""Retriever — auto-detects hybrid vs dense, fuses with RRF, optional rerank.

Replaces the inline retrieval logic in lab-02-rerank-compress/src/retrieve.py
and the retrieval portions of various eval scripts. Caller picks a collection
name; Retriever inspects the schema and routes to the right path:

  - dense collection → BGE-M3 dense kNN → top_n candidates
  - hybrid collection → dense + sparse one-pass encode →
      query both named vectors → RRF fuse → top_n candidates

The fusion strategy is configurable. Manual (Python-side) RRF is the default
for parity with the current retrieve.py callsite — its frozen results were
generated this way. Native (Qdrant Prefetch + FusionQuery) is the
production-correct default for new code: one HTTP roundtrip instead of two.
Switch via ``RetrieveConfig(fusion=FusionStrategy.NATIVE_RRF)``.

Reranking is composed externally — the caller passes a
:class:`CrossEncoderReranker` instance. This keeps Retriever testable
without 2 GB of cross-encoder weights and matches the open / closed
boundary the existing code already uses.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from qdrant_client.models import (
    Fusion,
    FusionQuery,
    NamedSparseVector,
    NamedVector,
    Prefetch,
    SparseVector,
)

from .encoder import DenseEncoder, HybridEncoder
from .fusion import FusionStrategy, RRF_DEFAULT_K, rrf_fuse
from .schema import detect_collection_mode

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

    from .rerank import CrossEncoderReranker


@dataclass(frozen=True)
class RetrieveConfig:
    """Retrieval tunables.

    ``fusion`` selects RRF impl. MANUAL_RRF preserves the current retrieve.py
    behavior (Python-side rank merge across two HTTP calls); NATIVE_RRF uses
    Qdrant 1.10+'s server-side ``Prefetch + FusionQuery`` (one HTTP call,
    same RRF formula).

    ``top_n`` is the candidate budget — how many points come back from
    retrieval before optional reranking. Defaults to 50 (the cross-encoder
    latency budget; raising past 50 risks p95 > 200 ms on MPS).
    """
    fusion: FusionStrategy = FusionStrategy.MANUAL_RRF
    rrf_k: int = RRF_DEFAULT_K
    top_n: int = 50


class Retriever:
    """Vector retrieval over a Qdrant collection. Auto-detects hybrid/dense.

    Construction discovers the collection's mode once (cheap — one HTTP call
    to ``get_collection``). Every subsequent ``.search()`` call uses the
    cached mode.

    For hybrid collections, an encoder with ``supports_sparse=True`` is
    required. For dense, either encoder works.
    """

    def __init__(
        self,
        qd: "QdrantClient",
        collection: str,
        encoder: HybridEncoder | DenseEncoder,
        *,
        cfg: RetrieveConfig | None = None,
        mode: str | None = None,
        dense_vector_name: str = "dense",
        sparse_vector_name: str = "sparse",
    ) -> None:
        self.qd = qd
        self.collection = collection
        self.encoder = encoder
        self.cfg = cfg or RetrieveConfig()
        self.mode = mode or detect_collection_mode(qd, collection)
        self.dense_name = dense_vector_name
        self.sparse_name = sparse_vector_name

        if self.mode == "hybrid" and not isinstance(encoder, HybridEncoder):
            raise TypeError(
                f"hybrid collection {collection!r} requires HybridEncoder; "
                f"got {type(encoder).__name__}"
            )

    def search(self, query: str, *, top_n: int | None = None) -> list[Any]:
        """Return up to ``top_n`` candidate ScoredPoints, ordered by relevance.

        For hybrid mode, the relevance score is RRF-fused across dense and
        sparse rankings. For dense mode it's the raw cosine similarity from
        the dense vector index.
        """
        n = top_n if top_n is not None else self.cfg.top_n
        if self.mode == "hybrid":
            return self._search_hybrid(query, n)
        return self._search_dense(query, n)

    def search_with_rerank(
        self,
        query: str,
        reranker: "CrossEncoderReranker",
        *,
        k: int = 5,
        top_n: int | None = None,
    ) -> list[tuple[str, str, float]]:
        """Two-stage retrieval: vector search → cross-encoder rerank → top-k.

        Composition helper. Caller can do this manually if they want to inspect
        the candidate list between stages."""
        candidates = self.search(query, top_n=top_n)
        return reranker.rerank(query, candidates, top_k=k)

    # ------------------ private dispatch ------------------

    def _search_dense(self, query: str, top_n: int) -> list[Any]:
        prefix = self.encoder.cfg.spec.query_prefix
        text = (prefix + query) if prefix else query

        if isinstance(self.encoder, HybridEncoder):
            out = self.encoder.encode([text], with_sparse=False)
            qv = out["dense_vecs"][0].tolist()
        else:
            qv = self.encoder.encode([text]).tolist()[0]

        return self.qd.query_points(
            self.collection,
            query=qv,
            limit=top_n,
            with_payload=True,
        ).points

    def _search_hybrid(self, query: str, top_n: int) -> list[Any]:
        assert isinstance(self.encoder, HybridEncoder)  # construction-time guard
        out = self.encoder.encode([query], with_sparse=True)
        dense_vec = out["dense_vecs"][0].tolist()
        sparse_dict = out["lexical_weights"][0]
        sparse_q = SparseVector(
            indices=list(map(int, sparse_dict.keys())),
            values=list(map(float, sparse_dict.values())),
        )

        if self.cfg.fusion is FusionStrategy.NATIVE_RRF:
            return self._search_hybrid_native(dense_vec, sparse_q, top_n)
        return self._search_hybrid_manual(dense_vec, sparse_q, top_n)

    def _search_hybrid_native(
        self, dense_vec: list[float], sparse_q: SparseVector, top_n: int
    ) -> list[Any]:
        return self.qd.query_points(
            self.collection,
            prefetch=[
                Prefetch(query=dense_vec, using=self.dense_name, limit=top_n),
                Prefetch(query=sparse_q, using=self.sparse_name, limit=top_n),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_n,
            with_payload=True,
        ).points

    def _search_hybrid_manual(
        self, dense_vec: list[float], sparse_q: SparseVector, top_n: int
    ) -> list[Any]:
        dense_pts = self.qd.query_points(
            self.collection,
            query=NamedVector(name=self.dense_name, vector=dense_vec),
            limit=top_n,
            with_payload=True,
        ).points
        sparse_pts = self.qd.query_points(
            self.collection,
            query=NamedSparseVector(name=self.sparse_name, vector=sparse_q),
            limit=top_n,
            with_payload=True,
        ).points
        # rrf_fuse returns docs sorted by aggregate score; truncate to top_n
        # (each ranking already capped at top_n, so fused list has at most
        # 2 * top_n unique docs).
        return rrf_fuse([dense_pts, sparse_pts], k=self.cfg.rrf_k)[:top_n]
