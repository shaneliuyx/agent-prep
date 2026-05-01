"""Qdrant collection creation — dispatch on spec type.

Three callsites in the labs each hand-roll qd.recreate_collection() with
slightly different argument shapes (named vectors for hybrid, single
VectorParams for dense). Centralizing here removes the drift surface and
gives one place to add HNSW tuning when autoconfig probes the corpus size.

The labs always recreate (drop and replace). Pass ``recreate=False`` to
preserve existing data — useful for incremental ingest, not used today.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from qdrant_client.models import (
    SparseVectorParams,
    VectorParams,
)

from .models import CollectionSpec, HybridCollectionSpec

if TYPE_CHECKING:
    from qdrant_client import QdrantClient


def ensure_collection(
    qd: "QdrantClient",
    spec: CollectionSpec | HybridCollectionSpec,
    *,
    recreate: bool = True,
) -> None:
    """Create or replace the collection per the spec.

    ``recreate=True`` (default) drops any existing data — matches every
    existing lab ingest script. ``recreate=False`` creates only if missing
    (raises if collection exists with a different schema; qdrant's check).

    Dispatches on spec type:
      - HybridCollectionSpec → dense + sparse named vectors
      - CollectionSpec → single dense vector
    """
    if isinstance(spec, HybridCollectionSpec):
        _ensure_hybrid(qd, spec, recreate)
        return
    if isinstance(spec, CollectionSpec):
        _ensure_dense(qd, spec, recreate)
        return
    raise TypeError(
        f"unsupported spec type: {type(spec).__name__}. "
        f"Expected CollectionSpec or HybridCollectionSpec."
    )


def _ensure_dense(qd: "QdrantClient", spec: CollectionSpec, recreate: bool) -> None:
    m = spec.model
    params = VectorParams(size=m.dim, distance=m.distance)
    if recreate:
        qd.recreate_collection(collection_name=spec.name, vectors_config=params)
    else:
        existing = {c.name for c in qd.get_collections().collections}
        if spec.name not in existing:
            qd.create_collection(collection_name=spec.name, vectors_config=params)


def _ensure_hybrid(qd: "QdrantClient", spec: HybridCollectionSpec, recreate: bool) -> None:
    m = spec.model
    if not m.supports_sparse:
        raise ValueError(
            f"HybridCollectionSpec {spec.name!r} references model {m.name!r} "
            f"which has supports_sparse=False. Mark the spec sparse-capable or "
            f"use a CollectionSpec for dense-only."
        )
    vectors_config = {spec.dense_vector_name: VectorParams(size=m.dim, distance=m.distance)}
    sparse_config = {spec.sparse_vector_name: SparseVectorParams()}
    if recreate:
        qd.recreate_collection(
            collection_name=spec.name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_config,
        )
    else:
        existing = {c.name for c in qd.get_collections().collections}
        if spec.name not in existing:
            qd.create_collection(
                collection_name=spec.name,
                vectors_config=vectors_config,
                sparse_vectors_config=sparse_config,
            )


def detect_collection_mode(qd: "QdrantClient", name: str) -> str:
    """Return ``"hybrid"`` if collection has a sparse_vectors_config, else
    ``"dense"``. Used by Retriever to auto-route. Mirrors the auto-detection
    already present in retrieve.py — the seam that makes
    ``QDRANT_COLLECTION=...`` environment-driven mode selection work."""
    info = qd.get_collection(name)
    sparse_cfg = getattr(info.config.params, "sparse_vectors", None)
    return "hybrid" if sparse_cfg else "dense"
