"""Atomic model and collection specs for lab-02.

The principle: things that must stay in sync travel as one record.
Same pattern as lab-01/src/model_config.py — extended for what lab-02 adds.

Lab-02 introduces:
- BGE-M3 hybrid mode (one forward pass → dense + sparse outputs)
- BGE-reranker-v2-m3 (cross-encoder for two-stage retrieval)
- BEIR-FiQA-2018 corpus (harder benchmark, breaks the MS MARCO recall ceiling)

New spec types beyond lab-01:
- RerankerSpec — wraps a cross-encoder (path, batch_size, candidate budget)
- HybridCollectionSpec — Qdrant collection with named dense + sparse vectors
"""
import os
from dataclasses import dataclass
from qdrant_client.http.models import Distance

HOME = os.path.expanduser("~")


@dataclass(frozen=True)
class EmbedModelSpec:
    """Bi-encoder embedding model. One forward pass produces vectors for the index."""
    name: str
    path: str
    dim: int
    distance: Distance
    doc_prefix: str = ""
    query_prefix: str = ""
    trust_remote_code: bool = False
    # NEW for lab-02 — only models with supports_sparse=True can be paired with HybridCollectionSpec
    supports_sparse: bool = False


@dataclass(frozen=True)
class RerankerSpec:
    """Cross-encoder reranker. Takes (query, doc) pairs, returns relevance scores.

    Latency budget reasoning (see Week 2 Theory Concept 2):
      - p95 ≤ 200ms is the practical ceiling for synchronous RAG
      - BGE-reranker-v2-m3 at batch_size=32 over 50 candidates → 80-140ms on MPS
      - Cutting candidates 50→30 saves ~40ms; raising batch_size risks MPS OOM
    """
    name: str
    path: str
    batch_size: int = 32          # MPS-friendly default; 16 for tight memory, 64 for short passages
    max_pairs_per_query: int = 50  # candidate budget — tightens latency at the cost of recall


@dataclass(frozen=True)
class HybridCollectionSpec:
    """Qdrant collection with BOTH dense and sparse named vectors.

    Single-collection hybrid retrieval: one storage layer, two vector representations,
    queryable separately or fused via RRF (Cormack et al. 2009, k=60 default).
    """
    name: str
    model: EmbedModelSpec               # MUST have supports_sparse=True (asserted by ingest scripts)
    dense_vector_name: str = "dense"
    sparse_vector_name: str = "sparse"
    hnsw_m: int = 16
    hnsw_ef_construct: int = 128


@dataclass(frozen=True)
class CollectionSpec:
    """Standard single-vector Qdrant collection (matches lab-01's CollectionSpec)."""
    name: str
    model: EmbedModelSpec
    hnsw_m: int = 16
    hnsw_ef_construct: int = 128
    source_collection: str | None = None  # if set, ingest copies vectors from here


@dataclass(frozen=True)
class ChunkSweepCollectionSpec:
    """Collection for the chunking sweep — same model, varying chunk_size + overlap.

    Each sweep variant gets its own collection so eval can compare them on identical queries.
    Name follows the convention `sweep_s{chunk_size}_o{overlap}` for grep-friendly results.
    """
    name: str
    model: EmbedModelSpec
    chunk_size: int
    overlap: int
    hnsw_m: int = 16
    hnsw_ef_construct: int = 128


# --- Models ---

BGE_M3 = EmbedModelSpec(
    name="bge-m3",
    path=f"{HOME}/models/bge-m3",
    dim=1024,
    distance=Distance.COSINE,
    supports_sparse=True,   # NEW for lab-02 — unlocks hybrid mode
    # BGE-M3 handles bare queries — no asymmetric prefix needed
)

# Cross-encoder reranker — added in lab-02 for two-stage retrieval
BGE_RERANKER_V2_M3 = RerankerSpec(
    name="bge-reranker-v2-m3",
    path=f"{HOME}/models/bge-reranker-v2-m3",
    batch_size=32,
    max_pairs_per_query=50,
)


# --- Collections ---

# Hybrid collection — MS MARCO 10K with dense + sparse named vectors
BGE_M3_HYBRID = HybridCollectionSpec(
    name="bge_m3_hybrid",
    model=BGE_M3,
)

# Reused from lab-01 (the dense-only baseline collection — used as upstream for the reranker)
BGE_M3_HNSW = CollectionSpec(
    name="bge_m3_hnsw",
    model=BGE_M3,
    hnsw_m=16,
    hnsw_ef_construct=128,
)

# BEIR-FiQA-2018 — harder benchmark, same encoder, different corpus
FIQA_HYBRID = HybridCollectionSpec(
    name="fiqa_hybrid",
    model=BGE_M3,
)

ALL_COLLECTIONS = (BGE_M3_HYBRID, BGE_M3_HNSW, FIQA_HYBRID)


# --- Chunking sweep (Phase 4): 3 chunk sizes × 3 overlaps = 9 collections ---

CHUNK_SIZES = (256, 512, 1024)
CHUNK_OVERLAPS = (0, 64, 128)

SWEEP_VARIANTS = tuple(
    ChunkSweepCollectionSpec(
        name=f"sweep_s{size}_o{overlap}",
        model=BGE_M3,
        chunk_size=size,
        overlap=overlap,
    )
    for size in CHUNK_SIZES
    for overlap in CHUNK_OVERLAPS
)
