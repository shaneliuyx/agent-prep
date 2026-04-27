"""Atomic model and collection specs.

The principle: things that must stay in sync travel as one record.

An embedding model is defined by (path, dim, distance, prefixes, trust_remote_code) —
drift between any two of these is a silent-failure surface (wrong prefix -> noise vectors;
wrong dim -> wrong collection schema; wrong trust_remote_code -> stale cached modeling
code). A collection is defined by (name, model, hnsw config) — and optionally a source
collection if it copies vectors from another instead of encoding fresh.

Adding a new model is a one-record addition here, not an N-file diff.
"""
import os
from dataclasses import dataclass
from qdrant_client.http.models import Distance

HOME = os.path.expanduser("~")


@dataclass(frozen=True)
class EmbedModelSpec:
    name: str
    path: str
    dim: int
    distance: Distance
    doc_prefix: str = ""
    query_prefix: str = ""
    trust_remote_code: bool = False


@dataclass(frozen=True)
class CollectionSpec:
    name: str
    model: EmbedModelSpec
    hnsw_m: int = 16
    hnsw_ef_construct: int = 128
    source_collection: str | None = None  # if set, ingest copies vectors from this collection


# --- Models ----------------------------------------------------------------

BGE_M3 = EmbedModelSpec(
    name="bge-m3",
    path=f"{HOME}/models/bge-m3",
    dim=1024,
    distance=Distance.COSINE,
    # BGE-M3 handles bare queries — no asymmetric prefix needed
)

NOMIC_V2 = EmbedModelSpec(
    name="nomic-embed-text-v2-moe",
    path=f"{HOME}/models/nomic-embed-v2",
    dim=768,
    distance=Distance.COSINE,
    # Nomic v2 requires asymmetric prefixes — silent failure if these don't match the
    # prefixes the model was trained with. Cache-staleness can also produce silent failure
    # of a different kind; see Phase 3.2 gotcha #4 in the Week 1 runbook.
    doc_prefix="search_document: ",
    query_prefix="search_query: ",
    trust_remote_code=True,
)


# --- Collections -----------------------------------------------------------

BGE_M3_HNSW = CollectionSpec(
    name="bge_m3_hnsw",
    model=BGE_M3,
    hnsw_m=16,
    hnsw_ef_construct=128,
)

BGE_M3_HNSW_FAST = CollectionSpec(
    name="bge_m3_hnsw_fast",
    model=BGE_M3,                      # same model -> same dim, same prefixes, same encoder cache
    hnsw_m=8,
    hnsw_ef_construct=64,
    source_collection="bge_m3_hnsw",   # ingest copies from here instead of re-encoding
)

NOMIC_HNSW = CollectionSpec(
    name="nomic_hnsw",
    model=NOMIC_V2,
    hnsw_m=16,
    hnsw_ef_construct=128,
)

ALL_COLLECTIONS = (BGE_M3_HNSW, BGE_M3_HNSW_FAST, NOMIC_HNSW)
