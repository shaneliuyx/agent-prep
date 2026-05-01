"""Ingest the lab-02.5 tech-founders corpus into a hybrid Qdrant collection
(dense + sparse named vectors per point).

This is the upgrade path from `ingest_to_vector.py` (dense-only). The hybrid
collection enables sparse + dense + RRF retrieval — the production-grade
vector pipeline that catches lexical-match edge cases (rare proper nouns,
acronyms, exact phrases) which dense embeddings alone tend to miss.

Why a separate ingest script:
  - Different collection schema (BGE_M3_HYBRID vs BGE_M3_HNSW). Two named
    vectors per point ('dense' + 'sparse') vs one.
  - Different encoder API path. BGEM3FlagModel.encode(return_dense=True,
    return_sparse=True) emits both in one forward pass; SentenceTransformer
    only does dense.
  - Keeps the dense-only collection alive for backwards-compatible callers.

Run from the lab-02-5-graphrag directory:
  ./.venv/bin/python src/ingest_to_vector_hybrid.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Reuse the W2 model + hybrid-collection spec.
sys.path.insert(0, "../lab-02-rerank-compress/src")
from model_config import BGE_M3, BGE_M3_HYBRID  # noqa: E402

from FlagEmbedding import BGEM3FlagModel  # noqa: E402
from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.models import (  # noqa: E402
    PointStruct, SparseVector, SparseVectorParams, VectorParams,
)

# New collection — same hybrid schema as W2's bge_m3_hybrid (dense 1024-d
# cosine + sparse named vectors) but a different name so this script
# does not collide with the existing financial-passages collection.
COLLECTION_NAME = "tech_corpus_hybrid"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
ENCODE_BATCH = 64
UPSERT_BATCH = 256


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping windows. Same chunker as ingest_to_vector.py."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    step = size - overlap
    for i in range(0, len(text), step):
        chunk = text[i : i + size]
        if chunk:
            chunks.append(chunk)
        if i + size >= len(text):
            break
    return chunks


def main() -> None:
    corpus = json.loads(Path("data/corpus.json").read_text())
    print(f"Loaded {len(corpus)} articles from data/corpus.json")

    passages: list[tuple[str, str, str]] = []
    for art in corpus:
        for i, chunk in enumerate(chunk_text(art["text"])):
            passages.append((chunk, f"{art['id']}_{i}", art["title"]))
    print(f"Chunked into {len(passages)} passages "
          f"({len(passages) / len(corpus):.1f} per article on average)")

    qd = QdrantClient(url="http://127.0.0.1:6333", timeout=60)
    qd.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            BGE_M3_HYBRID.dense_vector_name: VectorParams(
                size=BGE_M3.dim, distance=BGE_M3.distance,
            ),
        },
        sparse_vectors_config={
            BGE_M3_HYBRID.sparse_vector_name: SparseVectorParams(),
        },
    )
    print(
        f"Created collection {COLLECTION_NAME!r} (dense dim={BGE_M3.dim}, "
        f"distance={BGE_M3.distance}, +sparse named vector)"
    )

    # BGE-M3 in hybrid mode emits dense + sparse from one forward pass.
    # use_fp16=False matches W2's 00_hybrid_ingest.py — keeps numerical
    # stability on MPS where fp16 occasionally produces NaN sparse weights.
    model = BGEM3FlagModel(BGE_M3.path, use_fp16=False, device="mps")

    t0 = time.time()
    points: list[PointStruct] = []
    for batch_start in range(0, len(passages), ENCODE_BATCH):
        batch = passages[batch_start : batch_start + ENCODE_BATCH]
        texts = [p[0] for p in batch]
        out = model.encode(
            texts,
            batch_size=ENCODE_BATCH,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense_vecs = out["dense_vecs"]              # (B, 1024)
        sparse_dicts = out["lexical_weights"]       # list of {token_id: weight}
        for j, (text, doc_id, title) in enumerate(batch):
            sparse = SparseVector(
                indices=list(map(int,   sparse_dicts[j].keys())),
                values =list(map(float, sparse_dicts[j].values())),
            )
            points.append(PointStruct(
                id=batch_start + j,
                vector={
                    BGE_M3_HYBRID.dense_vector_name:  dense_vecs[j].tolist(),
                    BGE_M3_HYBRID.sparse_vector_name: sparse,
                },
                payload={"doc_id": doc_id, "text": text, "article_title": title},
            ))
        if (batch_start // ENCODE_BATCH) % 10 == 0:
            done = batch_start + len(batch)
            print(f"  encoded {done}/{len(passages)} ({time.time() - t0:.0f}s)")
    print(f"Encoding done in {time.time() - t0:.0f}s")

    t0 = time.time()
    for i in range(0, len(points), UPSERT_BATCH):
        qd.upsert(COLLECTION_NAME, points[i : i + UPSERT_BATCH])
    print(f"Upsert done in {time.time() - t0:.0f}s")

    info = qd.get_collection(COLLECTION_NAME)
    print(f"Collection {COLLECTION_NAME!r}: {info.points_count} points")


if __name__ == "__main__":
    main()
