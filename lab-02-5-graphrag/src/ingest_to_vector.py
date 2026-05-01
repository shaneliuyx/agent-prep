"""Ingest the lab-02.5 tech-founders corpus into a fresh Qdrant collection.

Why this script exists: lab-02.5's compare.py runs both GraphRAG and VectorRAG
on the same eval set, but lab-02's existing Qdrant collection (`bge_m3_hnsw`)
holds MS MARCO financial passages — completely unrelated to the tech-founders
graph. Without ingesting the same corpus into both backends, VectorRAG returns
"insufficient context" on every query and the head-to-head is meaningless.

This script:
  1. Reads data/corpus.json (the 400-article tech corpus, same input the
     graph build consumed).
  2. Chunks each article into 512-char passages with 64-char overlap. The
     overlap preserves entity mentions that straddle chunk boundaries
     (e.g. "Steve Jobs co-founded Apple Inc. in 1976" — splitting between
     "Apple" and "Inc." would break the entity for retrieval).
  3. Encodes passages with BGE-M3 (dense-only, same model as W2's
     bge_m3_hnsw) and upserts into a NEW collection `tech_corpus_hnsw`.
     Different name so this script does not overwrite W2's existing
     financial-passages collection.

Run from the lab-02-5-graphrag directory:
  ./.venv/bin/python src/ingest_to_vector.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Reuse the W2 model + collection spec definitions.
sys.path.insert(0, "../lab-02-rerank-compress/src")
from model_config import BGE_M3  # noqa: E402

from FlagEmbedding import BGEM3FlagModel  # noqa: E402
from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.models import PointStruct, VectorParams  # noqa: E402

# New collection — same schema as W2's BGE_M3_HNSW (dense-only, cosine
# distance) but a different name so this script does not collide with
# the existing financial-passages collection.
COLLECTION_NAME = "tech_corpus_hnsw"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
ENCODE_BATCH = 64  # smaller than W2's 128 because passages are shorter
UPSERT_BATCH = 256


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping windows. Returns list of chunks; an empty
    or whitespace-only article returns an empty list."""
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

    # Build (passage_text, doc_id, article_title) tuples for every chunk.
    passages: list[tuple[str, str, str]] = []
    for art in corpus:
        for i, chunk in enumerate(chunk_text(art["text"])):
            passages.append((chunk, f"{art['id']}_{i}", art["title"]))
    print(f"Chunked into {len(passages)} passages "
          f"({len(passages) / len(corpus):.1f} per article on average)")

    qd = QdrantClient(url="http://127.0.0.1:6333", timeout=60)
    qd.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=BGE_M3.dim, distance=BGE_M3.distance),
    )
    print(f"Created collection {COLLECTION_NAME!r} (dim={BGE_M3.dim}, distance={BGE_M3.distance})")

    model = BGEM3FlagModel(BGE_M3.path, use_fp16=False, device="mps")

    # Encode in batches.
    t0 = time.time()
    points: list[PointStruct] = []
    for batch_start in range(0, len(passages), ENCODE_BATCH):
        batch = passages[batch_start : batch_start + ENCODE_BATCH]
        texts = [p[0] for p in batch]
        out = model.encode(
            texts,
            batch_size=ENCODE_BATCH,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        dense = out["dense_vecs"]
        for j, (text, doc_id, title) in enumerate(batch):
            points.append(PointStruct(
                id=batch_start + j,
                vector=dense[j].tolist(),
                payload={"doc_id": doc_id, "text": text, "article_title": title},
            ))
        if (batch_start // ENCODE_BATCH) % 10 == 0:
            done = batch_start + len(batch)
            print(f"  encoded {done}/{len(passages)} ({time.time() - t0:.0f}s)")
    print(f"Encoding done in {time.time() - t0:.0f}s")

    # Upsert in chunks.
    t0 = time.time()
    for i in range(0, len(points), UPSERT_BATCH):
        qd.upsert(COLLECTION_NAME, points[i : i + UPSERT_BATCH])
    print(f"Upsert done in {time.time() - t0:.0f}s")

    info = qd.get_collection(COLLECTION_NAME)
    print(f"Collection {COLLECTION_NAME!r}: {info.points_count} points")


if __name__ == "__main__":
    main()
