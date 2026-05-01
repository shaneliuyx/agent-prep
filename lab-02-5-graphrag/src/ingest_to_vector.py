"""Ingest the lab-02.5 tech-founders corpus into a dense-only Qdrant collection.

Library-driven version — uses shared/rag_hybrid for chunking, encoding, schema,
and upsert. The script's role is now just declaring WHAT to ingest (which
corpus, which chunker config, which collection); HOW to do it lives in
rag_hybrid.Ingestor.

Why this exists: lab-02.5's compare.py runs both GraphRAG and VectorRAG on
the same eval set. W2's ``bge_m3_hnsw`` holds MS MARCO financial passages —
unrelated to the tech-founders graph. This script populates a separate
collection (``tech_corpus_hnsw``) so VectorRAG has a fair head-to-head.

512-char chunks with 64-char overlap. Overlap preserves entities that
straddle a boundary (e.g. "Steve Jobs co-founded Apple Inc. in 1976" —
splitting between "Apple" and "Inc." would break entity match).

Run from the lab-02-5-graphrag directory:
  ./.venv/bin/python src/ingest_to_vector.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Add shared/ to sys.path — same pattern as W2.5's other cross-lab imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))

from qdrant_client import QdrantClient

from rag_hybrid import (  # noqa: E402
    BGE_M3,
    CollectionSpec,
    EncoderConfig,
    HybridEncoder,
    IngestConfig,
    Ingestor,
    char_window_chunks,
    chunk_corpus,
)

SPEC = CollectionSpec(name="tech_corpus_hnsw", model=BGE_M3)
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


def main() -> None:
    corpus = json.loads(Path("data/corpus.json").read_text())
    print(f"Loaded {len(corpus)} articles from data/corpus.json")

    payloads = chunk_corpus(
        corpus,
        chunker=lambda t: char_window_chunks(t, CHUNK_SIZE, CHUNK_OVERLAP),
        extra_payload=("title",),
    )
    # Rename "title" → "article_title" to match the payload schema used by
    # downstream graph queries and reranking. Cheap rename; cleaner than
    # threading the rename through chunk_corpus.
    for p in payloads:
        if "title" in p:
            p["article_title"] = p.pop("title")
    print(
        f"Chunked into {len(payloads)} passages "
        f"({len(payloads) / len(corpus):.1f} per article on average)"
    )

    qd = QdrantClient(url="http://127.0.0.1:6333", timeout=60)
    # Use HybridEncoder with sparse skip (matches original which used
    # BGEM3FlagModel(return_sparse=False)). Ingestor's dense path dispatches
    # on spec type, so HybridEncoder is fine for dense collections too.
    encoder = HybridEncoder(EncoderConfig(spec=BGE_M3, default_batch_size=64))
    ing = Ingestor(qd=qd, encoder=encoder, cfg=IngestConfig(encode_batch=64))
    ing.run(payloads, SPEC)


if __name__ == "__main__":
    main()
