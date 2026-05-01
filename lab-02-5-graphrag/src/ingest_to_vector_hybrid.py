"""Ingest the lab-02.5 tech-founders corpus into a hybrid (dense + sparse)
Qdrant collection.

Library-driven version — uses shared/rag_hybrid for chunking, hybrid
encoding, schema, and upsert. The script declares WHAT to ingest (corpus,
chunk size, collection); rag_hybrid handles HOW.

This is the upgrade path from the dense-only ``tech_corpus_hnsw``. Hybrid
catches the lexical-match cases (rare proper nouns, acronyms, exact phrases)
that dense embeddings dilute across 1024 dimensions. Production GraphRAG
benchmarks (Microsoft GraphRAG, Sarmah et al. 2024 HybridRAG) use hybrid as
the default vector baseline.

Run from the lab-02-5-graphrag directory:
  ./.venv/bin/python src/ingest_to_vector_hybrid.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))

from qdrant_client import QdrantClient

from rag_hybrid import (  # noqa: E402
    BGE_M3,
    HybridCollectionSpec,
    HybridEncoder,
    Ingestor,
    autoconfig,
    char_window_chunks,
    chunk_corpus,
)

# Same name as the existing collection (created by the prior version of
# this script if it ran). Same chunker config as the dense ingest so the
# two collections cover identical text — only the index representation
# differs.
SPEC = HybridCollectionSpec(name="tech_corpus_hybrid", model=BGE_M3)
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
    for p in payloads:
        if "title" in p:
            p["article_title"] = p.pop("title")
    print(
        f"Chunked into {len(payloads)} passages "
        f"({len(payloads) / len(corpus):.1f} per article on average)"
    )

    qd = QdrantClient(url="http://127.0.0.1:6333", timeout=60)
    # System-aware config: BGE-M3 sparse head NaN-prone on MPS in fp16, so
    # autoconfig keeps use_fp16=False on MPS and flips to True on CUDA.
    # encode_batch comes from the memory tier — 128 on this 51 GB host vs
    # 64 on a 16 GB laptop, both correct for their respective targets.
    encoder_cfg = autoconfig.encoder_config_for(BGE_M3)
    print(f"[autoconfig] device={encoder_cfg.device} batch={encoder_cfg.default_batch_size} fp16={encoder_cfg.use_fp16}")
    encoder = HybridEncoder(encoder_cfg)
    ing = Ingestor(qd=qd, encoder=encoder, cfg=autoconfig.ingest_config())
    ing.run(payloads, SPEC)


if __name__ == "__main__":
    main()
