"""Ingest Berkshire 2023 corpus into Qdrant — dense-only collection.

Reuses shared/rag_hybrid wholesale:
  - chunk_corpus + char_window_chunks for chunking
  - HybridEncoder + autoconfig for embeddings (works for dense-only spec
    too; just skips sparse output)
  - Ingestor for upsert orchestration
  - CollectionSpec(BGE_M3) for the collection schema

Equivalent to lab-03's src/ingest_to_vector.py + lab-02-5's variant —
proof that the shared library is the single source of truth for dense
ingest across the curriculum.

Run from project root:
    cd ~/code/agent-prep/lab-02-7-pageindex
    source .venv/bin/activate
    uv pip install -e .
    python src/_brk_corpus.py            # writes data/brk_corpus.json
    python src/ingest_brk_to_vector.py   # writes Qdrant collection 'brk_2023_dense'
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from qdrant_client import QdrantClient

# Bootstrap shared/ onto sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "shared"))

from rag_hybrid import (  # noqa: E402
    BGE_M3,
    CollectionSpec,
    HybridEncoder,
    Ingestor,
    autoconfig,
    char_window_chunks,
    chunk_corpus,
)

SPEC = CollectionSpec(name="brk_2023_dense", model=BGE_M3)
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


def main() -> None:
    corpus_path = Path("data/brk_corpus.json")
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Missing {corpus_path}. Run: python src/_brk_corpus.py"
        )

    corpus = json.loads(corpus_path.read_text())
    print(f"Loaded {len(corpus)} sections from {corpus_path}")

    payloads = chunk_corpus(
        corpus,
        chunker=lambda t: char_window_chunks(t, CHUNK_SIZE, CHUNK_OVERLAP),
        extra_payload=("title",),
    )
    # Rename 'title' → 'article_title' to match the convention used by W2 +
    # lab-02-5 + lab-03 — keeps cross-lab payload schemas consistent.
    for p in payloads:
        if "title" in p:
            p["article_title"] = p.pop("title")
    print(
        f"Chunked into {len(payloads)} passages "
        f"({len(payloads) / max(len(corpus), 1):.1f} per section on average)"
    )

    qd = QdrantClient(url="http://127.0.0.1:6333", timeout=60)
    encoder_cfg = autoconfig.encoder_config_for(BGE_M3)
    print(
        f"[autoconfig] device={encoder_cfg.device} "
        f"batch={encoder_cfg.default_batch_size} "
        f"fp16={encoder_cfg.use_fp16}"
    )
    encoder = HybridEncoder(encoder_cfg)
    ing = Ingestor(qd=qd, encoder=encoder, cfg=autoconfig.ingest_config())
    ing.run(payloads, SPEC)


if __name__ == "__main__":
    main()
