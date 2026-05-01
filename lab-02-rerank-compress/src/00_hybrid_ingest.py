"""Re-ingest 10K MS MARCO docs into a hybrid collection with dense + sparse
named vectors.

BGE-M3's signature capability: one forward pass → dense embedding (1024-d)
+ sparse lexical weights (token_id → weight dict). We index both as named
vectors in one Qdrant collection so a single query can search both and
fuse the rankings.

Library-driven version — uses shared/rag_hybrid for encoding, schema
creation, batched upsert with retry. The script declares the collection
spec and corpus path; rag_hybrid.Ingestor handles the loop.

ENCODE_BATCH=128 / UPSERT_BATCH=256 — M5 Pro tuning (48 GB unified memory
+ Metal 4). Once autoconfig.recommend() lands as the default, these
constants come from the system probe.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))

from qdrant_client import QdrantClient

from rag_hybrid import (  # noqa: E402
    BGE_M3_HYBRID,
    EncoderConfig,
    HybridEncoder,
    IngestConfig,
    Ingestor,
)

C = BGE_M3_HYBRID
M = C.model
assert M.supports_sparse, (
    f"{M.name} doesn't expose sparse output — can't be paired with HybridCollectionSpec"
)

# M5 Pro tunings
ENCODE_BATCH = 128
UPSERT_BATCH = 256


def main() -> None:
    qd = QdrantClient(url="http://127.0.0.1:6333", timeout=60)

    # docs.jsonl is already pre-chunked (MS MARCO passages are short
    # enough to index whole). Pass through directly as final payloads.
    docs = [json.loads(line) for line in open("data/docs.jsonl")]
    payloads = [{"doc_id": d["id"], "text": d["text"]} for d in docs]
    print(f"loaded {len(docs)} docs into {C.name} via {M.name} (dense + sparse)")

    encoder = HybridEncoder(
        EncoderConfig(spec=M, use_fp16=False, default_batch_size=ENCODE_BATCH)
    )
    ing = Ingestor(
        qd=qd,
        encoder=encoder,
        cfg=IngestConfig(encode_batch=ENCODE_BATCH, upsert_batch=UPSERT_BATCH),
    )
    ing.run(payloads, C)


if __name__ == "__main__":
    main()
