"""Re-ingest 10K MS MARCO docs into a hybrid collection with dense + sparse named vectors.

M5 Pro optimized — same metrics, ~1.5× faster than the M1/M2 conservative defaults.

BGE-M3's signature capability: one forward pass -> dense embedding (1024-d) + sparse
lexical weights (token_id -> weight dict). We index both as named vectors in one Qdrant
collection so a single query can search both and fuse the rankings.

Optimizations vs the original:
  1. BATCH 64 → 128 (M5 Pro has 48 GB unified memory + Metal 4 — plenty of headroom)
  2. timeout=60s on QdrantClient (default 5s breaks once HNSW indexing competes with upserts)
  3. _upsert_with_retry helper for resilience to transient slowdowns

All model + collection params come from src/model_config.py (atomic config principle —
see lab-01 Phase 4.5). To swap encoder or change collection schema, edit the spec, re-run.
"""
import json, time
from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, SparseVectorParams, PointStruct, SparseVector,
)
from model_config import BGE_M3_HYBRID

C = BGE_M3_HYBRID
M = C.model
assert M.supports_sparse, f"{M.name} doesn't expose sparse output — can't be paired with HybridCollectionSpec"

# M5 Pro tunings
ENCODE_BATCH = 128   # was 64; M5 Pro can comfortably handle 2× the M1/M2 default
UPSERT_BATCH = 256   # HTTP body chunks — already sized correctly for Qdrant's 32 MB limit

qd = QdrantClient(url="http://127.0.0.1:6333", timeout=60)
m  = BGEM3FlagModel(M.path, use_fp16=False, device="mps")

docs  = [json.loads(l) for l in open("data/docs.jsonl")]
texts = [d["text"] for d in docs]
print(f"loaded {len(docs)} docs into {C.name} via {M.name} (dense + sparse)")

# Two named vectors per point: "dense" (1024-d float, cosine) and "sparse" (inverted index).
qd.recreate_collection(
    collection_name=C.name,
    vectors_config={C.dense_vector_name: VectorParams(size=M.dim, distance=M.distance)},
    sparse_vectors_config={C.sparse_vector_name: SparseVectorParams()},
)


def _upsert_with_retry(pts, attempts=4):
    """Retry on transient httpx.ReadTimeout — Qdrant gets slow during background HNSW build."""
    for k in range(attempts):
        try:
            qd.upsert(C.name, pts)
            return
        except Exception as e:
            if k == attempts - 1:
                raise
            wait_s = 2 ** k
            print(f"  upsert retry {k + 1}/{attempts} after {type(e).__name__} — sleeping {wait_s}s")
            time.sleep(wait_s)


t0 = time.time()
points = []
for i in range(0, len(texts), ENCODE_BATCH):
    chunk = texts[i : i + ENCODE_BATCH]
    out = m.encode(
        chunk,
        batch_size=ENCODE_BATCH,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,   # ColBERT is a separate experiment
    )
    dense_vecs   = out["dense_vecs"]         # (B, 1024)
    sparse_dicts = out["lexical_weights"]    # list of {token_id: weight}

    for j, (dv, sd) in enumerate(zip(dense_vecs, sparse_dicts)):
        idx = i + j
        sparse = SparseVector(
            indices=list(map(int,   sd.keys())),
            values =list(map(float, sd.values())),
        )
        points.append(PointStruct(
            id=idx,
            vector={
                C.dense_vector_name:  dv.tolist(),
                C.sparse_vector_name: sparse,
            },
            payload={"doc_id": docs[idx]["id"], "text": docs[idx]["text"]},
        ))

    if (i // ENCODE_BATCH) % 10 == 0:
        print(f"  encoded {i + len(chunk)}/{len(texts)}")

t_encode = time.time() - t0
print(f"encoding done in {t_encode:.1f}s ({len(texts)/t_encode:.0f} docs/s)")

# Upsert in chunks so request bodies stay reasonable (and survive transient timeouts)
t0 = time.time()
for i in range(0, len(points), UPSERT_BATCH):
    _upsert_with_retry(points[i : i + UPSERT_BATCH])
t_upsert = time.time() - t0
print(f"upsert done in {t_upsert:.1f}s")

info = qd.get_collection(C.name)
print(f"done: {info.points_count} points in {C.name}")
