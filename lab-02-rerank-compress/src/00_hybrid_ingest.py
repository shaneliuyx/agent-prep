"""Re-ingest 10K MS MARCO docs into a hybrid collection with dense + sparse named vectors.

BGE-M3's signature capability: one forward pass -> dense embedding (1024-d) + sparse
lexical weights (token_id -> weight dict). We index both as named vectors in one Qdrant
collection so a single query can search both and fuse the rankings.

All model + collection params come from src/model_config.py (atomic config principle —
see lab-01 Phase 4.5). To swap encoder or change collection schema, edit the spec, re-run.
"""
import json
from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, SparseVectorParams, PointStruct, SparseVector,
)
from model_config import BGE_M3_HYBRID

C = BGE_M3_HYBRID
M = C.model
assert M.supports_sparse, f"{M.name} doesn't expose sparse output — can't be paired with HybridCollectionSpec"

qd = QdrantClient(url="http://127.0.0.1:6333")
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

BATCH = 64
points = []
for i in range(0, len(texts), BATCH):
    chunk = texts[i : i + BATCH]
    out = m.encode(
        chunk,
        batch_size=BATCH,
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

    if (i // BATCH) % 10 == 0:
        print(f"  encoded {i + len(chunk)}/{len(texts)}")

# Upsert in chunks so request bodies stay reasonable
for i in range(0, len(points), 256):
    qd.upsert(C.name, points[i : i + 256])

info = qd.get_collection(C.name)
print(f"done: {info.points_count} points in {C.name}")
