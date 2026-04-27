"""Embed all 10K docs with BGE-M3 and upsert into bge_m3_hnsw."""
import json, time
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from model_config import BGE_M3_HNSW

C = BGE_M3_HNSW
M = C.model

docs = [json.loads(l) for l in open("data/docs.jsonl")]
print(f"loaded {len(docs)} docs into {C.name} via {M.name}")

m = SentenceTransformer(M.path, device="mps", trust_remote_code=M.trust_remote_code)
qd = QdrantClient(url="http://127.0.0.1:6333")

BATCH = 64  # too high causes MPS OOM on M1; drop to 16 if you see crashes
t0 = time.time()
for i in range(0, len(docs), BATCH):
    batch = docs[i : i + BATCH]
    vecs = m.encode(
        [M.doc_prefix + d["text"] for d in batch],
        normalize_embeddings=True, show_progress_bar=False,
    )
    points = [
        # payload stores original doc_id for qrel lookup; text truncated to 500 chars to cap Qdrant storage
        PointStruct(id=i + j, vector=vec.tolist(), payload={"doc_id": d["id"], "text": d["text"][:500]})
        for j, (d, vec) in enumerate(zip(batch, vecs))
    ]
    qd.upsert(collection_name=C.name, points=points)
    if i % (BATCH * 10) == 0:  # progress every 10 batches (~640 docs)
        elapsed = time.time() - t0
        eta = elapsed / max(i + BATCH, 1) * (len(docs) - i - BATCH)
        print(f"  {i+len(batch)}/{len(docs)}  elapsed {elapsed:.0f}s  eta {eta:.0f}s")

print(f"done in {time.time()-t0:.0f}s")
print(f"collection count: {qd.get_collection(C.name).points_count}")
