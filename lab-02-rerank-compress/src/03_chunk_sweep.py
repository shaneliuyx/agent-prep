"""Build 9 retrieval indices: chunk_size × overlap = {256, 512, 1024} × {0, 64, 128}.

All variants come from src/model_config.py SWEEP_VARIANTS (atomic config — change the
sweep grid in one place, scripts pick it up automatically).
"""
import json, time
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, HnswConfigDiff
from model_config import SWEEP_VARIANTS, LONG_DOC_PASSAGES

# All variants share the same model (BGE-M3); pull it from any variant
M = SWEEP_VARIANTS[0].model
N = LONG_DOC_PASSAGES   # passages per synthetic long doc — MUST match 03b_sweep_eval.py

qd = QdrantClient(url="http://127.0.0.1:6333")
m  = SentenceTransformer(M.path, device="mps", trust_remote_code=M.trust_remote_code)

# Simulate long docs by concatenating N passages each
raw = [json.loads(l) for l in open("data/docs.jsonl")]
LONG = []
for i in range(0, len(raw), N):
    merged = " ".join(d["text"] for d in raw[i : i + N])
    LONG.append({"id": f"long_{i//N}", "text": merged, "src_ids": [d["id"] for d in raw[i : i + N]]})
print(f"{len(LONG)} synthetic long docs · {len(SWEEP_VARIANTS)} variants to build")

def chunk(text, size, overlap):
    words = text.split()
    step = max(1, size - overlap)
    return [" ".join(words[i : i + size]) for i in range(0, len(words), step) if words[i : i + size]]

def index_variant(spec):
    if qd.collection_exists(spec.name): qd.delete_collection(spec.name)
    qd.create_collection(
        spec.name,
        vectors_config=VectorParams(size=spec.model.dim, distance=spec.model.distance),
        hnsw_config=HnswConfigDiff(m=spec.hnsw_m, ef_construct=spec.hnsw_ef_construct),
    )
    batch, pid = [], 0
    for doc in LONG:
        for chunk_text in chunk(doc["text"], spec.chunk_size, spec.overlap):
            batch.append((pid, chunk_text, doc["id"]))
            pid += 1
            if len(batch) >= 128:
                vecs = m.encode([b[1] for b in batch], normalize_embeddings=True, show_progress_bar=False)
                qd.upsert(spec.name, points=[PointStruct(id=b[0], vector=v.tolist(), payload={"parent": b[2], "text": b[1][:500]}) for b, v in zip(batch, vecs)])
                batch = []
    if batch:
        vecs = m.encode([b[1] for b in batch], normalize_embeddings=True, show_progress_bar=False)
        qd.upsert(spec.name, points=[PointStruct(id=b[0], vector=v.tolist(), payload={"parent": b[2], "text": b[1][:500]}) for b, v in zip(batch, vecs)])
    return pid

for spec in SWEEP_VARIANTS:
    t0 = time.time()
    n = index_variant(spec)
    print(f"s={spec.chunk_size} o={spec.overlap}  points={n}  wall={time.time()-t0:.0f}s")
print("done")