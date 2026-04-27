"""Create empty Qdrant collections with different index params.

All specs come from src/model_config.py — this script never repeats a collection name,
dim, or HNSW value. Edit the spec, re-run this script.
"""
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, HnswConfigDiff
from model_config import ALL_COLLECTIONS

qd = QdrantClient(url="http://127.0.0.1:6333")

for c in ALL_COLLECTIONS:
    if qd.collection_exists(c.name):
        qd.delete_collection(c.name)  # idempotent — safe to re-run
    qd.create_collection(
        collection_name=c.name,
        vectors_config=VectorParams(size=c.model.dim, distance=c.model.distance),
        hnsw_config=HnswConfigDiff(ef_construct=c.hnsw_ef_construct, m=c.hnsw_m),
    )
    print(f"created {c.name}  model={c.model.name}  dim={c.model.dim}  m={c.hnsw_m}  ef_construct={c.hnsw_ef_construct}")

print("\nCollections:")
for col in qd.get_collections().collections:
    print(" -", col.name)
