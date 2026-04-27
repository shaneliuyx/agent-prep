"""Create empty Qdrant collections with different index params."""
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, HnswConfigDiff, OptimizersConfigDiff

qd = QdrantClient(url="http://127.0.0.1:6333")

specs = [
    # (name, dim, distance, hnsw_ef_construct, m)
    ("bge_m3_hnsw",      1024, Distance.COSINE, 128, 16),  # standard HNSW — quality baseline
    ("bge_m3_hnsw_fast",  1024, Distance.COSINE,  64,  8), # ablation: half the graph density
    ("nomic_hnsw",         768, Distance.COSINE, 128, 16),  # different dim — must match model output exactly
]

for name, dim, dist, ef, m in specs:
    if qd.collection_exists(name):
        qd.delete_collection(name)  # idempotent — safe to re-run the script
    qd.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance=dist),
        hnsw_config=HnswConfigDiff(ef_construct=ef, m=m),  # graph params set at collection creation, not changeable later without re-indexing
    )
    print(f"created {name}  dim={dim}  m={m}  ef_construct={ef}")

print("\nCollections:")
for c in qd.get_collections().collections:
    print(" -", c.name)