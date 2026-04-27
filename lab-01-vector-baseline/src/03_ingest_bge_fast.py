"""Re-use BGE vectors already in bge_m3_hnsw — re-upsert them into bge_m3_hnsw_fast for index-config ablation.

Why this script exists:
  We want to compare HNSW recall/latency at two graph densities (m=16 vs m=8) on IDENTICAL
  vectors. Re-embedding the corpus would conflate two variables. The cheapest correct setup
  is to embed once (Phase 3.1) and copy the vectors into a second collection whose HNSW
  config is the only thing that differs. The graph topology (m, ef_construct) comes from
  collection-level config set in Phase 2's 02_collections.py — this script is a pure data copy.
"""
import time
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct

qd = QdrantClient(url="http://127.0.0.1:6333")

SRC = "bge_m3_hnsw"
DST = "bge_m3_hnsw_fast"
BATCH = 256  # Qdrant scroll batch — bound by network/serialization, not GPU memory; 256 is safe up to ~10M vectors

# Sanity: verify source has data before we copy (catches "ran 02_collections but skipped 03_ingest_bge")
src_count = qd.get_collection(SRC).points_count
print(f"source {SRC} contains {src_count} points")
if src_count == 0:
    raise RuntimeError(f"{SRC} is empty — run 03_ingest_bge.py first")

offset = None  # opaque cursor returned by Qdrant; do not handcraft
total = 0
t0 = time.time()
while True:
    points, offset = qd.scroll(
        collection_name=SRC,
        limit=BATCH,
        with_vectors=True,   # CRITICAL — default is False (saves bandwidth); without it you copy empty vectors and HNSW returns nothing
        with_payload=True,   # preserves doc_id + truncated text from the original ingest
        offset=offset,
    )
    if not points:           # defensive: empty page should only occur if collection mutated mid-scroll
        break
    qd.upsert(
        collection_name=DST,
        points=[PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in points],
    )
    total += len(points)
    if total % (BATCH * 4) == 0:  # progress every ~1K points
        elapsed = time.time() - t0
        eta = elapsed / total * (src_count - total)
        print(f"  {total}/{src_count}  elapsed {elapsed:.0f}s  eta {eta:.0f}s")
    if offset is None:       # Qdrant signals end-of-collection by returning None offset
        break

dst_count = qd.get_collection(DST).points_count
print(f"done in {time.time()-t0:.0f}s")
print(f"copied {total} points; {DST} now contains {dst_count} (expected {src_count})")
assert dst_count == src_count, f"COUNT MISMATCH: copied {dst_count} but source has {src_count}"