"""Compare dense-only vs sparse-only vs hybrid (RRF) on the MS MARCO 6,980-query dev set.

M5 Pro optimized — same metrics, ~5-8× faster than the per-query loop.

Optimizations vs the original:
  1. Encode ALL queries in ONE forward pass (was: 6,980 single-query encodes per mode × 3 modes)
  2. Reuse the same encoded queries across dense / sparse / hybrid modes (was: re-encoded per mode)
  3. query_batch_points for bulk Qdrant retrieval (was: 6,980 sequential HTTP roundtrips per mode)
  4. Hybrid mode batches via QueryRequest with prefetch + FusionQuery (Qdrant 1.10+ supports this)

Qdrant 1.10+ has native RRF via Prefetch + FusionQuery — no manual rank-merging needed.
All model + collection params come from src/model_config.py (atomic config).
"""
import json, time, math
from pathlib import Path
from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Prefetch, FusionQuery, Fusion, SparseVector, QueryRequest,
)
from model_config import BGE_M3_HYBRID

C = BGE_M3_HYBRID
M = C.model
K = 10

# M5 Pro tunings (48 GB unified memory + Metal 4)
ENCODE_BATCH = 128  # query encoding batch
QDRANT_BATCH = 64   # queries per HTTP roundtrip in query_batch_points

qd = QdrantClient(url="http://127.0.0.1:6333", timeout=60)
m  = BGEM3FlagModel(M.path, use_fp16=False, device="mps")

queries = json.loads(Path("data/queries.json").read_text())
qrels   = json.loads(Path("data/qrels.json").read_text())

qids   = list(queries.keys())
qtexts = [queries[qid] for qid in qids]
N      = len(qids)
print(f"queries: {N}  encoder: {M.name}  collection: {C.name}")

# === Stage 1: encode ALL queries once (reused across all 3 modes) ===
t0 = time.time()
out = m.encode(
    qtexts,
    batch_size=ENCODE_BATCH,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=False,
)
dense_vecs:  list[list[float]] = [v.tolist() for v in out["dense_vecs"]]
sparse_vecs: list[SparseVector] = [
    SparseVector(indices=list(map(int,   sd.keys())),
                 values =list(map(float, sd.values())))
    for sd in out["lexical_weights"]
]
t_encode = time.time() - t0
print(f"encoded {N} queries in {t_encode:.1f}s ({N/t_encode:.0f} q/s)")


def bulk_search(mode: str) -> list[list]:
    """Build N QueryRequests for the given mode, submit in batches, return list of N point-lists."""
    if mode == "dense":
        reqs = [
            QueryRequest(query=dense_vecs[i], using=C.dense_vector_name,
                         limit=K, with_payload=True)
            for i in range(N)
        ]
    elif mode == "sparse":
        reqs = [
            QueryRequest(query=sparse_vecs[i], using=C.sparse_vector_name,
                         limit=K, with_payload=True)
            for i in range(N)
        ]
    elif mode == "hybrid":
        # Qdrant native RRF — k=60 default, matches Cormack et al. 2009
        # Swap to FusionQuery(fusion=Fusion.DBSF) to test Distribution-Based Score Fusion;
        # see §1.3.1 in Week 2 runbook for the empirical comparison (DBSF +1.5 MRR, -0.5 recall).
        reqs = [
            QueryRequest(
                prefetch=[
                    Prefetch(query=dense_vecs[i],  using=C.dense_vector_name,  limit=50),
                    Prefetch(query=sparse_vecs[i], using=C.sparse_vector_name, limit=50),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=K,
                with_payload=True,
            )
            for i in range(N)
        ]
    else:
        raise ValueError(mode)

    results: list[list] = []
    for s in range(0, N, QDRANT_BATCH):
        batch = qd.query_batch_points(C.name, requests=reqs[s : s + QDRANT_BATCH])
        results.extend([resp.points for resp in batch])
    return results


def metrics(mode: str) -> dict:
    t0 = time.time()
    all_hits = bulk_search(mode)
    n = recall_hits = 0
    rr_sum = ndcg_sum = 0.0
    for qi, qid in enumerate(qids):
        gold = set(qrels[qid])
        ids  = [h.payload["doc_id"] for h in all_hits[qi]]
        n += 1
        if any(d in gold for d in ids):
            recall_hits += 1
        for rank, d in enumerate(ids, 1):
            if d in gold:
                rr_sum += 1.0 / rank
                break
        dcg   = sum(1.0 / math.log2(rank + 1) for rank, d in enumerate(ids, 1) if d in gold)
        ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold), K) + 1))
        ndcg_sum += (dcg / ideal) if ideal else 0.0
    return {
        "mode": mode,
        f"recall@{K}": recall_hits / n,
        f"mrr@{K}":    rr_sum / n,
        f"ndcg@{K}":   ndcg_sum / n,
        "wall_sec":    time.time() - t0,
        "n_queries":   n,
    }


results = [metrics("dense"), metrics("sparse"), metrics("hybrid")]
Path("results/hybrid_metrics.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
