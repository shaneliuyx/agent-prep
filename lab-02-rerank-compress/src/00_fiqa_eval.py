"""Run dense / sparse / hybrid against BEIR-FiQA-2018. Where Week 1's ceiling effect dies.

M5 Pro optimized — eval portion uses batch encode + bulk Qdrant retrieval, ~5× faster than
the per-query loop. Resumable ingest from earlier patch preserved unchanged.

All model + collection params come from src/model_config.py (atomic config).
"""
import json, math, time
from pathlib import Path
from datasets import load_dataset
from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, SparseVectorParams, PointStruct, SparseVector,
    Prefetch, FusionQuery, Fusion, QueryRequest,
)
from model_config import FIQA_HYBRID

C = FIQA_HYBRID
M = C.model
K = 10

# M5 Pro tunings (48 GB unified memory + Metal 4)
ENCODE_BATCH = 128  # query encoding batch
INGEST_BATCH = 128  # doc encoding batch (was: 64)
QDRANT_BATCH = 64   # queries per HTTP roundtrip in query_batch_points

qd = QdrantClient(url="http://127.0.0.1:6333", timeout=60)  # default 5s breaks once HNSW indexing competes with upserts
m  = BGEM3FlagModel(M.path, use_fp16=False, device="mps")

# BEIR mirrors are the standard way to grab FiQA (corpus + queries + qrels)
corpus  = load_dataset("BeIR/fiqa", "corpus", split="corpus")
queries = load_dataset("BeIR/fiqa", "queries", split="queries")
qrels   = load_dataset("BeIR/fiqa-qrels", split="test")

qid2gold: dict[str, set[str]] = {}
for r in qrels:
    if r["score"] > 0:
        qid2gold.setdefault(str(r["query-id"]), set()).add(str(r["corpus-id"]))
test_qids = set(qid2gold.keys())
queries = [q for q in queries if str(q["_id"]) in test_qids]
print(f"FiQA: {len(corpus)} docs, {len(queries)} test queries, "
      f"{sum(len(g) for g in qid2gold.values())} qrels total")

# === Resumable ingest — survives transient timeouts and continues from last successful batch ===
existing = [col.name for col in qd.get_collections().collections]
if C.name not in existing:
    qd.recreate_collection(
        collection_name=C.name,
        vectors_config={C.dense_vector_name: VectorParams(size=M.dim, distance=M.distance)},
        sparse_vectors_config={C.sparse_vector_name: SparseVectorParams()},
    )

docs = list(corpus)
# Round down to nearest batch boundary; re-upserts of identical ids are idempotent
start = (qd.get_collection(C.name).points_count // INGEST_BATCH) * INGEST_BATCH


def _upsert_with_retry(pts, attempts=4):
    """Retry upsert on transient httpx.ReadTimeout — Qdrant gets slow during background HNSW build."""
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


if start < len(docs):
    if start > 0:
        print(f"resuming from doc {start}/{len(docs)}")
    for i in range(start, len(docs), INGEST_BATCH):
        chunk = docs[i : i + INGEST_BATCH]
        texts = [(d["title"] + " " + d["text"]).strip() for d in chunk]
        out = m.encode(texts, batch_size=INGEST_BATCH, return_dense=True, return_sparse=True)
        points = []
        for j, (dv, sd) in enumerate(zip(out["dense_vecs"], out["lexical_weights"])):
            points.append(PointStruct(
                id=i + j,
                vector={
                    C.dense_vector_name:  dv.tolist(),
                    C.sparse_vector_name: SparseVector(
                        indices=list(map(int,   sd.keys())),
                        values =list(map(float, sd.values())),
                    ),
                },
                payload={"doc_id": str(chunk[j]["_id"])},
            ))
        _upsert_with_retry(points)
        if (i // INGEST_BATCH) % 20 == 0:
            print(f"  {i + len(chunk)}/{len(docs)}")

print(f"collection: {qd.get_collection(C.name).points_count} points (corpus={len(docs)})")

# === Eval — batch encode + bulk Qdrant retrieval ===

qtexts = [q["text"] for q in queries]
qids_eval = [str(q["_id"]) for q in queries]
N = len(qtexts)

# Stage 1: encode ALL queries once (reused across all 3 modes)
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
    rr = ndcg = 0.0
    for qi, qid in enumerate(qids_eval):
        gold = qid2gold[qid]
        ids  = [h.payload["doc_id"] for h in all_hits[qi]]
        n += 1
        if any(d in gold for d in ids):
            recall_hits += 1
        for rank, d in enumerate(ids, 1):
            if d in gold:
                rr += 1.0 / rank
                break
        dcg   = sum(1.0 / math.log2(rank + 1) for rank, d in enumerate(ids, 1) if d in gold)
        ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold), K) + 1))
        ndcg += (dcg / ideal) if ideal else 0.0
    return {
        "benchmark": "BEIR-FiQA-2018", "mode": mode,
        f"recall@{K}": recall_hits / n,
        f"mrr@{K}":    rr / n,
        f"ndcg@{K}":   ndcg / n,
        "wall_sec":    time.time() - t0,
        "n_queries":   n,
    }


results = [metrics("dense"), metrics("sparse"), metrics("hybrid")]
Path("results/fiqa_metrics.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
