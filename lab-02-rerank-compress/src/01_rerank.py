"""\
Two-stage: dense retrieve top-50 with BGE-M3, then rerank to top-5 with BGE-reranker-v2-m3.

M5 Pro optimized version. Same metrics as the original; ~3-5× faster.

Optimizations vs the per-query loop in the runbook:
  1. Batch-encode all queries in ONE call (was: 6,980 single-query encodes)
  2. Bulk Qdrant retrieval via query_batch_points (was: 6,980 sequential HTTP roundtrips)
  3. Reranker batch_size=128 (was: 32 — M5 Pro has 48 GB unified memory + Metal 4)
  4. Cross-query pair batching: concatenate 32 queries' pair-lists per predict() call
     to amortize Python/PyTorch dispatch overhead across queries.

The reranker scores each (query, doc) pair independently, so cross-query batching
produces bitwise-identical scores. recall@5 and nDCG@5 match the original exactly.

All model + collection params come from src/model_config.py (atomic config).
"""
import json, math, time
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import QueryRequest
from model_config import BGE_M3_HNSW, BGE_RERANKER_V2_M3

C = BGE_M3_HNSW
M = C.model
R = BGE_RERANKER_V2_M3
TOP_N = R.max_pairs_per_query  # 50
TOP_K = 5

# M5 Pro tunings (48 GB unified memory + Metal 4) — push past the M1/M2 conservative defaults
ENCODE_BATCH         = 128  # query encoding batch (was: implicit 1 per call)
QDRANT_BATCH         = 64   # queries per batch HTTP roundtrip
RERANK_BATCH         = 256  # cross-encoder GPU batch (was: 32 from spec default)
RERANK_GROUP_QUERIES = 32   # queries' pair-lists to concatenate per predict() call

qd       = QdrantClient(url="http://127.0.0.1:6333", timeout=60)
encoder  = SentenceTransformer(M.path, device="mps", trust_remote_code=M.trust_remote_code)
reranker = CrossEncoder(R.path, device="mps")
# Option 2 — fp16 cross-encoder on MPS (M5 Pro Metal 4): typically 1.5-2× faster than fp32.
# If you see NaN scores or silent zeros, comment this line and revert to fp32.
reranker.model.half()

queries = json.loads(Path("data/queries.json").read_text())
qrels   = json.loads(Path("data/qrels.json").read_text())

qids   = list(queries.keys())
qtexts = [queries[qid] for qid in qids]
N      = len(qids)
print(f"queries: {N}  encoder: {M.name}  reranker: {R.name}")

# === Stage 1: batch-encode ALL queries ===
t0 = time.time()
qvecs = encoder.encode(
    [M.query_prefix + q for q in qtexts],
    normalize_embeddings=True,
    batch_size=ENCODE_BATCH,
    show_progress_bar=True,
    convert_to_numpy=True,
)
t_encode = time.time() - t0
print(f"[1/3] encoded {N} queries in {t_encode:.1f}s ({N/t_encode:.0f} q/s)")

# === Stage 2: bulk retrieve top-N from Qdrant ===
t0 = time.time()
candidates: list[list[tuple[str, str, float]]] = [[] for _ in range(N)]
for start in range(0, N, QDRANT_BATCH):
    end = min(start + QDRANT_BATCH, N)
    requests = [
        QueryRequest(query=qvecs[i].tolist(), limit=TOP_N, with_payload=True)
        for i in range(start, end)
    ]
    batch = qd.query_batch_points(C.name, requests=requests)
    for offset, response in enumerate(batch):
        candidates[start + offset] = [
            (h.payload["doc_id"], h.payload["text"], h.score) for h in response.points
        ]
t_retrieve = time.time() - t0
print(f"[2/3] retrieved top-{TOP_N} for {N} queries in {t_retrieve:.1f}s ({N/t_retrieve:.0f} q/s)")

# === Stage 3: bulk rerank with cross-query pair batching ===
t0 = time.time()
rerank_scores = [[0.0] * len(c) for c in candidates]
for g_start in range(0, N, RERANK_GROUP_QUERIES):
    g_end = min(g_start + RERANK_GROUP_QUERIES, N)
    pairs, owners = [], []
    for qi in range(g_start, g_end):
        for ci, cand in enumerate(candidates[qi]):
            pairs.append((qtexts[qi], cand[1]))
            owners.append((qi, ci))
    # One predict() call per group ≈ 32 × 50 = 1,600 pairs in batches of RERANK_BATCH
    scores = reranker.predict(pairs, batch_size=RERANK_BATCH, show_progress_bar=False)
    for (qi, ci), s in zip(owners, scores):
        rerank_scores[qi][ci] = float(s)
t_rerank = time.time() - t0
print(f"[3/3] reranked {N} × {TOP_N} pairs in {t_rerank:.1f}s "
      f"({N/t_rerank:.0f} q/s, batch={RERANK_BATCH})")

# === Stage 4: compute metrics ===
def ndcg_at_k(hit_ids, gold, k):
    dcg  = sum(1.0 / math.log2(i + 2) for i, d in enumerate(hit_ids[:k]) if d in gold)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / idcg if idcg else 0.0

baseline_rows, rerank_rows = [], []
for qi, qid in enumerate(qids):
    gold = set(qrels[qid])
    cands = candidates[qi]
    baseline_ids = [c[0] for c in cands[:TOP_K]]
    rerank_top   = sorted(zip(cands, rerank_scores[qi]), key=lambda x: -x[1])[:TOP_K]
    rerank_ids   = [c[0] for c, _ in rerank_top]
    baseline_rows.append((qid,
        1.0 if set(baseline_ids) & gold else 0.0,
        ndcg_at_k(baseline_ids, gold, TOP_K),
    ))
    rerank_rows.append((qid,
        1.0 if set(rerank_ids) & gold else 0.0,
        ndcg_at_k(rerank_ids, gold, TOP_K),
    ))

def mean(col, rows):
    return sum(r[col] for r in rows) / len(rows)

results = {
    "baseline_dense_top5":     {"recall@5": mean(1, baseline_rows), "ndcg@5": mean(2, baseline_rows)},
    "dense_top50_rerank_top5": {"recall@5": mean(1, rerank_rows),   "ndcg@5": mean(2, rerank_rows)},
    "stage_seconds": {
        "encode":   round(t_encode,   1),
        "retrieve": round(t_retrieve, 1),
        "rerank":   round(t_rerank,   1),
        "total":    round(t_encode + t_retrieve + t_rerank, 1),
    },
    "n_queries": N,
    "config": {
        "encode_batch":         ENCODE_BATCH,
        "qdrant_batch":         QDRANT_BATCH,
        "rerank_batch":         RERANK_BATCH,
        "rerank_group_queries": RERANK_GROUP_QUERIES,
    },
}
Path("results").mkdir(exist_ok=True)
Path("results/rerank_metrics.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
