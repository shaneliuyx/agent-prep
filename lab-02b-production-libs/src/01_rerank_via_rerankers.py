"""Two-stage retrieval (dense top-50 → cross-encoder rerank top-5) using `rerankers` package.

Demonstrates the `rerankers` library — a unified API across BGE / ColBERT / Cohere /
Jina / T5 / RankZephyr cross-encoders. Same model as lab-02 (BGE-reranker-v2-m3),
just a cleaner API. To swap in a different reranker, change one string argument.

Should produce IDENTICAL recall@5 / nDCG@5 numbers to lab-02's 01_rerank.py — both
use the same model, same upstream collection, same top-5 cut. The only difference is
the reranker API surface.
"""
import json, math, os, time
from pathlib import Path
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from rerankers import Reranker

HOME = os.path.expanduser("~")
COLLECTION = "bge_m3_hnsw"   # lab-01's dense baseline — used as upstream
K_RETRIEVE = 50
K_FINAL    = 5

qd = QdrantClient(url="http://127.0.0.1:6333")
encoder = SentenceTransformer(f"{HOME}/models/bge-m3", device="mps", trust_remote_code=True)

# The headline line — single API across vendors. To swap to ColBERT:
#   ranker = Reranker("colbert-ir/colbertv2.0", model_type="colbert")
# To swap to Cohere:
#   ranker = Reranker("rerank-english-v2.0", model_type="cohere", api_key=os.environ["COHERE_API_KEY"])
#
# Lab-02 §2.2.1 lesson: fp16 on M5 Pro Metal 4 is a 2.86× speedup with recall@5
# bitwise-identical and nDCG@5 within 1.6×10⁻⁴ — well below noise. rerankers v0.5+
# accepts dtype as a constructor kwarg, so weights load directly in fp16 (cleaner
# than .half() after construction; no try/except for wrapper-shape drift).
ranker = Reranker(
    f"{HOME}/models/bge-reranker-v2-m3",
    model_type="cross-encoder",
    device="mps",
    dtype="fp16",
)

queries = json.loads(Path("data/queries.json").read_text())
qrels   = json.loads(Path("data/qrels.json").read_text())
print(f"two-stage retrieval (dense top-{K_RETRIEVE} → rerank top-{K_FINAL}) on {len(queries)} queries via `rerankers`")

def retrieve(q_text, top_n=K_RETRIEVE):
    qv = encoder.encode([q_text], normalize_embeddings=True)[0]
    hits = qd.query_points(COLLECTION, query=qv.tolist(), limit=top_n, with_payload=True).points
    return [(h.payload["doc_id"], h.payload["text"]) for h in hits]

def ndcg_at_k(hit_ids, gold, k):
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(hit_ids[:k]) if d in gold)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / idcg if idcg else 0.0

baseline_rows, rerank_rows = [], []
t0 = time.time()
for qid, qtext in queries.items():
    gold = set(qrels[qid])
    cands = retrieve(qtext)

    # Baseline = top-5 from dense, no rerank
    baseline_ids = [doc_id for doc_id, _ in cands[:K_FINAL]]
    baseline_rows.append((
        1.0 if set(baseline_ids) & gold else 0.0,
        ndcg_at_k(baseline_ids, gold, K_FINAL),
    ))

    # Reranked = `rerankers` API with one call
    docs = [text for _, text in cands]
    doc_ids = [doc_id for doc_id, _ in cands]
    results = ranker.rank(query=qtext, docs=docs, doc_ids=doc_ids)
    # Iterate in ranked order, take top-K_FINAL
    rerank_ids = [r.doc_id for r in results.top_k(K_FINAL)]
    rerank_rows.append((
        1.0 if set(rerank_ids) & gold else 0.0,
        ndcg_at_k(rerank_ids, gold, K_FINAL),
    ))

def mean(col, rows):
    return sum(r[col] for r in rows) / len(rows)

result = {
    "library":              "rerankers",
    "model":                "BAAI/bge-reranker-v2-m3",
    "baseline_dense_top5":  {"recall@5": mean(0, baseline_rows), "ndcg@5": mean(1, baseline_rows)},
    "rerank_top5":          {"recall@5": mean(0, rerank_rows),   "ndcg@5": mean(1, rerank_rows)},
    "wall_sec":             time.time() - t0,
    "n_queries":            len(queries),
}
Path("results").mkdir(exist_ok=True)
Path("results/rerankers_metrics.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
print("\nCompare to lab-02/results/rerank_metrics.json — recall@5 + nDCG@5 should match.")
print("If you swap the model_type to 'colbert' or 'cohere', the rest of the script doesn't change.")
