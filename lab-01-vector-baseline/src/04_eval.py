"""Retrieve top-k for every query, compute recall@10, MRR@10, nDCG@10."""
import json, math, os, time
from pathlib import Path
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

HOME = os.path.expanduser("~")
qd = QdrantClient(url="http://127.0.0.1:6333")

queries = json.loads(Path("data/queries.json").read_text())
qrels = json.loads(Path("data/qrels.json").read_text())
print(f"evaluating on {len(queries)} queries")

CONFIGS = [
    # (collection, embed_model_path, query_prefix)
    ("bge_m3_hnsw",      f"{HOME}/models/bge-m3",         ""),            # no prefix — BGE handles bare queries
    ("bge_m3_hnsw_fast", f"{HOME}/models/bge-m3",         ""),
    ("nomic_hnsw",       f"{HOME}/models/nomic-embed-v2", "search_query: "),  # Nomic requires asymmetric prefixes
]

K = 10
encoders = {}
def get_encoder(path):
    if path not in encoders:
        encoders[path] = SentenceTransformer(path, device="mps", trust_remote_code=True)
    return encoders[path]  # cache avoids reloading a 2 GB model for each config

def metrics_for(collection, model_path, prefix):
    m = get_encoder(model_path)
    qids = list(queries.keys())
    texts = [prefix + queries[qid] for qid in qids]
    t0 = time.time()
    q_vecs = m.encode(texts, normalize_embeddings=True, batch_size=128, show_progress_bar=False)
    t_embed = time.time() - t0

    recall_sum, mrr_sum, ndcg_sum = 0.0, 0.0, 0.0
    t0 = time.time()
    for qid, qv in zip(qids, q_vecs):
        hits = qd.query_points(collection, query=qv.tolist(), limit=K).points
        hit_ids = [h.payload["doc_id"] for h in hits]
        gold = set(qrels[qid])

        # Recall@10: did any gold doc appear in the top-10? (binary, not rank-sensitive)
        recall_sum += 1.0 if gold & set(hit_ids) else 0.0
        # MRR@10: reciprocal rank of the first gold doc hit
        rank = next((i + 1 for i, d in enumerate(hit_ids) if d in gold), None)
        mrr_sum += 1.0 / rank if rank else 0.0
        # nDCG@10: graded gain discounted by log rank; IDCG = best achievable DCG with this K
        dcg = sum((1.0 / math.log2(i + 2)) for i, d in enumerate(hit_ids) if d in gold)
        idcg = sum((1.0 / math.log2(i + 2)) for i in range(min(len(gold), K)))
        ndcg_sum += dcg / idcg if idcg else 0.0

    t_search = time.time() - t0
    n = len(qids)
    return {
        "collection": collection,
        "recall@10": recall_sum / n,
        "mrr@10":    mrr_sum / n,
        "ndcg@10":   ndcg_sum / n,
        "embed_sec": t_embed,
        "search_sec": t_search,
        "n_queries": n,
    }

results = [metrics_for(c, m, p) for c, m, p in CONFIGS]
Path("results").mkdir(exist_ok=True)
Path("results/retrieval_metrics.json").write_text(json.dumps(results, indent=2))

# Print a terminal-friendly table
print(f"\n{'collection':<22}{'recall@10':>12}{'mrr@10':>10}{'ndcg@10':>10}{'embed(s)':>10}{'search(s)':>10}")
for r in results:
    print(f"{r['collection']:<22}{r['recall@10']:>12.3f}{r['mrr@10']:>10.3f}{r['ndcg@10']:>10.3f}{r['embed_sec']:>10.1f}{r['search_sec']:>10.1f}")