"""Retrieve top-k for every query, compute recall@10, MRR@10, nDCG@10.

Iterates ALL_COLLECTIONS from src/model_config.py — adding a new collection or model
requires zero edits in this file: append to model_config.py, re-run this script.
"""
import json, math, time
from pathlib import Path
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from model_config import ALL_COLLECTIONS

qd = QdrantClient(url="http://127.0.0.1:6333")

queries = json.loads(Path("data/queries.json").read_text())
qrels = json.loads(Path("data/qrels.json").read_text())
print(f"evaluating {len(queries)} queries across {len(ALL_COLLECTIONS)} collections")

K = 10
encoders = {}
def get_encoder(model_spec):
    # cache avoids reloading a 2 GB model when two collections share an encoder (e.g. BGE-M3 used by both _hnsw and _hnsw_fast)
    if model_spec.path not in encoders:
        encoders[model_spec.path] = SentenceTransformer(
            model_spec.path,
            device="mps",
            trust_remote_code=model_spec.trust_remote_code,
        )
    return encoders[model_spec.path]

def metrics_for(collection_spec):
    C = collection_spec
    M = C.model
    m = get_encoder(M)
    qids = list(queries.keys())
    texts = [M.query_prefix + queries[qid] for qid in qids]   # query_prefix is "" for BGE, "search_query: " for Nomic
    t0 = time.time()
    q_vecs = m.encode(texts, normalize_embeddings=True, batch_size=128, show_progress_bar=False)
    t_embed = time.time() - t0

    recall_sum, mrr_sum, ndcg_sum = 0.0, 0.0, 0.0
    t0 = time.time()
    for qid, qv in zip(qids, q_vecs):
        hits = qd.query_points(C.name, query=qv.tolist(), limit=K).points
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
        "collection": C.name,
        "model":      M.name,
        "recall@10":  recall_sum / n,
        "mrr@10":     mrr_sum / n,
        "ndcg@10":    ndcg_sum / n,
        "embed_sec":  t_embed,
        "search_sec": t_search,
        "n_queries":  n,
    }

results = [metrics_for(c) for c in ALL_COLLECTIONS]
Path("results").mkdir(exist_ok=True)
Path("results/retrieval_metrics.json").write_text(json.dumps(results, indent=2))

# Print a terminal-friendly table
print(f"\n{'collection':<22}{'recall@10':>12}{'mrr@10':>10}{'ndcg@10':>10}{'embed(s)':>10}{'search(s)':>10}")
for r in results:
    print(f"{r['collection']:<22}{r['recall@10']:>12.3f}{r['mrr@10']:>10.3f}{r['ndcg@10']:>10.3f}{r['embed_sec']:>10.1f}{r['search_sec']:>10.1f}")
