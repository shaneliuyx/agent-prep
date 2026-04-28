"""1:1 with lab-02/src/00_hybrid_eval.py — but via langchain-qdrant retrievers.

Same task: dense / sparse / hybrid retrieval comparison on MS MARCO 6,980-query dev set.
Different code: langchain-qdrant exposes 3 modes via RetrievalMode.{DENSE, SPARSE, HYBRID}.

Reads the lc_hybrid collection built by 00_hybrid_ingest_via_langchain.py (NOT lab-02's
bge_m3_hybrid — the sparse encoder differs: SPLADE++ here vs BGE-M3 sparse there).

Expected delta vs lab-02:
  - dense:  identical numbers (same encoder, same vectors)
  - sparse: differs ~3-7pp (different sparse encoder)
  - hybrid: differs ~2-5pp (RRF on different sparse rankings)
"""
import json, math, os, time
from pathlib import Path
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_huggingface import HuggingFaceEmbeddings

HOME = os.path.expanduser("~")
COLLECTION = "lc_hybrid"
K = 10

dense  = HuggingFaceEmbeddings(model_name=f"{HOME}/models/bge-m3",
                                model_kwargs={"device": "mps", "trust_remote_code": True},
                                encode_kwargs={"normalize_embeddings": True})
sparse = FastEmbedSparse(model_name="prithivida/Splade_PP_en_v1")

def build_store(mode):
    return QdrantVectorStore.from_existing_collection(
        embedding=dense, sparse_embedding=sparse,
        collection_name=COLLECTION, url="http://127.0.0.1:6333",
        retrieval_mode=mode,
    )

stores = {
    "dense":  build_store(RetrievalMode.DENSE),
    "sparse": build_store(RetrievalMode.SPARSE),
    "hybrid": build_store(RetrievalMode.HYBRID),
}

queries = json.loads(Path("data/queries.json").read_text())
qrels   = json.loads(Path("data/qrels.json").read_text())
print(f"evaluating 3 retrieval modes × {len(queries)} queries via langchain-qdrant")

def metrics(mode):
    store = stores[mode]
    n = recall_hits = 0
    rr_sum = ndcg_sum = 0.0
    t0 = time.time()
    for qid, qtext in queries.items():
        gold = set(qrels[qid])
        docs = store.similarity_search(qtext, k=K)
        ids  = [d.metadata.get("doc_id") for d in docs]
        n += 1
        if any(d in gold for d in ids): recall_hits += 1
        for rank, d in enumerate(ids, 1):
            if d in gold: rr_sum += 1.0 / rank; break
        dcg   = sum(1.0 / math.log2(rank + 1) for rank, d in enumerate(ids, 1) if d in gold)
        ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold), K) + 1))
        ndcg_sum += (dcg / ideal) if ideal else 0.0
    return {"library": "langchain-qdrant", "mode": mode,
            f"recall@{K}": recall_hits / n, f"mrr@{K}": rr_sum / n, f"ndcg@{K}": ndcg_sum / n,
            "wall_sec": time.time() - t0, "n_queries": n}

results = [metrics("dense"), metrics("sparse"), metrics("hybrid")]
Path("results").mkdir(exist_ok=True)
Path("results/langchain_hybrid_metrics.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
print("\nCompare to lab-02/results/hybrid_metrics.json — dense should match exactly.")
print("Sparse/hybrid will differ: SPLADE++ != BGE-M3 sparse. Numbers tell you the gap.")
