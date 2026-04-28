"""1:1 with lab-02/src/00_fiqa_eval.py — but via langchain-qdrant + datasets.

Same task: dense / sparse / hybrid retrieval against BEIR-FiQA-2018.
Different code: langchain-qdrant abstracts the collection ops; datasets handles the BEIR mirror.

Builds an `lc_fiqa_hybrid` collection (separate from lab-02's `fiqa_hybrid`) using
SPLADE sparse instead of BGE-M3 sparse. Recall numbers will differ from lab-02 by ~3-5pp
on the sparse and hybrid modes — that gap is the lesson.
"""
import json, math, os, time
from pathlib import Path
from datasets import load_dataset
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient

HOME = os.path.expanduser("~")
COLLECTION = "lc_fiqa_hybrid"
K = 10

dense  = HuggingFaceEmbeddings(model_name=f"{HOME}/models/bge-m3",
                                model_kwargs={"device": "mps", "trust_remote_code": True},
                                encode_kwargs={"normalize_embeddings": True})
sparse = FastEmbedSparse(model_name="prithivida/Splade_PP_en_v1")
qd = QdrantClient(url="http://127.0.0.1:6333")

# Bootstrap: load BEIR-FiQA, ingest if collection empty
corpus  = load_dataset("BeIR/fiqa", "corpus", split="corpus")
queries = load_dataset("BeIR/fiqa", "queries", split="queries")
qrels   = load_dataset("BeIR/fiqa-qrels", split="test")

qid2gold = {}
for r in qrels:
    if r["score"] > 0:
        qid2gold.setdefault(str(r["query-id"]), set()).add(str(r["corpus-id"]))
test_qids = set(qid2gold.keys())
queries = [q for q in queries if str(q["_id"]) in test_qids]
print(f"FiQA: {len(corpus)} docs · {len(queries)} test queries")

existing = [c.name for c in qd.get_collections().collections]
if COLLECTION not in existing or qd.get_collection(COLLECTION).points_count == 0:
    docs_list = list(corpus)
    texts = [(d["title"] + " " + d["text"]).strip() for d in docs_list]
    metadatas = [{"doc_id": str(d["_id"])} for d in docs_list]
    QdrantVectorStore.from_texts(
        texts=texts, embedding=dense, sparse_embedding=sparse,
        metadatas=metadatas, collection_name=COLLECTION,
        url="http://127.0.0.1:6333", retrieval_mode=RetrievalMode.HYBRID,
    )
    print(f"ingested {len(texts)} points into {COLLECTION}")

# Build retrievers for each mode (langchain-qdrant exposes 3 modes via retrieval_mode=)
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

def metrics(mode):
    store = stores[mode]
    n = recall_hits = 0
    rr_sum = ndcg_sum = 0.0
    t0 = time.time()
    for q in queries:
        qid  = str(q["_id"])
        gold = qid2gold[qid]
        docs = store.similarity_search(q["text"], k=K)
        ids  = [d.metadata["doc_id"] for d in docs]
        n += 1
        if any(d in gold for d in ids): recall_hits += 1
        for rank, d in enumerate(ids, 1):
            if d in gold: rr_sum += 1.0 / rank; break
        dcg   = sum(1.0 / math.log2(rank + 1) for rank, d in enumerate(ids, 1) if d in gold)
        ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold), K) + 1))
        ndcg_sum += (dcg / ideal) if ideal else 0.0
    return {"library": "langchain-qdrant", "benchmark": "BEIR-FiQA-2018", "mode": mode,
            f"recall@{K}": recall_hits / n, f"mrr@{K}": rr_sum / n, f"ndcg@{K}": ndcg_sum / n,
            "wall_sec": time.time() - t0, "n_queries": n}

results = [metrics("dense"), metrics("sparse"), metrics("hybrid")]
Path("results").mkdir(exist_ok=True)
Path("results/langchain_fiqa_metrics.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
print("\nCompare to lab-02/results/fiqa_metrics.json — dense should match (same encoder).")
print("Sparse/hybrid will differ: SPLADE++ != BGE-M3 sparse. Typically -3 to -5pp on hybrid.")
