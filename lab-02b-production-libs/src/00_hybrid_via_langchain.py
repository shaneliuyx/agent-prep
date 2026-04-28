"""Wrap an existing Qdrant collection as a LangChain VectorStore.

Demonstrates langchain-qdrant — the integration glue that lets you slot a Qdrant
collection into any LangChain chain / agent / retriever pipeline.

Pattern: connect to the *already-populated* bge_m3_hnsw collection (built by lab-01),
wrap it as a QdrantVectorStore, expose it as a Retriever, run the same eval as lab-01.
The recall numbers should match lab-01's 04_eval.py exactly — same encoder, same
collection, just a different API on top.

Note on hybrid: LangChain's RetrievalMode.HYBRID uses fastembed/SPLADE for the sparse
side, NOT BGE-M3's sparse output. For true BGE-M3 hybrid you'd drop back to native
qdrant-client + FlagEmbedding (lab-02's approach). This script intentionally stays in
the dense-only path to show LangChain as integration glue, not as a hybrid abstraction.
"""
import json, math, os, time
from pathlib import Path
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient

HOME = os.path.expanduser("~")
COLLECTION = "bge_m3_hnsw"   # lab-01's dense baseline collection
K = 10

# Connect to the existing collection — no ingestion happens here
qd = QdrantClient(url="http://127.0.0.1:6333")
embedding = HuggingFaceEmbeddings(
    model_name=f"{HOME}/models/bge-m3",
    model_kwargs={"device": "mps", "trust_remote_code": True},
    encode_kwargs={"normalize_embeddings": True},
)

vectorstore = QdrantVectorStore.from_existing_collection(
    embedding=embedding,
    collection_name=COLLECTION,
    url="http://127.0.0.1:6333",
)

# Now you have a LangChain Retriever — usable in any RetrievalQA chain, agent tool, etc.
retriever = vectorstore.as_retriever(search_kwargs={"k": K})

# --- Eval ---
queries = json.loads(Path("data/queries.json").read_text())
qrels   = json.loads(Path("data/qrels.json").read_text())
print(f"evaluating {len(queries)} queries via langchain-qdrant retriever (k={K})")

recall_hits = 0
mrr_sum = ndcg_sum = 0.0
t0 = time.time()
for qid, qtext in queries.items():
    docs = retriever.invoke(qtext)   # LangChain's standard Retriever interface
    # Each Document has .page_content and .metadata; lab-01's payload had doc_id at top level
    ids = [d.metadata.get("doc_id") or d.metadata.get("metadata", {}).get("doc_id") for d in docs]
    gold = set(qrels[qid])

    if any(d in gold for d in ids):
        recall_hits += 1
    for rank, d in enumerate(ids, 1):
        if d in gold:
            mrr_sum += 1.0 / rank
            break
    dcg   = sum(1.0 / math.log2(rank + 1) for rank, d in enumerate(ids, 1) if d in gold)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold), K) + 1))
    ndcg_sum += (dcg / ideal) if ideal else 0.0

n = len(queries)
result = {
    "library":   "langchain-qdrant",
    "collection": COLLECTION,
    f"recall@{K}": recall_hits / n,
    f"mrr@{K}":    mrr_sum / n,
    f"ndcg@{K}":   ndcg_sum / n,
    "wall_sec":    time.time() - t0,
    "n_queries":   n,
}
Path("results").mkdir(exist_ok=True)
Path("results/langchain_qdrant_metrics.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
print("\nCompare to lab-01/results/retrieval_metrics.json [bge_m3_hnsw row] —")
print("recall@10 / mrr@10 / ndcg@10 should be IDENTICAL (same encoder + collection, just LangChain API on top).")
