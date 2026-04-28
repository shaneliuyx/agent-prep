"""1:1 with lab-02/src/00_hybrid_ingest.py — but via langchain-qdrant + FastEmbed.

The PRIMITIVES version (lab-02): builds bge_m3_hybrid collection with BGE-M3 sparse output.
The LIBRARY version (this file): builds lc_hybrid collection with fastembed/SPLADE sparse.

The two collections have IDENTICAL dense vectors (BGE-M3) but DIFFERENT sparse vectors:
  - lab-02:  BGE-M3's learned sparse weights (one forward pass produces both)
  - lab-02b: SPLADE++ via fastembed (a separate model — adds ~10-15% encode time)

This is an intentional teaching moment: "library hybrid" ≠ "BGE-M3 hybrid". They share
a name but use different sparse encoders. Recall numbers will differ; the gap is the lesson.
"""
import json
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_huggingface import HuggingFaceEmbeddings
import os

HOME = os.path.expanduser("~")
COLLECTION = "lc_hybrid"   # new collection — does NOT collide with lab-02's bge_m3_hybrid

# Dense embedding: same BGE-M3 as lab-02 (so dense recall numbers will match)
dense = HuggingFaceEmbeddings(
    model_name=f"{HOME}/models/bge-m3",
    model_kwargs={"device": "mps", "trust_remote_code": True},
    encode_kwargs={"normalize_embeddings": True},
)

# Sparse embedding: SPLADE++ via fastembed (NOT BGE-M3 sparse — that requires FlagEmbedding)
sparse = FastEmbedSparse(model_name="prithivida/Splade_PP_en_v1")

# Load corpus + texts (same data lab-02 uses)
docs = [json.loads(l) for l in open("data/docs.jsonl")]
texts = [d["text"] for d in docs]
metadatas = [{"doc_id": d["id"]} for d in docs]
print(f"loaded {len(texts)} docs · ingesting into {COLLECTION} (dense=BGE-M3, sparse=SPLADE++)")

# from_texts() handles: collection creation, encoding (both modes), upsert.
# Compare to lab-02's 65 lines: this is 5 lines.
vectorstore = QdrantVectorStore.from_texts(
    texts=texts,
    embedding=dense,
    sparse_embedding=sparse,
    metadatas=metadatas,
    collection_name=COLLECTION,
    url="http://127.0.0.1:6333",
    retrieval_mode=RetrievalMode.HYBRID,
    force_recreate=True,
)

print(f"done: {len(texts)} points in {COLLECTION}")
print("\nCompare to lab-02/src/00_hybrid_ingest.py — same operation, ~13× less code.")
print("Trade-off: SPLADE++ sparse instead of BGE-M3 sparse. See 00_hybrid_eval_via_langchain.py for recall delta.")
