"""1:1 with lab-02/src/03_chunk_sweep.py — but via LangChain text splitters + langchain-qdrant.

Same task: build 9 retrieval indices for the chunk_size × overlap = 3×3 sweep.
Different code: LangChain provides RecursiveCharacterTextSplitter (and many other
splitters: SemanticChunker, TokenTextSplitter, MarkdownHeaderTextSplitter, etc.) —
swap one class to compare splitting strategies. langchain-qdrant handles ingest.
"""
import json, time
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
import os

HOME = os.path.expanduser("~")

# Same 9-cell grid as lab-02 (could come from model_config.py SWEEP_VARIANTS)
#
# Lab-02 §4.4 empirical result on MS MARCO (parent-level recall@5, 50 queries):
#   chunk_size=256  + overlap=128 → 0.786  ← winner
#   chunk_size=256  + overlap=0   → 0.686
#   chunk_size=512  + overlap=128 → 0.541
#   chunk_size=1024 + any overlap → 0.493  (overlap is INERT at 1024)
#
# Smaller chunks dominate by 24.5pp — the "more context per chunk = better" intuition
# fails because BGE-M3 *averages* token signal into one pooled vector. A 100-token
# answer span is ~40% of a 256-token chunk's signal but only ~10% of a 1024-token
# chunk's. The full grid below stays for teaching purposes (and so the heatmap is
# meaningful), but in production prune the 1024 row entirely.
SIZES = (256, 512, 1024)
OVERLAPS = (0, 64, 128)
LONG_DOC_PASSAGES = 8   # MUST match lab-02's spec — see lab-02 model_config.py

# Same encoder as lab-02 — comparable retrieval numbers across labs
embedding = HuggingFaceEmbeddings(
    model_name=f"{HOME}/models/bge-m3",
    model_kwargs={"device": "mps", "trust_remote_code": True},
    encode_kwargs={"normalize_embeddings": True},
)

# Build synthetic long docs (same as lab-02)
raw = [json.loads(l) for l in open("data/docs.jsonl")]
LONG = []
for i in range(0, len(raw), LONG_DOC_PASSAGES):
    chunk_passages = raw[i : i + LONG_DOC_PASSAGES]
    merged = " ".join(d["text"] for d in chunk_passages)
    LONG.append({"id": f"long_{i//LONG_DOC_PASSAGES}", "text": merged})
print(f"{len(LONG)} synthetic long docs · {len(SIZES) * len(OVERLAPS)} variants to build")

for size in SIZES:
    for overlap in OVERLAPS:
        col = f"lc_sweep_s{size}_o{overlap}"
        # LangChain's RecursiveCharacterTextSplitter handles the chunking — handles edge
        # cases (empty chunks, boundary handling) that the hand-rolled chunk() fn doesn't.
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            length_function=lambda s: len(s.split()),   # word-based, matches lab-02
        )
        all_chunks = []
        all_metas  = []
        for doc in LONG:
            chunks = splitter.split_text(doc["text"])
            all_chunks.extend(chunks)
            all_metas.extend([{"parent": doc["id"]} for _ in chunks])
        t0 = time.time()
        QdrantVectorStore.from_texts(
            texts=all_chunks, embedding=embedding,
            metadatas=all_metas, collection_name=col,
            url="http://127.0.0.1:6333", force_recreate=True,
        )
        print(f"s={size} o={overlap}  chunks={len(all_chunks)}  wall={time.time()-t0:.0f}s")

print("done")
print("\nCompare to lab-02/src/03_chunk_sweep.py — same 9 collections, ~50% less code.")
print("LangChain text splitters offer many alternatives: SemanticChunker, TokenTextSplitter,")
print("MarkdownHeaderTextSplitter — change one class to A/B different splitting strategies.")
