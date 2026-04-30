"""1:1 with lab-02/src/02_compress.py — but via LangChain's ContextualCompressionRetriever.

Same task: take 5 reranked passages, extract only the sentences that directly answer
the query, output compressed context with original wording preserved.

Library version uses langchain.retrievers.ContextualCompressionRetriever wrapping
LLMChainExtractor — LangChain's canonical "extract relevant sentences" abstraction.
The prompt + temperature=0 + max_tokens cap are all baked into LLMChainExtractor's
default behavior; you can customize via from_llm(prompt=...) if needed.

Compare: lab-02's 02_compress.py is ~95 lines (prompt + retrieve + rerank + compress
+ metric loop). This is ~50 lines because the retrieval+compression chain is one object.
"""
import json, os, time
from pathlib import Path
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI

HOME = os.path.expanduser("~")
COLLECTION = "bge_m3_hnsw"

# --- Wrap the existing collection as a LangChain VectorStore ---
embedding = HuggingFaceEmbeddings(
    model_name=f"{HOME}/models/bge-m3",
    model_kwargs={"device": "mps", "trust_remote_code": True},
    encode_kwargs={"normalize_embeddings": True},
)
vectorstore = QdrantVectorStore.from_existing_collection(
    embedding=embedding, collection_name=COLLECTION, url="http://127.0.0.1:6333",
)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# --- Compressor: LLMChainExtractor uses an LLM to extract relevant sentences ---
# Point at oMLX (or any OpenAI-compatible endpoint)
#
# Lab-02 §3.1.1 lesson: oMLX local models (Gemma-4-26B-it-heretic AND gpt-oss-20b) are
# *reasoning models* — they put chain-of-thought in `reasoning_content` and only write
# the final answer to `content` AFTER thinking finishes. `max_tokens` must cover BOTH
# reasoning + output; otherwise content stays None and finish_reason='length'. The
# original 500 was the bug that produced `AttributeError: 'NoneType' has no .split()`.
# 3000 covers Gemma's worst-case reasoning length on MS MARCO 5-passage compression.
# The cleanest fix is to disable thinking at the oMLX server level; this is the
# defensive script-layer fallback for environments where the toggle isn't accessible.
llm = ChatOpenAI(
    model=os.getenv("MODEL_SONNET", "gemma-4-26B-A4B-it-heretic-4bit"),
    base_url=os.getenv("OMLX_BASE_URL"),
    api_key=os.getenv("OMLX_API_KEY"),
    temperature=0.0,
    max_tokens=3000,   # was 500 — see lab-02 §3.1.1 (reasoning-model token-budget fix)
)
compressor = LLMChainExtractor.from_llm(llm)

# --- Compose into one retriever — calls retrieval + compression in one shot ---
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever,
)

# --- Eval on 50-query sample ---
queries = json.loads(Path("data/queries.json").read_text())
sample_qids = list(queries.keys())[:50]
print(f"compressing context for {len(sample_qids)} queries via LangChain ContextualCompressionRetriever")

rows = []
t0 = time.time()
for qid in sample_qids:
    q = queries[qid]
    raw_docs = base_retriever.invoke(q)
    raw_text = "\n\n".join(d.page_content for d in raw_docs)
    compressed_docs = compression_retriever.invoke(q)
    compressed_text = "\n\n".join(d.page_content for d in compressed_docs)
    raw_words  = len(raw_text.split())
    comp_words = len(compressed_text.split())
    rows.append({
        "qid": qid,
        "raw_word_count": raw_words,
        "compressed_word_count": comp_words,
        "ratio": comp_words / max(raw_words, 1),
    })

summary = {
    "library": "langchain ContextualCompressionRetriever + LLMChainExtractor",
    "n": len(rows),
    "mean_raw_words":         sum(r["raw_word_count"]        for r in rows) / len(rows),
    "mean_compressed_words":  sum(r["compressed_word_count"] for r in rows) / len(rows),
    "mean_compression_ratio": sum(r["ratio"]                 for r in rows) / len(rows),
    "total_wall_sec":         time.time() - t0,
    "per_query_sec":          (time.time() - t0) / len(rows),
}
Path("results").mkdir(exist_ok=True)
Path("results/langchain_compression_metrics.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
print(json.dumps(summary, indent=2))
print("\nCompare to lab-02/results/compression_metrics.json — ratio should be in same 0.25-0.50 range.")
print("LangChain hides the prompt; for custom prompts use LLMChainExtractor.from_llm(llm, prompt=...).")
