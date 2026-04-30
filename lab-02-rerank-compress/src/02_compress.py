"""Compress retrieved context: extract only spans that answer the query."""
import json, os, time
from pathlib import Path
from openai import OpenAI

SONNET = os.getenv("MODEL_SONNET", "gemma-4-26B-A4B-it-heretic-4bit")
omlx = OpenAI(base_url=os.getenv("OMLX_BASE_URL"), api_key=os.getenv("OMLX_API_KEY"))

PROMPT = """You are a context compressor. Given a user query and 5 candidate passages, extract ONLY the sentences that directly answer the query. Preserve original wording for every extracted sentence. If a passage is irrelevant, drop it entirely. Output format:

[doc_id_1] sentence A. sentence B.
[doc_id_2] sentence C.
...

User query: {query}

Passages:
{passages}

Compressed context:"""

def compress(query, docs_with_ids):
    """Returns (content_str, usage). content_str is "" (not None) on refusal/empty completion.

    Note: oMLX's local models (Gemma-4-26B-it-heretic AND gpt-oss-20b) are reasoning models —
    they put chain-of-thought in `reasoning_content` and only write the final answer to `content`
    AFTER thinking finishes. `max_tokens` must cover BOTH reasoning + output, otherwise content
    stays None and finish_reason='length'.

    Empirical token usage on MS MARCO 5-passage compression:
      gpt-oss-20b:       ~200 tokens (reasoning ~150 + content ~50)
      gemma-4-26B:       ~300-2500+ tokens (reasoning is verbose; vague queries reason longer)

    3000 covers Gemma's worst case observed. If still hitting finish_reason=length, either:
      (a) bump to 4000+, or
      (b) switch to gpt-oss-20b: MODEL_SONNET=gpt-oss-20b-MXFP4-Q8 python src/02_compress.py
    """
    passages = "\n\n".join(f"[{did}] {text}" for did, text, _ in docs_with_ids)
    resp = omlx.chat.completions.create(
        model=SONNET,
        messages=[{"role": "user", "content": PROMPT.format(query=query, passages=passages)}],
        temperature=0.0,
        max_tokens=3000,   # was 500/1500 — covers Gemma's worst-case reasoning length
    )
    msg = resp.choices[0].message
    content = msg.content
    if content is None:
        finish_reason = resp.choices[0].finish_reason
        # Check for reasoning-model output channel (Gemma-it-heretic, gpt-oss reasoning, DeepSeek-R1)
        reasoning = getattr(msg, "reasoning_content", None)
        if reasoning and finish_reason == "length":
            print(f"  [WARN] content=None, reasoning_content truncated by max_tokens (finish_reason={finish_reason}). "
                  f"Bump max_tokens or switch to a non-reasoning model. Query: {query[:60]!r}")
        else:
            print(f"  [WARN] content=None (finish_reason={finish_reason}, has_reasoning={bool(reasoning)}) — query: {query[:60]!r}")
        content = ""
    return content, resp.usage

# Pull some rerank outputs from Phase 2 (or re-run inline)
# For speed, grab just 50 queries for the compression measurement
queries = json.loads(Path("data/queries.json").read_text())
sample_qids = list(queries.keys())[:50]

# ─── retrieve() + rerank() — duplicated verbatim from 01_rerank.py ──────────────
# Curriculum convention: each script reads top-to-bottom without inter-script imports.
# In production, extract to src/pipeline.py and import from both scripts. If you fix a
# bug here, fix it in 01_rerank.py too — they MUST stay in sync.
# ────────────────────────────────────────────────────────────────────────────────
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from model_config import BGE_M3_HNSW, BGE_RERANKER_V2_M3

C = BGE_M3_HNSW
M = C.model
R = BGE_RERANKER_V2_M3

qd = QdrantClient(url="http://127.0.0.1:6333")
encoder  = SentenceTransformer(M.path, device="mps", trust_remote_code=M.trust_remote_code)
reranker = CrossEncoder(R.path, device="mps")

def retrieve(q, top_n=R.max_pairs_per_query):
    qv = encoder.encode([M.query_prefix + q], normalize_embeddings=True)[0]
    return qd.query_points(C.name, query=qv.tolist(), limit=top_n, with_payload=True).points

def rerank(q, cands, k=5):
    pairs = [(q, c.payload["text"]) for c in cands]
    scores = reranker.predict(pairs, batch_size=R.batch_size)
    order = sorted(zip(cands, scores), key=lambda x: -x[1])[:k]
    return [(c.payload["doc_id"], c.payload["text"], float(s)) for c, s in order]

rows = []
failures = []
t0 = time.time()
for qid in sample_qids:
    q = queries[qid]
    try:
        cands = retrieve(q)   # top_n defaults to R.max_pairs_per_query (50)
        top5 = rerank(q, cands, 5)
        raw_tokens = sum(len(t.split()) for _, t, _ in top5)  # rough proxy
        compressed, usage = compress(q, top5)
        comp_tokens = len(compressed.split()) if compressed else 0
        if comp_tokens == 0:
            failures.append({"qid": qid, "reason": "empty_compressed_content"})
            continue   # skip empty results; don't pollute compression-ratio stats
        rows.append({
            "qid": qid,
            "raw_word_count": raw_tokens,
            "compressed_word_count": comp_tokens,
            "ratio": comp_tokens / max(raw_tokens, 1),
            "completion_tokens": usage.completion_tokens if usage else 0,
            "prompt_tokens":     usage.prompt_tokens     if usage else 0,
        })
    except Exception as e:
        print(f"  [ERROR] qid={qid}: {type(e).__name__}: {e}")
        failures.append({"qid": qid, "reason": f"{type(e).__name__}: {e}"})
        continue
    if (len(rows) + len(failures)) % 10 == 0:
        print(f"  {len(rows) + len(failures)}/{len(sample_qids)}  (ok={len(rows)}, failed={len(failures)})")

elapsed = time.time() - t0
n_ok = len(rows)
summary = {
    "n_attempted": len(sample_qids),
    "n_ok":        n_ok,
    "n_failed":    len(failures),
    "mean_raw_words":         sum(r["raw_word_count"]        for r in rows) / max(n_ok, 1),
    "mean_compressed_words":  sum(r["compressed_word_count"] for r in rows) / max(n_ok, 1),
    "mean_compression_ratio": sum(r["ratio"]                 for r in rows) / max(n_ok, 1),
    "mean_completion_tokens": sum(r["completion_tokens"]     for r in rows) / max(n_ok, 1),
    "total_wall_sec":  elapsed,
    "per_query_sec":   elapsed / max(len(sample_qids), 1),
}
Path("results").mkdir(exist_ok=True)
Path("results/compression_metrics.json").write_text(
    json.dumps({"summary": summary, "rows": rows, "failures": failures}, indent=2)
)
print(json.dumps(summary, indent=2))
if failures:
    print(f"\n{len(failures)} failures. First 3:")
    for f in failures[:3]:
        print(f"  qid={f['qid']}: {f['reason']}")