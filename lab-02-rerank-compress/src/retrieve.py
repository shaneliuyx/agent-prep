"""Vector-RAG search-with-rerank wrapper.

Public API for downstream comparison work (e.g. lab-02.5's `compare.py`).
Wraps the same BGE-M3 dense + Qdrant HNSW + BGE-reranker-v2-m3 + LLM-answer
pipeline that 02b_answer_eval.py uses, but exposed as a single
`search_with_rerank(query, k=5) -> {"answer", "chunks"}` callable.

Why a separate module instead of importing from 02b_answer_eval.py:
- Module names cannot start with digits in Python — `import 02b_answer_eval`
  is a syntax error.
- 02b_answer_eval.py runs the full eval at module-import time (it is a
  script, not a library). Importing it would trigger a 30-query eval as a
  side effect — wrong behavior for callers that just want a single search.

This module's import-time side effects are limited to: connecting to the
local Qdrant, loading the BGE-M3 encoder onto MPS, and loading the
cross-encoder reranker. All three are required for `search_with_rerank`
to function and are cached for the process lifetime."""
from __future__ import annotations

import os

from openai import OpenAI
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder, SentenceTransformer

from model_config import BGE_M3_HNSW, BGE_RERANKER_V2_M3

# Same model + collection configuration as 02b_answer_eval.py, but
# QDRANT_COLLECTION env var lets callers override the default collection
# (e.g. lab-02.5/compare.py points at `tech_corpus_hnsw` for fair head-
# to-head against GraphRAG on the same corpus).
SONNET = os.getenv("MODEL_SONNET", "gemma-4-26B-A4B-it-heretic-4bit")
_M, _R = BGE_M3_HNSW.model, BGE_RERANKER_V2_M3
_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", BGE_M3_HNSW.name)

omlx = OpenAI(base_url=os.getenv("OMLX_BASE_URL"), api_key=os.getenv("OMLX_API_KEY"))
qd = QdrantClient(url="http://127.0.0.1:6333")
encoder = SentenceTransformer(_M.path, device="mps", trust_remote_code=_M.trust_remote_code)
reranker = CrossEncoder(_R.path, device="mps")
reranker.model.half()


_ANSWER_PROMPT = """Using ONLY the context below, answer the query in 1-3 sentences. If the context doesn't contain the answer, reply exactly: insufficient context.

Context:
{ctx}

Query: {q}
Answer:"""


def _retrieve(q: str, top_n: int) -> list:
    qv = encoder.encode([_M.query_prefix + q], normalize_embeddings=True)[0]
    return qd.query_points(_COLLECTION_NAME, query=qv.tolist(), limit=top_n, with_payload=True).points


def _rerank(q: str, cands: list, k: int) -> list[tuple[str, str, float]]:
    pairs = [(q, c.payload["text"]) for c in cands]
    scores = reranker.predict(pairs, batch_size=_R.batch_size)
    order = sorted(zip(cands, scores), key=lambda x: -x[1])[:k]
    return [(c.payload["doc_id"], c.payload["text"], float(s)) for c, s in order]


def search_with_rerank(query: str, k: int = 5, top_n: int | None = None) -> dict:
    """Run the W2 hybrid retrieval + rerank + LLM-answer pipeline on one query.

    Returns dict with:
      - `answer`: LLM-generated 1-3 sentence answer grounded in the top-k chunks
      - `chunks`: list of (doc_id, text, rerank_score) tuples used as context

    `top_n` controls retrieval breadth before rerank; defaults to the lab's
    configured cross-encoder pair budget (R.max_pairs_per_query)."""
    cands = _retrieve(query, top_n or _R.max_pairs_per_query)
    top = _rerank(query, cands, k)
    ctx = "\n\n".join(f"[{did}] {text}" for did, text, _ in top)
    resp = omlx.chat.completions.create(
        model=SONNET,
        messages=[{"role": "user", "content": _ANSWER_PROMPT.format(ctx=ctx, q=query)}],
        temperature=0.0,
        max_tokens=400,
    )
    answer_text = (resp.choices[0].message.content or "").strip()
    return {"answer": answer_text, "chunks": top}
