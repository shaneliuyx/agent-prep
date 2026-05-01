"""Vector-RAG search-with-rerank wrapper. Public API for downstream comparison.

Auto-detects retrieval mode from the target Qdrant collection schema:

  - Dense-only collection (e.g. ``bge_m3_hnsw``, ``tech_corpus_hnsw``):
      BGE-M3 dense kNN → cross-encoder rerank → top-K → LLM answer

  - Hybrid dense+sparse collection (e.g. ``bge_m3_hybrid``,
    ``tech_corpus_hybrid``): BGE-M3 dense + sparse one-pass encode →
    query both named vectors → manual RRF fuse (Cormack 2009, k=60) →
    cross-encoder rerank → top-K → LLM answer

The hybrid path catches lexical-match edge cases (rare proper nouns,
acronyms, exact phrases) that dense embeddings dilute across 1024
dimensions.

Set ``QDRANT_COLLECTION`` env var to choose collection — mode is inferred
from the collection's vectors_config, no separate flag required.

Library-driven version: this file is a thin orchestrator over
shared/rag_hybrid. The HybridEncoder / CrossEncoderReranker / Retriever
classes are lazy — model loads happen on first call, not at import.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from openai import OpenAI
from qdrant_client import QdrantClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))

import dataclasses

from rag_hybrid import (  # noqa: E402
    BGE_M3_HNSW,
    BGE_RERANKER_V2_M3,
    CrossEncoderReranker,
    FusionStrategy,
    HybridEncoder,
    RetrieveConfig,
    Retriever,
    autoconfig,
)

SONNET = os.getenv("MODEL_SONNET", "gemma-4-26B-A4B-it-heretic-4bit")
_M = BGE_M3_HNSW.model
_R = BGE_RERANKER_V2_M3
_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", BGE_M3_HNSW.name)

omlx = OpenAI(base_url=os.getenv("OMLX_BASE_URL"), api_key=os.getenv("OMLX_API_KEY"))
qd = QdrantClient(url="http://127.0.0.1:6333")

# System-aware config: probes device + memory + CPU and produces matching
# encoder/reranker/ingest configs. On MPS this gives encoder fp16=False
# (BGE-M3 sparse-head NaN risk) and reranker fp16=True (cross-encoder safe).
# On CUDA both flip to True. autoconfig.notes is the audit trail.
_cfg = autoconfig.recommend(_M, _R)

# Reranker fp16 stays opt-in to preserve the frozen comparison.json hash.
# autoconfig already recommends True on MPS/CUDA; explicit replace() makes
# the choice visible at the callsite.
_encoder = HybridEncoder(_cfg.encoder)
_reranker = CrossEncoderReranker(dataclasses.replace(_cfg.reranker, fp16=True))

# fusion=NATIVE_RRF — server-side fusion via Qdrant Prefetch + FusionQuery.
# Same RRF formula (Cormack 2009, k=60) as MANUAL_RRF; tied scores may
# break differently but the metric output is equivalent. One HTTP roundtrip
# instead of two. The frozen comparison.json was produced against a dense
# collection (tech_corpus_hnsw), so this swap doesn't affect that hash —
# fusion only kicks in for hybrid collections.
_retriever = Retriever(
    qd=qd,
    collection=_COLLECTION_NAME,
    encoder=_encoder,
    cfg=RetrieveConfig(fusion=FusionStrategy.NATIVE_RRF),
)
print(f"[retrieve] collection={_COLLECTION_NAME!r} mode={_retriever.mode} | " + " | ".join(_cfg.notes))


_ANSWER_PROMPT = """Using ONLY the context below, answer the query in 1-3 sentences. If the context doesn't contain the answer, reply exactly: insufficient context.

Context:
{ctx}

Query: {q}
Answer:"""


def search_with_rerank(query: str, k: int = 5, top_n: int | None = None) -> dict:
    """Run vector retrieval + cross-encoder rerank + LLM answer on one query.

    Auto-routes to hybrid (dense+sparse+RRF) or dense-only retrieval based
    on the configured collection's schema. Reranking + LLM-answer steps
    are identical across modes.
    """
    top = _retriever.search_with_rerank(query, _reranker, k=k, top_n=top_n)
    ctx = "\n\n".join(f"[{did}] {text}" for did, text, _ in top)
    resp = omlx.chat.completions.create(
        model=SONNET,
        messages=[{"role": "user", "content": _ANSWER_PROMPT.format(ctx=ctx, q=query)}],
        temperature=0.0,
        max_tokens=400,
    )
    answer_text = (resp.choices[0].message.content or "").strip()
    return {"answer": answer_text, "chunks": top}
