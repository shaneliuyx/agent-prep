"""Pipeline variant: multi-query fusion with RRF.

v3 (2026-05-06): hand-rolled `rrf` function dropped — now uses
shared/rag_hybrid.rrf_fuse (identical formula, k=60 default per Cormack
et al. SIGIR 2009). Encoder/reranker/qdrant inherit from migrated
02_pipeline.py.

Encode kwarg note: shared DenseEncoder.encode() takes `normalize=True`
(default), NOT `normalize_embeddings=True` (the SentenceTransformer kwarg).
"""
import json
import os
import sys
from pathlib import Path

from openai import OpenAI

# Bootstrap shared/ onto sys.path BEFORE importing rag_hybrid.
# (02_pipeline.py also adds this path, but it loads later via script_wrap;
# we need rag_hybrid available at module-top import time.)
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "shared"))

from rag_hybrid import rrf_fuse  # noqa: E402

from src.script_wrap import load  # noqa: E402

# Universal loader from §2.2b — same pattern 03_hyde.py uses.
pipeline = load("02_pipeline.py")
_enc = pipeline._enc          # rag_hybrid.DenseEncoder (autoconfig'd)
qd = pipeline.qd
rerank = pipeline.rerank
answer_from = pipeline.answer_from

omlx = OpenAI(base_url=os.getenv("OMLX_BASE_URL"), api_key=os.getenv("OMLX_API_KEY"))
SONNET = os.getenv("MODEL_SONNET")

REWRITE_PROMPT = """Rewrite the question 3 different ways that preserve meaning but use different phrasings and keywords. Output JSON:
{{"rewrites": ["...", "...", "..."]}}

Question: {q}"""


def rewrites(q):
    r = omlx.chat.completions.create(
        model=SONNET,
        temperature=0.3,
        max_tokens=300,
        messages=[{"role": "user", "content": REWRITE_PROMPT.format(q=q)}],
        response_format={"type": "json_object"},
    )
    return json.loads(r.choices[0].message.content).get("rewrites", [])[:3]


def run_pipeline_mq(q):
    rewrites_list = rewrites(q)
    qs = [q] + rewrites_list
    lists = []
    for qq in qs:
        # rag_hybrid.DenseEncoder.encode kwarg is `normalize=True` (default),
        # not SentenceTransformer's `normalize_embeddings=True`.
        qv = _enc.encode([qq])[0]
        lists.append(qd.query_points("bge_m3_hnsw", query=qv.tolist(), limit=20, with_payload=True).points)
    # rrf_fuse: shared/rag_hybrid implementation, identical formula
    # (1.0 / (k + rank + 1), k=60 default per Cormack et al. SIGIR 2009).
    # Drop-in replacement for the prior hand-rolled rrf().
    fused = rrf_fuse(lists)[:30]
    top = rerank(q, fused, k=5)
    ans, _ = answer_from(q, top)
    return {
        "question": q,
        "answer": ans,
        "contexts": [h.payload["text"] for h in top],
        "context_ids": [h.payload["doc_id"] for h in top],
        "rewrites": rewrites_list,  # mirrors 03_hyde's `hypothetical` for debug audit
    }
