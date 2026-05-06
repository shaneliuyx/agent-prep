"""Pipeline variant: multi-query fusion with RRF.

v2 (2026-05-06): inherits the migrated 02_pipeline.py — `_enc` is now a
shared/rag_hybrid `DenseEncoder` (autoconfig'd device + batch tier), `qd`
is the same QdrantClient. Multi-query rewrite logic + RRF formula
unchanged — migration was at the encoder layer, not the fusion layer.

Encode kwarg note: shared DenseEncoder.encode() takes `normalize=True`
(default), NOT `normalize_embeddings=True` (the SentenceTransformer kwarg).
"""
import json
import os
from collections import defaultdict
from openai import OpenAI
from src.script_wrap import load

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


def rrf(result_lists, k=60):
    """Reciprocal Rank Fusion. result_lists = [[hit, ...], [hit, ...]]."""
    scores = defaultdict(float)
    lookup = {}
    for hits in result_lists:
        for rank, h in enumerate(hits):
            scores[h.id] += 1.0 / (k + rank + 1)
            lookup[h.id] = h
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [lookup[i] for i, _ in ranked]


def run_pipeline_mq(q):
    rewrites_list = rewrites(q)
    qs = [q] + rewrites_list
    lists = []
    for qq in qs:
        # rag_hybrid.DenseEncoder.encode kwarg is `normalize=True` (default),
        # not SentenceTransformer's `normalize_embeddings=True`.
        qv = _enc.encode([qq])[0]
        lists.append(qd.query_points("bge_m3_hnsw", query=qv.tolist(), limit=20, with_payload=True).points)
    fused = rrf(lists)[:30]
    top = rerank(q, fused, k=5)
    ans, _ = answer_from(q, top)
    return {
        "question": q,
        "answer": ans,
        "contexts": [h.payload["text"] for h in top],
        "context_ids": [h.payload["doc_id"] for h in top],
        "rewrites": rewrites_list,  # mirrors 03_hyde's `hypothetical` for debug audit
    }
