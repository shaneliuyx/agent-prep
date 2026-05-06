"""Multi-query fusion variant — PRE-MIGRATION reproduction.

This is a one-off standalone script for capturing the pre-migration
multi-query baseline AFTER migration has already shipped. It exists ONLY
to fill the missing "Run PreM-MQ" measurement so the post-migration run
(`04_multiquery.py` + `04b_ragas_multiquery.py`) has a within-±0.02 anchor
to validate against, mirroring the Run 4 → Run 5 contract for baseline
and Run pre/post for HyDE.

Pre-migration behavior recreated here:
- SentenceTransformer (NOT DenseEncoder) — hardcoded device="mps", normalize_embeddings=True
- CrossEncoder (NOT CrossEncoderReranker) — hardcoded device="mps", batch_size=32 fp32
- Hand-rolled RRF (NOT shared rrf_fuse) — last-write-wins per id
- No `rewrites` field in output dict (mirrors the old return shape)
- Direct Qdrant query_points (no Retriever wrapper)

Do NOT use this for production. Use it once, capture numbers, archive
the result file, leave the script in repo as the historical reproduction
artifact. After Run PreM-MQ is captured, this file is dead code — fine
to delete or keep as historical reference.

Run from project root:
    cd ~/code/agent-prep/lab-03-rag-eval
    set -a; source ../.env; set +a
    python src/04b_ragas_multiquery_premigration.py
"""
import json
import os
from collections import defaultdict

from openai import OpenAI
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder, SentenceTransformer

HOME = os.path.expanduser("~")
omlx = OpenAI(base_url=os.getenv("OMLX_BASE_URL"), api_key=os.getenv("OMLX_API_KEY"))
SONNET = os.getenv("MODEL_SONNET")

qd = QdrantClient(url="http://127.0.0.1:6333")

# Pre-migration: hand-loaded models, hardcoded device, no autoconfig.
# This is the "before" state we need to measure. Eager load (not lazy)
# matches pre-migration import-time behavior.
_enc = SentenceTransformer(f"{HOME}/models/bge-m3", device="mps", trust_remote_code=True)
_rr = CrossEncoder(f"{HOME}/models/bge-reranker-v2-m3", device="mps")

# Pre-migration: same v2 prompt — the prompt change is NOT part of the
# migration, so this stays identical to post-migration. Migration was
# encoder/reranker layer only, prompt was already v2.
ANSWER = """Use ONLY the context below.
Answer the exact question asked.
If the question asks why, give the reason.
If it asks how two things differ, state the contrast.
If it asks under what condition, state the condition.
Keep the answer concise, but include enough detail to directly satisfy the question.
If the context does not contain the answer, say exactly: insufficient context.
Answer in one sentence of fewer than 35 words.

Context:
{ctx}

Question: {q}
Answer:"""

REWRITE_PROMPT = """Rewrite the question 3 different ways that preserve meaning but use different phrasings and keywords. Output JSON:
{{"rewrites": ["...", "...", "..."]}}

Question: {q}"""


def retrieve(q, n=30):
    """Pre-migration retrieve: SentenceTransformer.encode with normalize_embeddings."""
    qv = _enc.encode([q], normalize_embeddings=True)[0]
    return qd.query_points(
        "bge_m3_hnsw",
        query=qv.tolist(),
        limit=n,
        with_payload=True,
    ).points


def rerank(q, hits, k=5):
    """Pre-migration rerank: CrossEncoder.predict with batch_size=32 fp32."""
    scores = _rr.predict([(q, h.payload["text"]) for h in hits], batch_size=32)
    ordered = sorted(zip(hits, scores), key=lambda x: -x[1])[:k]
    return [h for h, _ in ordered]


def answer_from(q, hits):
    ctx = "\n\n".join(h.payload["text"] for h in hits)
    r = omlx.chat.completions.create(
        model=SONNET,
        temperature=0.0,
        max_tokens=120,
        messages=[{"role": "user", "content": ANSWER.format(ctx=ctx, q=q)}],
    )
    return r.choices[0].message.content.strip(), ctx


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
    """Pre-migration RRF — hand-rolled. last-write-wins per id (subtle
    correctness diff vs shared rrf_fuse which uses first-write-wins).
    Identical formula otherwise: 1.0 / (k + rank + 1)."""
    scores = defaultdict(float)
    lookup = {}
    for hits in result_lists:
        for rank, h in enumerate(hits):
            scores[h.id] += 1.0 / (k + rank + 1)
            lookup[h.id] = h
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [lookup[i] for i, _ in ranked]


def run_pipeline_mq(q):
    """Pre-migration multi-query: rewrites + RRF fuse + rerank + answer.
    NOTE: no `rewrites` in output dict — matches pre-migration return shape."""
    qs = [q] + rewrites(q)
    lists = []
    for qq in qs:
        qv = _enc.encode([qq], normalize_embeddings=True)[0]
        lists.append(qd.query_points("bge_m3_hnsw", query=qv.tolist(), limit=20, with_payload=True).points)
    fused = rrf(lists)[:30]
    top = rerank(q, fused, k=5)
    ans, _ = answer_from(q, top)
    return {
        "question": q,
        "answer": ans,
        "contexts": [h.payload["text"] for h in top],
        "context_ids": [h.payload["doc_id"] for h in top],
    }
