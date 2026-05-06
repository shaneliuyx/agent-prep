"""The pipeline under test. Returns (answer, retrieved_contexts, ...).

v2 (2026-05-06): migrated to shared/rag_hybrid library — autoconfig handles
device + memory tier; lazy-loaded encoder + reranker. Pipeline contract
(retrieve → rerank → answer_from → run_pipeline) preserved unchanged so
03_hyde.py, 04_multiquery.py, 05_trace.py continue to work without edits.

Drift fixed by this migration:
- Hardcoded device="mps" → autoconfig.probe_system() (mps / cuda / cpu)
- Hardcoded encode batch → autoconfig memory tier (32 / 64 / 128)
- Hand-loaded SentenceTransformer + CrossEncoder → DenseEncoder + CrossEncoderReranker
  (lazy load — module import no longer pulls 2 GB of weights)
- Hardcoded fp16 off → autoconfig.fp16_safe_for_encoder / .fp16_safe_for_reranker
  (BGE-M3 NaN-clean on MPS per shared/tests/probe_bge_m3_fp16_mps.py;
  cross-encoder fp16-safe on MPS + CUDA — ~30% rerank latency win)
- Pre-existing curly close-quote bug on prompt line repaired
"""
import os
import sys
from pathlib import Path

from openai import OpenAI
from qdrant_client import QdrantClient

# Bootstrap shared/ onto sys.path so `import rag_hybrid` resolves from any
# project layout (lab-03 lives one level deeper than shared/).
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "shared"))

from rag_hybrid import (  # noqa: E402
    BGE_M3,
    BGE_RERANKER_V2_M3,
    CrossEncoderReranker,
    DenseEncoder,
    autoconfig,
)

omlx = OpenAI(base_url=os.getenv("OMLX_BASE_URL"), api_key=os.getenv("OMLX_API_KEY"))
SONNET = os.getenv("MODEL_SONNET")

qd = QdrantClient(url="http://127.0.0.1:6333")

# autoconfig probes device (mps/cuda/cpu) + memory tier (encode_batch
# 32/64/128). DenseEncoder for the dense-only `bge_m3_hnsw` collection —
# avoids pulling FlagEmbedding (only needed for hybrid sparse). Both
# wrappers lazy-load: model weights stay off disk until first .encode() /
# .rerank() call.
_enc = DenseEncoder(autoconfig.encoder_config_for(BGE_M3))

# Reranker config from the same probe used by the encoder. Cross-encoder
# fp16 is safe on MPS + CUDA (no NaN risk like BGE-M3 sparse head); autoconfig
# enables fp16 wherever safe. ~30% latency win on M-series.
_recommended = autoconfig.recommend(BGE_M3, BGE_RERANKER_V2_M3)
_rr = CrossEncoderReranker(_recommended.reranker)

# v2 prompt — adopted 2026-05-06 after the v1 (concise) → v1.5 (enhanced, no length cap)
# → v2 (enhanced + strict one-sentence < 35-words) A/B sweep on the harder dev set.
# v1.5 was too verbose (faithfulness 0.94, answer_relevancy 0.65); v2 recovered
# faithfulness to 0.99 while lifting answer_relevancy to 0.7494.
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


def retrieve(q, n=30):
    """Dense kNN retrieval over `bge_m3_hnsw`. Returns Qdrant ScoredPoints
    (preserved interface — downstream consumers read `.payload['text']` and
    `.payload['doc_id']`)."""
    qv = _enc.encode([q])[0]  # autoconfig batch_size; normalize=True default
    return qd.query_points(
        "bge_m3_hnsw",
        query=qv.tolist(),
        limit=n,
        with_payload=True,
    ).points


def rerank(q, hits, k=5):
    """Cross-encoder rerank. Preserves Qdrant ScoredPoint output shape so
    03_hyde.py, 04_multiquery.py, 05_trace.py continue to work without edits.

    Uses the lazy-loaded shared CrossEncoderReranker model directly via
    `_model.predict` rather than `.rerank()` (which returns triples) — this
    keeps the contract that downstream code expects ScoredPoints, not
    `(doc_id, text, score)` tuples.
    """
    _rr._ensure_loaded()
    pairs = [(q, h.payload["text"]) for h in hits]
    scores = _rr._model.predict(pairs, batch_size=_rr.cfg.spec.batch_size)
    ordered = sorted(zip(hits, scores), key=lambda x: -x[1])[:k]
    return [h for h, _ in ordered]


def answer_from(q, hits):
    ctx = "\n\n".join(h.payload["text"] for h in hits)
    # v2: max_tokens=120 (down from 200). One-sentence answer never needs more
    # than ~120 tokens; tighter cap reduces verbosity-induced faithfulness loss.
    r = omlx.chat.completions.create(
        model=SONNET,
        temperature=0.0,
        max_tokens=120,
        messages=[{"role": "user", "content": ANSWER.format(ctx=ctx, q=q)}],
    )
    return r.choices[0].message.content.strip(), ctx


def run_pipeline(q):
    cands = retrieve(q, n=30)
    top = rerank(q, cands, k=5)
    ans, ctx = answer_from(q, top)
    return {
        "question": q,
        "answer": ans,
        "contexts": [h.payload["text"] for h in top],
        "context_ids": [h.payload["doc_id"] for h in top],
    }
