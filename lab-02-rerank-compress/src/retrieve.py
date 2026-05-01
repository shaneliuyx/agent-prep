"""Vector-RAG search-with-rerank wrapper. Public API for downstream comparison.

Two retrieval modes, auto-detected from the target Qdrant collection schema:

1. **Dense-only** (e.g. `bge_m3_hnsw`, `tech_corpus_hnsw`):
     BGE-M3 dense kNN → cross-encoder rerank → top-K → LLM answer

2. **Hybrid dense+sparse** (e.g. `bge_m3_hybrid`, `tech_corpus_hybrid`):
     BGE-M3 dense + sparse one-pass encode → query both named vectors →
     RRF fuse rankings (Cormack 2009, k=60) → cross-encoder rerank →
     top-K → LLM answer

The hybrid path catches lexical-match edge cases (rare proper nouns,
acronyms, exact phrases) that dense embeddings dilute across 1024
dimensions. Production GraphRAG benchmarks (Microsoft GraphRAG, Sarmah
et al. 2024 HybridRAG) use hybrid as the default vector baseline.

Set `QDRANT_COLLECTION` env var to choose collection. Mode is inferred
from the collection's vectors_config — no separate flag required.
"""
from __future__ import annotations

import os

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import NamedSparseVector, NamedVector, SparseVector
from sentence_transformers import CrossEncoder

from model_config import BGE_M3_HNSW, BGE_M3_HYBRID, BGE_RERANKER_V2_M3

SONNET = os.getenv("MODEL_SONNET", "gemma-4-26B-A4B-it-heretic-4bit")
_M, _R = BGE_M3_HNSW.model, BGE_RERANKER_V2_M3
_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", BGE_M3_HNSW.name)
_RRF_K = 60  # Cormack et al. 2009 default; rank-domain fusion constant.

omlx = OpenAI(base_url=os.getenv("OMLX_BASE_URL"), api_key=os.getenv("OMLX_API_KEY"))
qd = QdrantClient(url="http://127.0.0.1:6333")
reranker = CrossEncoder(_R.path, device="mps")
reranker.model.half()


def _is_hybrid_collection(client: QdrantClient, name: str) -> bool:
    """Inspect collection schema to determine retrieval mode.

    Hybrid collections have a sparse_vectors_config with at least one
    named sparse vector. Dense-only collections have only a regular
    vectors_config. Returns True for hybrid, False for dense-only."""
    info = client.get_collection(name)
    sparse_cfg = getattr(info.config.params, "sparse_vectors", None)
    return bool(sparse_cfg)


_IS_HYBRID = _is_hybrid_collection(qd, _COLLECTION_NAME)
print(f"[retrieve] collection={_COLLECTION_NAME!r} mode={'hybrid' if _IS_HYBRID else 'dense'}")


# Encoder: hybrid mode needs BGE-M3 with sparse output enabled.
# Dense-only mode uses SentenceTransformer (lighter, no sparse weights).
if _IS_HYBRID:
    from FlagEmbedding import BGEM3FlagModel
    encoder = BGEM3FlagModel(_M.path, use_fp16=False, device="mps")
else:
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(_M.path, device="mps", trust_remote_code=_M.trust_remote_code)


_ANSWER_PROMPT = """Using ONLY the context below, answer the query in 1-3 sentences. If the context doesn't contain the answer, reply exactly: insufficient context.

Context:
{ctx}

Query: {q}
Answer:"""


def _retrieve_dense(q: str, top_n: int) -> list:
    qv = encoder.encode([_M.query_prefix + q], normalize_embeddings=True)[0]
    return qd.query_points(
        _COLLECTION_NAME, query=qv.tolist(), limit=top_n, with_payload=True,
    ).points


def _retrieve_hybrid(q: str, top_n: int) -> list:
    """Run dense and sparse queries against the hybrid collection, fuse via RRF.

    Both vectors come from a single BGE-M3 forward pass. Each ranking
    contributes a 1/(_RRF_K + rank) score; the fused list is sorted
    by aggregate score before reranking."""
    out = encoder.encode(
        [q],
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    dense_vec = out["dense_vecs"][0].tolist()
    sparse_dict = out["lexical_weights"][0]

    dense_pts = qd.query_points(
        _COLLECTION_NAME,
        query=NamedVector(name=BGE_M3_HYBRID.dense_vector_name, vector=dense_vec),
        limit=top_n,
        with_payload=True,
    ).points
    sparse_pts = qd.query_points(
        _COLLECTION_NAME,
        query=NamedSparseVector(
            name=BGE_M3_HYBRID.sparse_vector_name,
            vector=SparseVector(
                indices=list(map(int,   sparse_dict.keys())),
                values =list(map(float, sparse_dict.values())),
            ),
        ),
        limit=top_n,
        with_payload=True,
    ).points

    # Reciprocal Rank Fusion (Cormack et al. 2009). k=60 default.
    scores: dict[int, float] = {}
    pt_by_id: dict[int, object] = {}
    for rank, pt in enumerate(dense_pts):
        pid = pt.id
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (_RRF_K + rank + 1)
        pt_by_id[pid] = pt
    for rank, pt in enumerate(sparse_pts):
        pid = pt.id
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (_RRF_K + rank + 1)
        pt_by_id.setdefault(pid, pt)

    fused_ids = sorted(scores.keys(), key=lambda pid: -scores[pid])[:top_n]
    return [pt_by_id[pid] for pid in fused_ids]


def _retrieve(q: str, top_n: int) -> list:
    return _retrieve_hybrid(q, top_n) if _IS_HYBRID else _retrieve_dense(q, top_n)


def _rerank(q: str, cands: list, k: int) -> list[tuple[str, str, float]]:
    pairs = [(q, c.payload["text"]) for c in cands]
    scores = reranker.predict(pairs, batch_size=_R.batch_size)
    order = sorted(zip(cands, scores), key=lambda x: -x[1])[:k]
    return [(c.payload["doc_id"], c.payload["text"], float(s)) for c, s in order]


def search_with_rerank(query: str, k: int = 5, top_n: int | None = None) -> dict:
    """Run vector retrieval + cross-encoder rerank + LLM answer on one query.

    Auto-routes to hybrid (dense+sparse+RRF) or dense-only retrieval based
    on the configured collection's schema. Reranking + LLM-answer steps
    are identical across modes."""
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
