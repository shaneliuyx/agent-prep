"""
01_rerank_mlx.py — Two-stage retrieval, MLX reranker (mirrors 01_rerank.py exactly).

Same collection, same queries, same metric formulas as 01_rerank.py — the only
difference is Stage 3 uses MLX cross-encoder instead of PyTorch+MPS+fp16.
This makes the two scripts' results/*.json outputs directly comparable.

Run both, then diff:
    python src/01_rerank.py            # writes results/rerank_metrics.json     (PyTorch)
    python src/01_rerank_mlx.py        # writes results/rerank_mlx_metrics.json (MLX)
    diff results/rerank_metrics.json results/rerank_mlx_metrics.json

For fast iteration during development, use the SAMPLE_QUERIES env var:
    SAMPLE_QUERIES=50 python src/01_rerank_mlx.py    # 50-query sample (~30s)
    python src/01_rerank_mlx.py                       # full 6,980 queries

Dependencies (one-time):
    uv pip install -U mlx mlx-lm transformers
    # Optional: uv pip install mlx-embeddings  (preferred loader if available)

Model conversion (one-time, ~30s on first run):
    Auto-converts ~/models/bge-reranker-v2-m3 to MLX format if not already done,
    storing in ~/models/bge-reranker-v2-m3-mlx. If mlx-community/bge-reranker-v2-m3
    exists on HF, that's used instead.

Output schema matches 01_rerank.py exactly, plus an "mlx" config block:
    {
      "baseline_dense_top5":     {"recall@5": ..., "ndcg@5": ...},
      "dense_top50_rerank_top5": {"recall@5": ..., "ndcg@5": ...},
      "stage_seconds":           {"encode": ..., "retrieve": ..., "rerank": ..., "total": ...},
      "n_queries":               6980,
      "engine":                  "mlx",
      "config":                  {"encode_batch": 128, "qdrant_batch": 64, ...},
      "mlx":                     {"load_method": "mlx-lm", "mlx_batch": 128, ...}
    }
"""
import json
import math
import os
import time
from pathlib import Path

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import QueryRequest
from model_config import BGE_M3_HNSW, BGE_RERANKER_V2_M3

# === Configuration — match 01_rerank.py exactly ===
C, M, R = BGE_M3_HNSW, BGE_M3_HNSW.model, BGE_RERANKER_V2_M3
TOP_N = R.max_pairs_per_query  # 50
TOP_K = 5

# Encoder + retrieval batching (identical to 01_rerank.py)
ENCODE_BATCH = 128
QDRANT_BATCH = 64

# MLX-specific reranker tuning
MLX_BATCH         = 128  # cross-encoder GPU batch within MLX
MLX_GROUP_QUERIES = 1    # queries per predict() call (start simple; tune later)

# Optional sample size for fast iteration (set via env var SAMPLE_QUERIES=50)
_sample = os.getenv("SAMPLE_QUERIES")
SAMPLE_QUERIES = int(_sample) if _sample else None

# MLX model paths
MLX_LOCAL_DIR = Path.home() / "models" / "bge-reranker-v2-m3-mlx"
MLX_HF_REPO   = "mlx-community/bge-reranker-v2-m3"

print("=" * 60)
print("MLX reranker — full pipeline mirroring 01_rerank.py")
print("=" * 60)


# === Load PyTorch encoder + Qdrant (Stage 1+2 unchanged from 01_rerank.py) ===
qd = QdrantClient(url="http://127.0.0.1:6333", timeout=60)
encoder = SentenceTransformer(M.path, device="mps", trust_remote_code=M.trust_remote_code)

queries = json.loads(Path("data/queries.json").read_text())
qrels   = json.loads(Path("data/qrels.json").read_text())

qids = list(queries.keys())
if SAMPLE_QUERIES is not None:
    qids = qids[:SAMPLE_QUERIES]
qtexts = [queries[qid] for qid in qids]
N = len(qids)
print(f"queries: {N}  encoder: {M.name}  reranker: {R.name} (MLX)")


# === Stage 1: batch-encode ALL queries (identical to 01_rerank.py) ===
t0 = time.perf_counter()
qvecs = encoder.encode(
    [M.query_prefix + q for q in qtexts],
    normalize_embeddings=True,
    batch_size=ENCODE_BATCH,
    show_progress_bar=True,
    convert_to_numpy=True,
)
t_encode = time.perf_counter() - t0
print(f"[1/3] encoded {N} queries in {t_encode:.1f}s ({N/t_encode:.0f} q/s)")


# === Stage 2: bulk retrieve top-N (identical to 01_rerank.py) ===
t0 = time.perf_counter()
candidates: list[list[tuple[str, str, float]]] = [[] for _ in range(N)]
for start in range(0, N, QDRANT_BATCH):
    end = min(start + QDRANT_BATCH, N)
    requests = [
        QueryRequest(query=qvecs[i].tolist(), limit=TOP_N, with_payload=True)
        for i in range(start, end)
    ]
    batch = qd.query_batch_points(C.name, requests=requests)
    for offset, response in enumerate(batch):
        candidates[start + offset] = [
            (h.payload["doc_id"], h.payload["text"], h.score) for h in response.points
        ]
t_retrieve = time.perf_counter() - t0
print(f"[2/3] retrieved top-{TOP_N} for {N} queries in {t_retrieve:.1f}s ({N/t_retrieve:.0f} q/s)")


# === Stage 3: MLX rerank (the experiment — replaces PyTorch CrossEncoder.predict) ===
print(f"[3/3] reranking with MLX (group={MLX_GROUP_QUERIES}, batch={MLX_BATCH})...")


def load_mlx_reranker():
    """Try several loaders for the MLX cross-encoder. Returns (model, tokenizer, method_name)."""
    errors = []

    # Path 1: mlx-embeddings
    try:
        from mlx_embeddings.utils import load as mlx_load
        target = str(MLX_LOCAL_DIR) if MLX_LOCAL_DIR.exists() else MLX_HF_REPO
        model, tok = mlx_load(target)
        return model, tok, f"mlx-embeddings ({target})"
    except ImportError as e:
        errors.append(f"mlx-embeddings not installed: {e}")
    except Exception as e:
        errors.append(f"mlx-embeddings load failed: {e}")

    # Path 2: mlx-lm
    try:
        from mlx_lm.utils import load as mlx_load
        target = str(MLX_LOCAL_DIR) if MLX_LOCAL_DIR.exists() else MLX_HF_REPO
        model, tok = mlx_load(target)
        return model, tok, f"mlx-lm ({target})"
    except ImportError as e:
        errors.append(f"mlx-lm not installed: {e}")
    except Exception as e:
        errors.append(f"mlx-lm load failed: {e}")

    # Path 3: convert from local PyTorch checkpoint, then load
    if not MLX_LOCAL_DIR.exists():
        try:
            from mlx_lm import convert
            print(f"  converting {R.path} -> {MLX_LOCAL_DIR} (one-time, ~30s)...")
            convert(R.path, mlx_path=str(MLX_LOCAL_DIR), quantize=False)
            from mlx_lm.utils import load as mlx_load
            model, tok = mlx_load(str(MLX_LOCAL_DIR))
            return model, tok, f"mlx-lm post-convert ({MLX_LOCAL_DIR})"
        except Exception as e:
            errors.append(f"mlx_lm.convert failed: {e}")

    raise RuntimeError(
        "Could not load MLX reranker. Tried:\n  - "
        + "\n  - ".join(errors)
        + f"\n\nNext steps:\n"
        f"  1. Install:           uv pip install -U mlx mlx-embeddings mlx-lm transformers\n"
        f"  2. Verify HF repo:    https://huggingface.co/{MLX_HF_REPO}\n"
        f"  3. Manual convert:    python -m mlx_lm.convert --hf-path {R.path} --mlx-path {MLX_LOCAL_DIR}"
    )


mx = None
load_method = "n/a"
try:
    import mlx.core as _mx
    mx = _mx
    mlx_model, mlx_tok, load_method = load_mlx_reranker()
    print(f"  loaded via: {load_method}")
except Exception as e:
    print(f"\nFATAL: MLX path failed.\n{e}")
    raise SystemExit(1)


def rerank_mlx_batch(qtext: str, cands: list, batch_size: int = MLX_BATCH) -> list[float]:
    """Score (qtext, doc) pairs with MLX cross-encoder; return list of relevance logits.

    Output extraction handles three known mlx-embeddings sequence-classifier patterns:
      1. out.logits (HF-style classifier head)
      2. out.pooler_output (mlx-embeddings convention per Minos-v1 example)
      3. out (raw tensor, shape (B, num_labels))
    """
    assert mx is not None and mlx_model is not None
    q_chunk = [qtext] * len(cands)
    d_chunk = [c[1] for c in cands]
    scores: list[float] = []
    for s in range(0, len(cands), batch_size):
        encoded = mlx_tok(
            q_chunk[s : s + batch_size],
            d_chunk[s : s + batch_size],
            padding=True, truncation=True, max_length=512, return_tensors="np",
        )
        input_ids = mx.array(encoded["input_ids"])
        attention_mask = mx.array(encoded["attention_mask"])
        out = mlx_model(input_ids, attention_mask=attention_mask)
        # Try each output convention in order
        if hasattr(out, "logits"):
            logits = out.logits
        elif hasattr(out, "pooler_output"):
            logits = out.pooler_output
        else:
            logits = out
        # For binary relevance (num_labels=1) the score is column 0
        if logits.ndim == 1:
            cls = logits
        elif logits.ndim == 2:
            cls = logits[:, 0]
        else:
            cls = logits[:, 0, 0]
        mx.eval(cls)
        scores.extend(cls.tolist())
    return scores


# Warmup (5 queries) — same pattern as 01_rerank.py
for i in range(min(5, N)):
    rerank_mlx_batch(qtexts[i], candidates[i])

# Full rerank pass — mirrors 01_rerank.py's Stage 3 outer loop
t0 = time.perf_counter()
rerank_scores: list[list[float]] = [[0.0] * len(c) for c in candidates]
for g_start in range(0, N, MLX_GROUP_QUERIES):
    g_end = min(g_start + MLX_GROUP_QUERIES, N)
    # MLX_GROUP_QUERIES=1: one query per loop iteration (per-query padding, no cross-query mixing)
    for qi in range(g_start, g_end):
        rerank_scores[qi] = rerank_mlx_batch(qtexts[qi], candidates[qi])
t_rerank = time.perf_counter() - t0
total_pairs = sum(len(c) for c in candidates)
print(f"  reranked {N} × {TOP_N} pairs in {t_rerank:.1f}s "
      f"({N/t_rerank:.1f} q/s, {t_rerank/total_pairs*1000:.2f} ms/pair)")


# === Stage 4: compute metrics (identical formulas to 01_rerank.py) ===
def ndcg_at_k(hit_ids, gold, k):
    dcg  = sum(1.0 / math.log2(i + 2) for i, d in enumerate(hit_ids[:k]) if d in gold)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / idcg if idcg else 0.0

baseline_rows, rerank_rows = [], []
for qi, qid in enumerate(qids):
    gold = set(qrels[qid])
    cands = candidates[qi]
    baseline_ids = [c[0] for c in cands[:TOP_K]]
    rerank_top   = sorted(zip(cands, rerank_scores[qi]), key=lambda x: -x[1])[:TOP_K]
    rerank_ids   = [c[0] for c, _ in rerank_top]
    baseline_rows.append((qid,
        1.0 if set(baseline_ids) & gold else 0.0,
        ndcg_at_k(baseline_ids, gold, TOP_K),
    ))
    rerank_rows.append((qid,
        1.0 if set(rerank_ids) & gold else 0.0,
        ndcg_at_k(rerank_ids, gold, TOP_K),
    ))

def mean(col, rows):
    return sum(r[col] for r in rows) / len(rows)

results = {
    "baseline_dense_top5":     {"recall@5": mean(1, baseline_rows), "ndcg@5": mean(2, baseline_rows)},
    "dense_top50_rerank_top5": {"recall@5": mean(1, rerank_rows),   "ndcg@5": mean(2, rerank_rows)},
    "stage_seconds": {
        "encode":   round(t_encode,   1),
        "retrieve": round(t_retrieve, 1),
        "rerank":   round(t_rerank,   1),
        "total":    round(t_encode + t_retrieve + t_rerank, 1),
    },
    "n_queries": N,
    "engine":    "mlx",
    "config": {
        "encode_batch":  ENCODE_BATCH,
        "qdrant_batch":  QDRANT_BATCH,
    },
    "mlx": {
        "load_method":       load_method,
        "mlx_batch":         MLX_BATCH,
        "mlx_group_queries": MLX_GROUP_QUERIES,
        "ms_per_pair":       round(t_rerank / total_pairs * 1000, 2),
    },
}

Path("results").mkdir(exist_ok=True)
out_path = Path("results/rerank_mlx_metrics.json")
out_path.write_text(json.dumps(results, indent=2))
print(f"\nSaved to {out_path}")
print(json.dumps(results, indent=2))


# === Side-by-side comparison with PyTorch baseline if available ===
pt_path = Path("results/rerank_metrics.json")
if pt_path.exists():
    pt = json.loads(pt_path.read_text())
    if pt.get("n_queries") == N:
        print("\n" + "=" * 60)
        print(f"Comparison vs PyTorch baseline ({pt_path})")
        print("=" * 60)

        pt_rerank = pt["dense_top50_rerank_top5"]
        ml_rerank = results["dense_top50_rerank_top5"]
        pt_stages = pt["stage_seconds"]
        ml_stages = results["stage_seconds"]

        print(f"  metric                        PyTorch        MLX            Δ")
        print(f"  recall@5                      {pt_rerank['recall@5']:.6f}     {ml_rerank['recall@5']:.6f}     {ml_rerank['recall@5'] - pt_rerank['recall@5']:+.6f}")
        print(f"  nDCG@5                        {pt_rerank['ndcg@5']:.6f}     {ml_rerank['ndcg@5']:.6f}     {ml_rerank['ndcg@5'] - pt_rerank['ndcg@5']:+.6f}")
        print(f"  rerank_sec                    {pt_stages['rerank']:>7.1f}        {ml_stages['rerank']:>7.1f}        {ml_stages['rerank'] / pt_stages['rerank']:.2f}× of PyTorch")
        print(f"  total_sec                     {pt_stages['total']:>7.1f}        {ml_stages['total']:>7.1f}        {ml_stages['total'] / pt_stages['total']:.2f}× of PyTorch")
        speedup = pt_stages["rerank"] / ml_stages["rerank"]
        print(f"\n  MLX rerank speedup vs PyTorch: {speedup:.2f}×")

        recall_drift = abs(ml_rerank["recall@5"] - pt_rerank["recall@5"])
        ndcg_drift   = abs(ml_rerank["ndcg@5"]   - pt_rerank["ndcg@5"])
        if recall_drift < 1e-3 and ndcg_drift < 1e-3:
            print(f"  ✓ MLX is a drop-in replacement (recall Δ={recall_drift:.6f}, nDCG Δ={ndcg_drift:.6f})")
        elif recall_drift < 5e-3:
            print(f"  ⚠  Metrics close but not identical — verify ranking quality")
        else:
            print(f"  ✗ Metrics diverge significantly — likely a model-conversion bug")
    else:
        print(f"\n  Note: {pt_path} has n_queries={pt.get('n_queries')} but this run has {N}; can't compare directly.")
else:
    print(f"\n  Note: no PyTorch baseline at {pt_path}. Run 01_rerank.py first to enable side-by-side comparison.")
