"""Throughput-stack rerank — cross-query pair batching via CrossEncoder.

Same model, same dataset, same metrics as 01_rerank_via_rerankers.py — but uses
the throughput-shaped APIs:

  Encode    : SentenceTransformer.encode(list, batch=128)               ← bulk query encode
  Retrieve  : qdrant_client.query_batch_points(64 queries / HTTP call)  ← bulk top-50 lookup
  Rerank    : CrossEncoder.predict(flat_pair_list, batch=256) + fp16    ← cross-query pair batching
  Metrics   : ranx                                                      ← Numba JIT

Note: FlagReranker (FlagEmbedding) is the natural choice here but breaks on
transformers ≥ 5.x because it calls the removed `prepare_for_model` tokenizer
API. CrossEncoder (sentence_transformers) is the working alternative — same
model weights, same fp16 lever, lab-02-tested.

Demonstrates §7.5 "what does NOT transfer" finding empirically: rerankers'
per-query API surface (`.rank(query=, docs=)`) prevents cross-query batching;
FlagReranker's flat pair list lets you concatenate ~32 queries' pair-lists into
one .compute_score() call, amortizing GPU dispatch overhead across queries.

Reuses the `bge_m3_hnsw` collection (lab-01's MS MARCO 10K dense baseline).
"""
import json, os, sys, time
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import QueryRequest
from ranx import Qrels, Run, evaluate

# Suppress qdrant-client teardown warnings
def _hush_qdrant_unraisable(unraisable):
    tb = unraisable.exc_traceback
    if tb and "qdrant" in tb.tb_frame.f_code.co_filename:
        return
    sys.__unraisablehook__(unraisable)
sys.unraisablehook = _hush_qdrant_unraisable

HOME = os.path.expanduser("~")
COLLECTION = "bge_m3_hnsw"
TOP_N = 50          # candidates from dense; mirrors lab-02
TOP_K = 5           # final cut after rerank
QDRANT_BATCH = 64   # queries per HTTP roundtrip
ENCODE_BATCH = 128  # query encode batch
RERANK_BATCH = 256  # cross-encoder GPU batch (fp16 on M5 Pro)
GROUP_QUERIES = 32  # concatenate this many queries' pair-lists per .compute_score() call

queries = json.loads(Path("data/queries.json").read_text())
qrels   = json.loads(Path("data/qrels.json").read_text())

qids = list(queries.keys())
qtexts = [queries[qid] for qid in qids]
N = len(qids)
print(f"two-stage retrieve+rerank · {N} queries · top-{TOP_N}→top-{TOP_K} · throughput stack")

# ranx Qrels expects {qid: {doc_id: rel_score}}
qrels_obj = Qrels({qid: {doc_id: 1 for doc_id in gold} for qid, gold in qrels.items()})

# === Stage 1: bulk-encode all queries ===
t0 = time.time()
encoder = SentenceTransformer(f"{HOME}/models/bge-m3", device="mps", trust_remote_code=True)
qvecs = encoder.encode(qtexts, batch_size=ENCODE_BATCH, normalize_embeddings=True,
                       show_progress_bar=True, convert_to_numpy=True)
t_encode = time.time() - t0
print(f"[1/3] encoded {N} queries in {t_encode:.1f}s ({N/t_encode:.0f} q/s)")

# === Stage 2: bulk-retrieve top-N from Qdrant ===
qd = QdrantClient(url="http://127.0.0.1:6333", timeout=60)
t0 = time.time()
candidates: list[list[tuple[str, str]]] = [[] for _ in range(N)]
reqs = [QueryRequest(query=qvecs[i].tolist(), limit=TOP_N, with_payload=True)
        for i in range(N)]
for s in range(0, N, QDRANT_BATCH):
    batch = qd.query_batch_points(COLLECTION, requests=reqs[s : s + QDRANT_BATCH])
    for offset, response in enumerate(batch):
        candidates[s + offset] = [(h.payload["doc_id"], h.payload["text"]) for h in response.points]
t_retrieve = time.time() - t0
print(f"[2/3] retrieved top-{TOP_N} for {N} queries in {t_retrieve:.1f}s ({N/t_retrieve:.0f} q/s)")

# === Stage 3: cross-query batched rerank ===
t0 = time.time()
# CrossEncoder (sentence_transformers) — FlagReranker breaks on transformers 5.x
# (calls removed `prepare_for_model` API). CrossEncoder still works and supports
# the same fp16 lever via .model.half() — lab-02 §2.2.1's 2.86× speedup applies.
ranker = CrossEncoder(f"{HOME}/models/bge-reranker-v2-m3", device="mps")
ranker.model.half()
rerank_scores: list[list[float]] = [[0.0] * len(c) for c in candidates]

for g_start in range(0, N, GROUP_QUERIES):
    g_end = min(g_start + GROUP_QUERIES, N)
    pairs: list[tuple[str, str]] = []
    owners: list[tuple[int, int]] = []
    for qi in range(g_start, g_end):
        for ci, (_, doc_text) in enumerate(candidates[qi]):
            pairs.append((qtexts[qi], doc_text))
            owners.append((qi, ci))
    # One .predict() call ≈ 32 queries × 50 docs = 1600 pairs in batches of RERANK_BATCH
    scores = ranker.predict(pairs, batch_size=RERANK_BATCH, show_progress_bar=False)
    for (qi, ci), s in zip(owners, scores):
        rerank_scores[qi][ci] = float(s)
t_rerank = time.time() - t0
print(f"[3/3] reranked {N} × {TOP_N} pairs in {t_rerank:.1f}s ({N/t_rerank:.0f} q/s, batch={RERANK_BATCH})")

# === Stage 4: build runs (baseline + reranked), score with ranx ===
baseline_run: dict[str, dict[str, float]] = {}
rerank_run: dict[str, dict[str, float]] = {}
for qi, qid in enumerate(qids):
    cands = candidates[qi]
    # Baseline: dense top-K (no rerank); score by reverse rank
    baseline_run[qid] = {cands[r][0]: 1.0 / (r + 1) for r in range(min(TOP_K, len(cands)))}
    # Reranked: take rerank scores, take top-K
    ranked = sorted(zip(cands, rerank_scores[qi]), key=lambda x: -x[1])[:TOP_K]
    rerank_run[qid]   = {c[0]: s for c, s in ranked}

baseline_metrics = evaluate(qrels_obj, Run(baseline_run),
                             [f"recall@{TOP_K}", f"ndcg@{TOP_K}"])
rerank_metrics   = evaluate(qrels_obj, Run(rerank_run),
                             [f"recall@{TOP_K}", f"ndcg@{TOP_K}"])

result = {
    "library":              "throughput-stack (FlagEmbedding + qdrant-batch + FlagReranker + ranx)",
    "model":                "BAAI/bge-reranker-v2-m3",
    "baseline_dense_top5":  baseline_metrics,
    "rerank_top5":          rerank_metrics,
    "stage_seconds": {
        "encode":   round(t_encode,   1),
        "retrieve": round(t_retrieve, 1),
        "rerank":   round(t_rerank,   1),
        "total":    round(t_encode + t_retrieve + t_rerank, 1),
    },
    "n_queries":            N,
    "config": {
        "encode_batch":   ENCODE_BATCH,
        "qdrant_batch":   QDRANT_BATCH,
        "rerank_batch":   RERANK_BATCH,
        "group_queries":  GROUP_QUERIES,
        "fp16":           True,
    },
}
Path("results").mkdir(exist_ok=True)
Path("results/throughput_stack_rerank_metrics.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
print("\nCompare to results/rerankers_metrics.json (per-query API):")
print("  rerankers (per-query) : ~1964s end-to-end on 6980 queries")
print("  this version (batch)  : expect ~400-500s end-to-end (4× speedup)")
