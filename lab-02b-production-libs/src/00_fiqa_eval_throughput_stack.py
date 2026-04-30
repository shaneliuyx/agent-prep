"""Throughput-stack variant — bypass LangChain for offline-eval workload.

Same task, same collection, same numbers as 00_fiqa_eval_via_langchain.py — but
uses the throughput-shaped libs end-to-end:

  Encode  : FlagEmbedding BGEM3FlagModel.encode(list, batch_size=128)   ← combined dense+sparse forward pass
  Search  : qdrant_client.query_batch_points(64 queries / HTTP call)    ← bulk vector search
  Metrics : ranx evaluate(qrels, run, [...])                            ← Numba JIT

Demonstrates: when the workload is offline-eval (throughput, not latency),
batch-shaped libs win over per-query serving wrappers. Same vectors → same
recall numbers, different wall-time.

Reuses the `lc_fiqa_hybrid` collection ingested by 00_fiqa_eval_via_langchain.py.
Sparse encoder must match what populated the collection — fastembed SPLADE++,
NOT BGE-M3's sparse head — or recall will collapse.
"""
import json, os, sys, time
from pathlib import Path
from beir import util as beir_util
from beir.datasets.data_loader import GenericDataLoader
from FlagEmbedding import BGEM3FlagModel
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    QueryRequest, Prefetch, FusionQuery, Fusion, SparseVector,
)
from ranx import Qrels, Run, evaluate

# Suppress qdrant-client's __del__ teardown warnings — lib bug, not ours
def _hush_qdrant_unraisable(unraisable):
    tb = unraisable.exc_traceback
    if tb and "qdrant" in tb.tb_frame.f_code.co_filename:
        return
    sys.__unraisablehook__(unraisable)
sys.unraisablehook = _hush_qdrant_unraisable

HOME = os.path.expanduser("~")
COLLECTION = "lc_fiqa_hybrid"
EXPECTED_POINTS = 57638
K = 10
QDRANT_BATCH = 64       # queries per HTTP roundtrip
ENCODE_BATCH = 128      # M5 Pro / 48 GB — matches lab-02's INGEST_BATCH

qd = QdrantClient(url="http://127.0.0.1:6333", timeout=60)

# Verify collection exists and is fully populated
existing = [c.name for c in qd.get_collections().collections]
if COLLECTION not in existing:
    raise SystemExit(f"\n✗ Collection '{COLLECTION}' missing. "
                     f"Run 00_fiqa_eval_via_langchain.py first.\n")
points = qd.get_collection(COLLECTION).points_count
if points < EXPECTED_POINTS:
    raise SystemExit(f"\n✗ Collection partial: {points}/{EXPECTED_POINTS}. Re-ingest.\n")

# Load BEIR-FiQA via BEIR loader — emits ranx-shaped dicts
data_root = Path("data/beir")
data_root.mkdir(parents=True, exist_ok=True)
fiqa_dir = data_root / "fiqa"
if not fiqa_dir.exists():
    print("downloading BEIR-FiQA …")
    beir_util.download_and_unzip(
        "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip",
        str(data_root),
    )
corpus, queries, qrels = GenericDataLoader(data_folder=str(fiqa_dir)).load(split="test")
qrels_obj = Qrels(qrels)
print(f"FiQA: {len(corpus)} docs · {len(queries)} test queries · {sum(len(g) for g in qrels.values())} qrels")
print(f"reusing {COLLECTION} ({points} points)")

qids = sorted(queries.keys())
query_texts = [queries[qid] for qid in qids]
N = len(qids)

# === Stage 1: batch-encode ALL queries (dense via FlagEmbedding direct, sparse via fastembed SPLADE) ===
print(f"encoding {N} queries (dense via FlagEmbedding MPS, sparse via fastembed SPLADE) …")
t0 = time.time()
m = BGEM3FlagModel(f"{HOME}/models/bge-m3", use_fp16=False, device="mps")
out = m.encode(query_texts, batch_size=ENCODE_BATCH, max_length=512,
               return_dense=True, return_sparse=False)
dense_vecs = [v.tolist() for v in out["dense_vecs"]]
t_dense = time.time() - t0

t0 = time.time()
sparse_model = SparseTextEmbedding("prithivida/Splade_PP_en_v1")
sparse_emb = list(sparse_model.embed(query_texts, batch_size=ENCODE_BATCH))
sparse_vecs = [
    SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
    for e in sparse_emb
]
t_sparse_enc = time.time() - t0
print(f"  dense encode  : {t_dense:.1f}s ({N/t_dense:.0f} q/s)")
print(f"  sparse encode : {t_sparse_enc:.1f}s ({N/t_sparse_enc:.0f} q/s)")


def _bulk_search(reqs: list[QueryRequest]) -> list[list]:
    """Submit all reqs in QDRANT_BATCH-sized HTTP calls; flatten responses."""
    out = []
    for s in range(0, len(reqs), QDRANT_BATCH):
        batch = qd.query_batch_points(COLLECTION, requests=reqs[s : s + QDRANT_BATCH])
        out.extend([resp.points for resp in batch])
    return out


def metrics(mode: str) -> dict:
    t0 = time.time()
    if mode == "dense":
        reqs = [QueryRequest(query=dense_vecs[i], limit=K, with_payload=True) for i in range(N)]
    elif mode == "sparse":
        reqs = [QueryRequest(query=sparse_vecs[i], using="langchain-sparse",
                             limit=K, with_payload=True) for i in range(N)]
    elif mode == "hybrid":
        reqs = [QueryRequest(
            prefetch=[
                Prefetch(query=dense_vecs[i], limit=50),
                Prefetch(query=sparse_vecs[i], using="langchain-sparse", limit=50),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=K,
            with_payload=True,
        ) for i in range(N)]
    else:
        raise ValueError(mode)
    hits = _bulk_search(reqs)
    # langchain QdrantVectorStore nests user metadata under payload["metadata"]
    run: dict[str, dict[str, float]] = {
        qid: {h.payload["metadata"]["doc_id"]: 1.0 / (rank + 1) for rank, h in enumerate(hits[i])}
        for i, qid in enumerate(qids)
    }
    scores = evaluate(qrels_obj, Run(run), [f"recall@{K}", f"mrr@{K}", f"ndcg@{K}"])
    return {"library": "throughput-stack (FlagEmbedding + qdrant-batch + ranx)",
            "benchmark": "BEIR-FiQA-2018", "mode": mode,
            **scores, "wall_sec": time.time() - t0, "n_queries": N}


print("running eval (dense / sparse / hybrid) …")
results = [metrics("dense"), metrics("sparse"), metrics("hybrid")]
for r in results:
    r["enc_dense_sec"] = round(t_dense, 2)
    r["enc_sparse_sec"] = round(t_sparse_enc, 2)
Path("results").mkdir(exist_ok=True)
Path("results/throughput_stack_fiqa_metrics.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
print("\nCompare wall_sec to results/langchain_fiqa_metrics.json:")
print("  langchain (per-query): dense ~3.4s, sparse ~5.5s, hybrid ~24s")
print("  this version (batch) : expect dense <1s, sparse <1s, hybrid <5s")
