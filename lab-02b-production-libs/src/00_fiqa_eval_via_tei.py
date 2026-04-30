"""TEI variant — BGE-M3 dense served by Text Embeddings Inference (HTTP).

Sibling to 00_fiqa_eval_via_langchain.py. Demonstrates the production
deployment split:
  - Offline ingest: done by 00_fiqa_eval_via_langchain.py (FlagEmbedding/MPS,
    ~5 min for 57k FiQA docs). This script reuses that collection.
  - Online query: TEI (HTTP) handles the 648 query embeddings at eval time.
    This is what a real RAG service does — bulk indexing happens once on
    whatever hardware is fast (GPU box, MPS Mac), and query-time embedding is
    served by a separate inference container the application talks to.

Why not ingest via TEI? Docker on macOS has no Metal/MPS access — TEI runs
CPU-only and ingests at ~3 docs/sec, vs ~200 docs/sec for FlagEmbedding on MPS.
Bulk indexing is the wrong workload for a CPU-bound TEI on Apple Silicon.
Query-time inference on 648 strings is fine — TEI finishes in <30s.

PREREQ: run 00_fiqa_eval_via_langchain.py first to populate `lc_fiqa_hybrid`.
This script will fail fast if the collection is missing or under-populated.

═══════════════════════════════════════════════════════════════════════════════
SETUP — start TEI in Docker before running this script
═══════════════════════════════════════════════════════════════════════════════

# 1. Pull the arm64 native image. Pinned arm64 tags like :cpu-arm64-1.7 do NOT exist —
#    :cpu-arm64-latest is the only published arm64-native variant.
docker pull ghcr.io/huggingface/text-embeddings-inference:cpu-arm64-latest

# 2. Start TEI with BGE-M3. First launch downloads weights (~2.3 GB) into the
#    mounted hf cache so subsequent restarts are instant. --max-input-length is
#    NOT a valid TEI flag; per-request truncation is via "truncate": true in the body
#    (already auto_truncate=true server-wide by default).
docker run -d --name tei-bge-m3 --platform linux/arm64 -p 8080:80 \
    -v "$HOME/.cache/huggingface/hub:/data" \
    ghcr.io/huggingface/text-embeddings-inference:cpu-arm64-latest \
    --model-id BAAI/bge-m3 \
    --max-batch-tokens 16384 \
    --max-client-batch-size 64

# 3. Wait until ready (cached weights: ~10s; first download: ~30-90s)
until curl -sf http://localhost:8080/health > /dev/null; do sleep 2; done && echo READY

# 4. Verify the right encoder is serving. /info.model_id MUST be "BAAI/bge-m3" —
#    a stale container with a name collision can silently serve a different model.
curl -sf http://localhost:8080/info | python3 -m json.tool

# 5. Smoke test
curl -X POST http://localhost:8080/embed \
    -H 'Content-Type: application/json' \
    -d '{"inputs": ["hello world"], "truncate": true}' | head -c 80

# Cleanup when done:
#   docker stop tei-bge-m3 && docker rm tei-bge-m3
═══════════════════════════════════════════════════════════════════════════════
"""
import json, os, sys, time
from pathlib import Path
import requests
from beir import util as beir_util
from beir.datasets.data_loader import GenericDataLoader
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient
from ranx import Qrels, Run, evaluate
from tqdm import tqdm

# Suppress qdrant-client's __del__ teardown noise — lib calls warnings.warn() after
# stdlib torn down, raising `'NoneType' object is not callable`. Not our bug.
def _hush_qdrant_unraisable(unraisable):
    tb = unraisable.exc_traceback
    if tb and "qdrant" in tb.tb_frame.f_code.co_filename:
        return
    sys.__unraisablehook__(unraisable)
sys.unraisablehook = _hush_qdrant_unraisable

HOME = os.path.expanduser("~")
COLLECTION = "lc_fiqa_hybrid"       # reuse langchain script's collection (offline-ingested via MPS)
EXPECTED_POINTS = 57638             # FiQA corpus size; partial-ingest detector
K = 10
TEI_URL = "http://localhost:8080"
TEI_BATCH = 32  # texts per HTTP call; TEI internally batches further by token count


class TEIEmbeddings(Embeddings):
    """LangChain Embeddings wrapper around a Text Embeddings Inference server.

    TEI exposes BGE-M3 dense via /embed. Server-side it does dynamic batching by
    token count (--max-batch-tokens), so we send moderate client-side batches
    and let TEI fuse them efficiently. Sparse is still done locally via fastembed
    because TEI doesn't currently expose BGE-M3's sparse head.
    """

    def __init__(self, base_url: str, batch_size: int = 32, timeout: int = 60):
        self._url = base_url.rstrip("/") + "/embed"
        self._batch_size = batch_size
        self._timeout = timeout

    def _post(self, inputs: list[str]) -> list[list[float]]:
        # TEI rejects empty/whitespace-only strings with `inputs cannot be empty`.
        # FiQA has docs where title+text strips to "". Replace with " " so the request
        # validates and produces a near-zero vector that won't match real queries.
        safe = [t if t.strip() else " " for t in inputs]
        r = requests.post(self._url, json={"inputs": safe, "truncate": True},
                          timeout=self._timeout)
        if not r.ok:
            raise requests.HTTPError(
                f"{r.status_code} {r.reason} from TEI: {r.text[:300]}", response=r
            )
        return r.json()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            out.extend(self._post(texts[i : i + self._batch_size]))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._post([text])[0]


# Fail fast if TEI isn't running — most common failure mode is forgetting docker run
try:
    health = requests.get(f"{TEI_URL}/health", timeout=2)
    health.raise_for_status()
    print(f"TEI ready at {TEI_URL}")
except Exception as e:
    raise SystemExit(
        f"\n✗ TEI not reachable at {TEI_URL}. Start it first:\n\n"
        f"  docker run -d --name tei-bge-m3 --platform linux/arm64 -p 8080:80 \\\n"
        f'    -v "$HOME/.cache/huggingface/hub:/data" \\\n'
        f"    ghcr.io/huggingface/text-embeddings-inference:cpu-arm64-latest \\\n"
        f"    --model-id BAAI/bge-m3 --max-batch-tokens 16384 --max-client-batch-size 64\n\n"
        f"  until curl -sf {TEI_URL}/health > /dev/null; do sleep 2; done && echo READY\n\n"
        f"Original error: {e}\n"
    )

dense = TEIEmbeddings(TEI_URL, batch_size=TEI_BATCH)
sparse = FastEmbedSparse(model_name="prithivida/Splade_PP_en_v1")
qd = QdrantClient(url="http://127.0.0.1:6333")

# Load BEIR-FiQA via the official BEIR loader. Returns ranx-shaped dicts:
#   corpus : {doc_id: {"title": str, "text": str}}
#   queries: {qid: query_text}
#   qrels  : {qid: {doc_id: relevance}}  ← already a Qrels-compatible dict
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

# Verify the offline-ingested collection exists and is fully populated.
# Partial collections silently return wrong recall numbers (see §7.5b operational rules).
existing = [c.name for c in qd.get_collections().collections]
if COLLECTION not in existing:
    raise SystemExit(
        f"\n✗ Collection '{COLLECTION}' not found in Qdrant.\n"
        f"  Run 00_fiqa_eval_via_langchain.py first to populate it via FlagEmbedding/MPS.\n"
    )
points = qd.get_collection(COLLECTION).points_count
if points < EXPECTED_POINTS:
    raise SystemExit(
        f"\n✗ Collection '{COLLECTION}' is partial: {points}/{EXPECTED_POINTS} points "
        f"({100*points/EXPECTED_POINTS:.1f}%).\n"
        f"  Re-run 00_fiqa_eval_via_langchain.py with the collection deleted first:\n"
        f"    curl -X DELETE http://127.0.0.1:6333/collections/{COLLECTION}\n"
    )
print(f"reusing {COLLECTION} ({points} points) — TEI will only embed queries")


def build_store(mode):
    return QdrantVectorStore.from_existing_collection(
        embedding=dense, sparse_embedding=sparse,
        collection_name=COLLECTION, url="http://127.0.0.1:6333",
        retrieval_mode=mode,
    )

stores = {
    "dense":  build_store(RetrievalMode.DENSE),
    "sparse": build_store(RetrievalMode.SPARSE),
    "hybrid": build_store(RetrievalMode.HYBRID),
}

# Stable ordering: qids sorted, query_texts aligned to that order so query_vecs[i] ↔ qids[i]
qids = sorted(queries.keys())
query_texts = [queries[qid] for qid in qids]
print("Pre-embedding queries via TEI …")
query_vecs = dense.embed_documents(query_texts)


def metrics(mode: str) -> dict:
    """Run retrieval for `mode`, compute metrics two ways for side-by-side comparison.

    Same retrieval results, two compute paths:
      - ranx: IR-canonical recall = |relevant ∩ retrieved| / |relevant|
      - hand-rolled: lab-02-style "recall" = 1 if any relevant in top-K else 0
        (this is actually HIT-RATE@K, mislabeled as recall in lab-02)

    Identical mrr/ndcg between the two confirms vector ordering is identical;
    differing recall numbers expose the formula gap.
    """
    import math
    store = stores[mode]
    run: dict[str, dict[str, float]] = {}
    t0 = time.time()
    # Hand-rolled accumulators (lab-02-style)
    n = hit_count = 0
    rr_sum = ndcg_sum = 0.0
    for i, qid in enumerate(tqdm(qids, desc=mode, leave=False)):
        if mode == "dense":
            docs = store.similarity_search_by_vector(query_vecs[i], k=K)
        else:
            docs = store.similarity_search(query_texts[i], k=K)
        ids = [d.metadata["doc_id"] for d in docs]
        run[qid] = {doc_id: 1.0 / (rank + 1) for rank, doc_id in enumerate(ids)}
        # Lab-02 hand-rolled compute (hit-rate, not true recall)
        gold = set(qrels[qid].keys())
        n += 1
        if any(d in gold for d in ids):
            hit_count += 1
        for rank, d in enumerate(ids, 1):
            if d in gold:
                rr_sum += 1.0 / rank
                break
        dcg   = sum(1.0 / math.log2(rank + 1) for rank, d in enumerate(ids, 1) if d in gold)
        ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold), K) + 1))
        ndcg_sum += (dcg / ideal) if ideal else 0.0
    ranx_scores = evaluate(qrels_obj, Run(run), [f"recall@{K}", f"mrr@{K}", f"ndcg@{K}"])
    return {
        "library": "tei + langchain-qdrant",
        "benchmark": "BEIR-FiQA-2018",
        "mode": mode,
        "ranx": {
            f"recall@{K}": ranx_scores[f"recall@{K}"],
            f"mrr@{K}":    ranx_scores[f"mrr@{K}"],
            f"ndcg@{K}":   ranx_scores[f"ndcg@{K}"],
        },
        "hand_rolled": {
            f"hit_rate@{K}": hit_count / n,   # lab-02 mislabeled this as recall@K
            f"mrr@{K}":      rr_sum / n,
            f"ndcg@{K}":     ndcg_sum / n,
        },
        "wall_sec": time.time() - t0,
        "n_queries": len(qids),
    }


results = [metrics("dense"), metrics("sparse"), metrics("hybrid")]
Path("results").mkdir(exist_ok=True)
Path("results/tei_fiqa_metrics.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
print("\nCompare to results/langchain_fiqa_metrics.json — dense should match (same encoder).")
print("If TEI dense diverges by >1pp, inspect /info endpoint to verify pooling config.")
