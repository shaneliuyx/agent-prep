"""Throughput-stack ingest — FlagEmbedding combined forward pass + qdrant native parallel upload.

Sibling to 00_fiqa_eval_via_langchain.py's ingest block. Same FiQA corpus, same
collection schema (dense unnamed `""` + sparse `"langchain-sparse"`), so the
existing eval scripts (00_fiqa_eval_via_langchain.py, 00_fiqa_eval_via_tei.py,
00_fiqa_eval_throughput_stack.py) all work against the resulting collection.

Replaces two hand-rolled patterns at once:
  1. lab-02's `_upsert_with_retry` loop  → qd.upload_points(parallel=4)
  2. langchain's QdrantVectorStore.from_texts → direct PointStruct construction

Demonstrates: qdrant-client's batch upload helpers (upload_collection /
upload_points) ship with parallel workers + retry built-in. The hand-rolled
retry loop in lab-02 §1.4 is reinventing wheels.

Comparison targets:
  - lab-02b langchain ingest : ~30 min (per-query LangChain tax + 8K-pad trap)
  - lab-02 hand-rolled       : ~5 min  (sequential upsert + manual retry)
  - this script              : ~3-4 min (parallel=4 upload + combined encode)
"""
import os, sys, time
from pathlib import Path
from beir import util as beir_util
from beir.datasets.data_loader import GenericDataLoader
from FlagEmbedding import BGEM3FlagModel
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, SparseVectorParams, Distance,
    PointStruct, SparseVector,
)

# Suppress qdrant-client teardown warnings
def _hush_qdrant_unraisable(unraisable):
    tb = unraisable.exc_traceback
    if tb and "qdrant" in tb.tb_frame.f_code.co_filename:
        return
    sys.__unraisablehook__(unraisable)
sys.unraisablehook = _hush_qdrant_unraisable

HOME = os.path.expanduser("~")
COLLECTION = "lc_fiqa_hybrid_fast"   # parallel target so we don't clobber existing collections
ENCODE_BATCH = 128         # M5 Pro / 48 GB
UPLOAD_BATCH = 512         # points per upsert HTTP call
UPLOAD_PARALLEL = 4        # concurrent upload workers — qdrant-client built-in

# === Stage 0: load BEIR-FiQA ===
data_root = Path("data/beir")
data_root.mkdir(parents=True, exist_ok=True)
fiqa_dir = data_root / "fiqa"
if not fiqa_dir.exists():
    print("downloading BEIR-FiQA …")
    beir_util.download_and_unzip(
        "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip",
        str(data_root),
    )
corpus, _, _ = GenericDataLoader(data_folder=str(fiqa_dir)).load(split="test")

doc_ids = list(corpus.keys())
texts = [(corpus[d].get("title", "") + " " + corpus[d].get("text", "")).strip() or " "
         for d in doc_ids]
N = len(doc_ids)
print(f"FiQA corpus: {N} docs")

# === Stage 1: create collection with same schema langchain would create ===
qd = QdrantClient(url="http://127.0.0.1:6333", timeout=120)
qd.recreate_collection(
    collection_name=COLLECTION,
    vectors_config={"": VectorParams(size=1024, distance=Distance.COSINE)},
    sparse_vectors_config={"langchain-sparse": SparseVectorParams()},
)
print(f"created collection {COLLECTION}")

# === Stage 2: bulk encode (dense via FlagEmbedding MPS, sparse via fastembed SPLADE) ===
print(f"encoding {N} docs (dense via FlagEmbedding/MPS, sparse via fastembed/SPLADE) …")
t0 = time.time()
m = BGEM3FlagModel(f"{HOME}/models/bge-m3", use_fp16=False, device="mps")
out = m.encode(texts, batch_size=ENCODE_BATCH, max_length=512,
               return_dense=True, return_sparse=False)
dense_vecs = out["dense_vecs"]
t_dense = time.time() - t0
print(f"  dense  : {t_dense:.1f}s ({N/t_dense:.0f} docs/s)")

t0 = time.time()
sparse_model = SparseTextEmbedding("prithivida/Splade_PP_en_v1")
sparse_emb = list(sparse_model.embed(texts, batch_size=ENCODE_BATCH))
t_sparse_enc = time.time() - t0
print(f"  sparse : {t_sparse_enc:.1f}s ({N/t_sparse_enc:.0f} docs/s)")

# === Stage 3: build PointStruct list with named dense + sparse + payload ===
t0 = time.time()
points = [
    PointStruct(
        id=i,
        vector={
            "": dense_vecs[i].tolist(),
            "langchain-sparse": SparseVector(
                indices=sparse_emb[i].indices.tolist(),
                values=sparse_emb[i].values.tolist(),
            ),
        },
        # Match langchain's nesting: payload["metadata"]["doc_id"] so existing
        # eval scripts (which read this path) work without modification.
        payload={"metadata": {"doc_id": doc_ids[i]}, "page_content": texts[i]},
    )
    for i in range(N)
]
t_build = time.time() - t0
print(f"  build {N} PointStructs: {t_build:.1f}s")

# === Stage 4: parallel upload — replaces lab-02's _upsert_with_retry loop ===
print(f"uploading {N} points (parallel={UPLOAD_PARALLEL}, batch={UPLOAD_BATCH}) …")
t0 = time.time()
qd.upload_points(
    collection_name=COLLECTION,
    points=points,
    batch_size=UPLOAD_BATCH,
    parallel=UPLOAD_PARALLEL,   # qdrant-client native — built-in retry on transient failures
    max_retries=3,
)
t_upload = time.time() - t0

final_count = qd.get_collection(COLLECTION).points_count
print(f"  upload : {t_upload:.1f}s ({N/t_upload:.0f} docs/s)")
print(f"\n=== summary ===")
print(f"  encode dense  : {t_dense:.1f}s")
print(f"  encode sparse : {t_sparse_enc:.1f}s")
print(f"  build points  : {t_build:.1f}s")
print(f"  upload        : {t_upload:.1f}s")
print(f"  TOTAL         : {t_dense + t_sparse_enc + t_build + t_upload:.1f}s")
print(f"  collection    : {final_count}/{N} points")

if final_count != N:
    sys.exit(f"\n✗ Partial ingest: {final_count}/{N}. Re-run.")
print("\nNext: point eval scripts at this collection by setting COLLECTION = 'lc_fiqa_hybrid_fast'")
print("Compare wall_sec to:")
print("  langchain ingest (00_fiqa_eval_via_langchain.py): ~30 min")
print("  lab-02 hand-rolled (lab-02-rerank-compress/src/00_fiqa_eval.py): ~5 min")
