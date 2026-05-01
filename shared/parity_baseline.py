"""parity_baseline.py — freeze ground-truth state before refactor.

Captures three signal classes so we can mechanically diff after each refactor step:

  1. Qdrant collection point counts — drift here means ingest broke.
  2. Sample vector signatures — drift here means encoder changed (different
     batch_size, different fp16 policy, different model load path).
  3. Result-file content hashes — drift here means downstream eval changed.

Mechanical parity only (Q1=a). Full eval rerun deferred to end.

Usage:
  ./.venv/bin/python shared/parity_baseline.py [--snapshot-name <name>]

Snapshots write to shared/parity/<name>.json (default: pre_refactor).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from qdrant_client import QdrantClient

QDRANT_URL = "http://127.0.0.1:6333"

# Fixed point IDs we sample. Picked to span the index range so encoder-batching
# bugs (which often manifest at batch boundaries) are likely to surface.
SAMPLE_POINT_IDS = [0, 1, 100, 1000, 5000]

COLLECTIONS = [
    "bge_m3_hnsw",          # W2 dense baseline
    "bge_m3_hybrid",        # W2 hybrid
    "fiqa_hybrid",          # W2 FiQA hybrid (the harder benchmark)
    "tech_corpus_hnsw",     # W2.5 dense
    "tech_corpus_hybrid",   # W2.5 hybrid (may not exist yet)
]

REPO_ROOT = Path(__file__).resolve().parent.parent

RESULT_FILES = [
    "lab-02-rerank-compress/results/fiqa_metrics.json",
    "lab-02-rerank-compress/results/hybrid_metrics.json",
    "lab-02-rerank-compress/results/rerank_metrics.json",
    "lab-02-rerank-compress/results/answer_eval.json",
    "lab-02-rerank-compress/results/compression_metrics.json",
    "lab-02-rerank-compress/results/chunk_sweep.json",
    "lab-02-5-graphrag/results/comparison.json",
]


def hash_file(path: Path) -> dict:
    """Short sha256 + size + line count. None alone is sufficient (file metadata
    changes don't always change content) but together spurious diffs are unlikely."""
    if not path.exists():
        return {"status": "missing", "sha256_16": None, "size": 0, "lines": 0}
    raw = path.read_bytes()
    return {
        "status": "ok",
        "sha256_16": hashlib.sha256(raw).hexdigest()[:16],
        "size": len(raw),
        "lines": raw.count(b"\n"),
    }


def vector_signature(client: QdrantClient, collection: str, point_id: int) -> dict:
    """Snapshot one point's vectors. Norm + first-3-dims for dense, nnz +
    sum-of-values for sparse. 6-decimal truncation — past that hits fp32 noise."""
    try:
        pts = client.retrieve(collection, ids=[point_id], with_vectors=True, with_payload=False)
    except Exception as e:
        return {"id": point_id, "status": f"error: {type(e).__name__}: {e}"}
    if not pts:
        return {"id": point_id, "status": "missing"}
    v = pts[0].vector
    if isinstance(v, dict):
        sig = {}
        for name, vec in v.items():
            if hasattr(vec, "indices"):
                sig[name] = {
                    "type": "sparse",
                    "nnz": len(vec.indices),
                    "sum_values": round(float(sum(vec.values)), 6),
                }
            else:
                norm = sum(x * x for x in vec) ** 0.5
                sig[name] = {
                    "type": "dense",
                    "norm": round(norm, 6),
                    "first3": [round(float(x), 6) for x in vec[:3]],
                }
        return {"id": point_id, "status": "ok", "vectors": sig}
    norm = sum(x * x for x in v) ** 0.5
    return {
        "id": point_id,
        "status": "ok",
        "type": "dense",
        "norm": round(norm, 6),
        "first3": [round(float(x), 6) for x in v[:3]],
    }


def snapshot() -> dict:
    qd = QdrantClient(url=QDRANT_URL, timeout=30)
    existing = {c.name for c in qd.get_collections().collections}

    out: dict = {"schema_version": 1, "collections": {}, "result_files": {}}

    for name in COLLECTIONS:
        if name not in existing:
            out["collections"][name] = {"status": "missing"}
            continue
        info = qd.get_collection(name)
        out["collections"][name] = {
            "status": "ok",
            "points_count": info.points_count,
            "vector_signatures": [
                vector_signature(qd, name, pid) for pid in SAMPLE_POINT_IDS
            ],
        }

    for rel in RESULT_FILES:
        out["result_files"][rel] = hash_file(REPO_ROOT / rel)

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot-name", default="pre_refactor")
    args = ap.parse_args()

    out_path = REPO_ROOT / f"shared/parity/{args.snapshot_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    snap = snapshot()
    out_path.write_text(json.dumps(snap, indent=2))

    n_ok = sum(1 for v in snap["collections"].values() if v.get("status") == "ok")
    n_files = sum(1 for v in snap["result_files"].values() if v.get("status") == "ok")
    print(f"snapshot saved: {out_path}")
    print(f"collections: {n_ok}/{len(snap['collections'])} ok")
    print(f"result files: {n_files}/{len(snap['result_files'])} present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
