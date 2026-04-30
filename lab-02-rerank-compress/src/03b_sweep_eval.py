"""Recall@5 for each of the 9 chunk_size × overlap variants.

Iterates SWEEP_VARIANTS from src/model_config.py — the same list 03_chunk_sweep.py used
to build the collections, so name + chunk_size + overlap stay in sync automatically.
"""
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from model_config import SWEEP_VARIANTS, LONG_DOC_PASSAGES

M = SWEEP_VARIANTS[0].model
N = LONG_DOC_PASSAGES   # passages per synthetic long doc — MUST match 03_chunk_sweep.py

qd = QdrantClient(url="http://127.0.0.1:6333")
m  = SentenceTransformer(M.path, device="mps", trust_remote_code=M.trust_remote_code)

queries = json.loads(Path("data/queries.json").read_text())
qrels   = json.loads(Path("data/qrels.json").read_text())

# A gold "parent" is the long_doc that originally contained the passage qrel
# Quick map: passage_id -> long_id (uses LONG_DOC_PASSAGES from spec to stay in sync with build)
raw = [json.loads(l) for l in open("data/docs.jsonl")]
p2long = {}
for i in range(0, len(raw), N):
    for d in raw[i : i + N]:
        p2long[d["id"]] = f"long_{i//N}"

grid = {}
for spec in SWEEP_VARIANTS:
    hit = 0
    for qid, qtext in queries.items():
        qv = m.encode([M.query_prefix + qtext], normalize_embeddings=True)[0]
        top = qd.query_points(spec.name, query=qv.tolist(), limit=5).points
        parents = {r.payload["parent"] for r in top}
        gold_parents = {p2long[g] for g in qrels[qid] if g in p2long}
        if parents & gold_parents:
            hit += 1
    recall = hit / len(queries)
    grid[(spec.chunk_size, spec.overlap)] = recall
    print(f"s={spec.chunk_size} o={spec.overlap}  recall@5 (parent-level) = {recall:.3f}")

Path("results/chunk_sweep.json").write_text(json.dumps(
    {f"{s}_{o}": v for (s, o), v in grid.items()}, indent=2
))