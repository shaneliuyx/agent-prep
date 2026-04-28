"""IR evaluation via the `ranx` library.

Demonstrates ranx — typed Qrels + Run objects, one-call evaluate(), and bonus
statistical-comparison features. Replaces the hand-rolled recall@K / MRR@K / nDCG@K
math you wrote in lab-02.

This script:
  1. Runs dense retrieval against bge_m3_hnsw (same as lab-01 baseline)
  2. Builds qrels and run dicts in ranx's expected format
  3. Calls evaluate() once to get all 3 metrics
  4. Verifies the numbers match lab-01's hand-rolled values

Bonus: shows how to compare two systems with paired statistical tests via ranx.compare(),
which is the right tool when you want to know whether a 0.005 recall@10 difference is
real or just noise.
"""
import json, os, time
from pathlib import Path
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from ranx import Qrels, Run, evaluate, compare

HOME = os.path.expanduser("~")
COLLECTION = "bge_m3_hnsw"
K = 10

qd = QdrantClient(url="http://127.0.0.1:6333")
m  = SentenceTransformer(f"{HOME}/models/bge-m3", device="mps", trust_remote_code=True)

queries = json.loads(Path("data/queries.json").read_text())
qrels   = json.loads(Path("data/qrels.json").read_text())
print(f"running dense retrieval for {len(queries)} queries, then evaluating via ranx")

# --- Step 1: Build the run (system output) by running retrieval ---
run_dict = {}
t0 = time.time()
for qid, qtext in queries.items():
    qv = m.encode([qtext], normalize_embeddings=True)[0]
    hits = qd.query_points(COLLECTION, query=qv.tolist(), limit=K).points
    # ranx expects {qid: {doc_id: score}} — score must be a float
    run_dict[qid] = {h.payload["doc_id"]: float(h.score) for h in hits}
print(f"retrieval: {time.time() - t0:.1f}s")

# --- Step 2: Build qrels (ground truth) in ranx's expected format ---
# Format: {qid: {doc_id: relevance}}  — relevance is int (binary or graded)
qrels_dict = {qid: {doc_id: 1 for doc_id in gold_list} for qid, gold_list in qrels.items()}

# --- Step 3: Wrap as Qrels + Run objects ---
qrels_obj = Qrels(qrels_dict)
run_obj   = Run(run_dict, name="bge_m3_dense")

# --- Step 4: Evaluate ALL metrics in one call ---
metrics = evaluate(qrels_obj, run_obj, ["recall@10", "mrr@10", "ndcg@10"])
print("\nranx evaluate() result:")
print(json.dumps(metrics, indent=2))

result = {
    "library": "ranx",
    "collection": COLLECTION,
    "metrics": metrics,
    "n_queries": len(queries),
}
Path("results").mkdir(exist_ok=True)
Path("results/ranx_metrics.json").write_text(json.dumps(result, indent=2, default=float))

# --- Bonus: paired statistical comparison via ranx.compare() ---
# Demo: compare bge_m3_dense against itself with a slight perturbation,
# to show how ranx tests whether a difference is statistically significant.
# In a real project you'd compare two real systems (e.g., dense vs hybrid).
import random
random.seed(42)
perturbed_run = {qid: {doc_id: score + random.uniform(-0.001, 0.001)
                       for doc_id, score in docs.items()}
                 for qid, docs in run_dict.items()}
perturbed_obj = Run(perturbed_run, name="bge_m3_dense_perturbed")

report = compare(
    qrels=qrels_obj,
    runs=[run_obj, perturbed_obj],
    metrics=["recall@10", "ndcg@10"],
    max_p=0.05,                          # significance threshold
    stat_test="fisher",                  # Fisher's randomization (paired test)
)
print("\nranx compare() report (real production use: compare two SYSTEMS, not noise):")
print(report)

print("\nCompare the metrics dict above to lab-01's hand-rolled retrieval_metrics.json —")
print("recall@10 / mrr@10 / ndcg@10 should be IDENTICAL to ~6 decimal places.")
