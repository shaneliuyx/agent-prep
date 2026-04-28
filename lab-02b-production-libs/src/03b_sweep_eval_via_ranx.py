"""1:1 with lab-02/src/03b_sweep_eval.py — but via the `ranx` library.

Same task: evaluate the 9 chunking-sweep variants (built by 03_chunk_sweep_via_langchain.py)
on parent-level recall@5. Different code: ranx replaces hand-rolled metric math with
typed Qrels + Run + evaluate() — and gives you bonus statistical-comparison via compare()
to test whether two variants differ significantly or just noise.
"""
import json, os
from pathlib import Path
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from ranx import Qrels, Run, evaluate, compare

HOME = os.path.expanduser("~")
SIZES = (256, 512, 1024)
OVERLAPS = (0, 64, 128)
LONG_DOC_PASSAGES = 8   # MUST match 03_chunk_sweep_via_langchain.py

K = 5

qd = QdrantClient(url="http://127.0.0.1:6333")
m  = SentenceTransformer(f"{HOME}/models/bge-m3", device="mps", trust_remote_code=True)

queries = json.loads(Path("data/queries.json").read_text())
qrels   = json.loads(Path("data/qrels.json").read_text())

# Build parent-level qrels — gold "parent" is the long_doc that contained the gold passage
raw = [json.loads(l) for l in open("data/docs.jsonl")]
p2long = {}
for i in range(0, len(raw), LONG_DOC_PASSAGES):
    for d in raw[i : i + LONG_DOC_PASSAGES]:
        p2long[d["id"]] = f"long_{i//LONG_DOC_PASSAGES}"

# ranx Qrels format: {qid: {doc_id: relevance}}
parent_qrels = {qid: {p2long[g]: 1 for g in gold_list if g in p2long}
                for qid, gold_list in qrels.items()}
qrels_obj = Qrels(parent_qrels)

# For each variant, build a Run (ranx style) and evaluate
all_runs = {}
for size in SIZES:
    for overlap in OVERLAPS:
        col = f"lc_sweep_s{size}_o{overlap}"
        run_dict = {}
        for qid, qtext in queries.items():
            qv = m.encode([qtext], normalize_embeddings=True)[0]
            top = qd.query_points(col, query=qv.tolist(), limit=K).points
            # Score parents by best chunk score
            parent_scores = {}
            for r in top:
                parent = r.payload.get("parent")
                if parent and (parent not in parent_scores or r.score > parent_scores[parent]):
                    parent_scores[parent] = float(r.score)
            run_dict[qid] = parent_scores
        run_name = f"s{size}_o{overlap}"
        all_runs[run_name] = Run(run_dict, name=run_name)

# Evaluate each variant — one call gets all metrics
print("--- Per-variant recall@5 / nDCG@5 ---")
grid = {}
for name, run in all_runs.items():
    metrics = evaluate(qrels_obj, run, ["recall@5", "ndcg@5"])
    grid[name] = metrics
    print(f"{name}  recall@5={metrics['recall@5']:.3f}  ndcg@5={metrics['ndcg@5']:.3f}")

Path("results").mkdir(exist_ok=True)
Path("results/ranx_sweep.json").write_text(json.dumps(grid, indent=2, default=float))

# --- Bonus: paired statistical comparison across all 9 variants ---
print("\n--- ranx.compare() across all 9 variants (Fisher's randomization, paired) ---")
report = compare(
    qrels=qrels_obj,
    runs=list(all_runs.values()),
    metrics=["recall@5", "ndcg@5"],
    max_p=0.05,
    stat_test="fisher",
)
print(report)

print("\nCompare to lab-02/results/chunk_sweep.json — recall@5 numbers should be in same range.")
print("Bonus: ranx.compare() tells you which variants differ SIGNIFICANTLY vs which are within noise.")
print("That's the question lab-02 can't answer with hand-rolled metrics alone.")
