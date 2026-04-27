"""Slice MS MARCO down to a manageable subset keyed by qrel-hit docs + filler."""
import ir_datasets, json, random
from pathlib import Path

OUT = Path("data")
OUT.mkdir(exist_ok=True)
random.seed(42)  # reproducible filler sampling across runs

ds = ir_datasets.load("msmarco-passage/dev/small")

# 1. Queries & qrels
queries = {q.query_id: q.text for q in ds.queries_iter()}
qrels = {}
for qrel in ds.qrels_iter():
    qrels.setdefault(qrel.query_id, []).append(qrel.doc_id)
print(f"loaded {len(queries)} queries, {sum(len(v) for v in qrels.values())} qrels")

# 2. Gold doc IDs (ensure recall is possible)
gold = {doc_id for docs in qrels.values() for doc_id in docs}
print(f"{len(gold)} gold docs")

# 3. Filler doc IDs — stream first N non-gold
filler_target = 10_000 - len(gold)  # pad to exactly 10K total docs
filler = []
for doc in ds.docs_iter():
    if doc.doc_id not in gold:
        filler.append(doc.doc_id)
        if len(filler) >= filler_target:
            break  # stop streaming early — 8M docs is expensive to scan fully
keep = gold | set(filler)
print(f"keeping {len(keep)} docs total")

# 4. Second pass: write only the kept docs
with (OUT / "docs.jsonl").open("w") as f:
    for doc in ds.docs_iter():
        if doc.doc_id in keep:
            f.write(json.dumps({"id": doc.doc_id, "text": doc.text}) + "\n")

# 5. Keep only queries that have at least one qrel in the kept set
keep_q = {}
keep_qrels = {}
for qid, gold_ids in qrels.items():
    hit = [g for g in gold_ids if g in keep]
    if hit:
        keep_q[qid] = queries[qid]
        keep_qrels[qid] = hit
print(f"retained {len(keep_q)} queries with qrels")

(OUT / "queries.json").write_text(json.dumps(keep_q, indent=2))
(OUT / "qrels.json").write_text(json.dumps(keep_qrels, indent=2))
print("done.")