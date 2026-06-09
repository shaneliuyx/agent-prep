"""Build a difficulty-STRATIFIED dev set over the EXISTING corpus, fairly.

The lab's dev sets are single-hop and easy: BGE-M3 puts the gold passage at rank 1, so
single-pass wins every bucket and §2.6 can't show where the corrective loop helps. This
generates a harder set WITHOUT gaming the result.

Fairness rule: difficulty is defined by the RETRIEVAL RANK of the known-gold passage, NOT by
which pipeline answers correctly. A question is:
  - easy   : gold passage at rank 1 in dense top-K
  - medium : gold at rank 2..5
  - hard   : gold at rank >5, or not in top-K at all
"Hard" = first-pass dense retrieval does not surface the gold at the top - exactly the regime
the grade->rewrite->retrieve loop (and the reranker) are meant to rescue. We then MEASURE
whether they actually do (they might not - that's an honest finding, not a rigged one).

The gold passage is located in the corpus by TEXT match (the corpus doc_id is a sequential
index, not the dev set's MS-MARCO source_doc_id). Rows whose gold text isn't in the corpus are
skipped and reported - never silently dropped.

Output keeps the harness schema {source_doc_id, source_text, question, short_answer} and adds
{gold_rank, difficulty} so §2.6 can stratify. Default writes the medium+hard rows.

    uv run python src/make_hard_dev_set.py \
        --source ~/code/agent-prep/lab-03-rag-eval/data/dev_candidates.jsonl \
        --out data/hard_dev_set.jsonl --pool 30 --include medium,hard
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/code/agent-prep/lab-03.7-agentic-rag/.env"))
sys.path.insert(0, os.path.expanduser("~/code/agent-prep/shared"))

from qdrant_client import QdrantClient  # noqa: E402
from rag_hybrid import BGE_M3, DenseEncoder, autoconfig  # noqa: E402

QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "bge_m3_hnsw")

_norm = lambda s: re.sub(r"\s+", " ", (s or "").strip().lower())


def difficulty_of(rank: int | None) -> str:
    if rank == 1:
        return "easy"          # gold already top-1: nothing to fix
    if rank is not None and 2 <= rank <= 5:
        return "medium"        # gold near top: rerank likely surfaces it
    if rank is not None:
        return "hard"          # gold at rank 6..pool: in the rerank pool but buried
    return "unreachable"       # gold NOT in the dense pool: only a query rewrite can retrieve it


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=os.path.expanduser(
        "~/code/agent-prep/lab-03-rag-eval/data/dev_candidates.jsonl"))
    ap.add_argument("--out", default="data/hard_dev_set.jsonl")
    ap.add_argument("--pool", type=int, default=30, help="dense top-K used to rank the gold")
    ap.add_argument("--include", default="medium,hard,unreachable",
                    help="comma list of buckets to write (easy,medium,hard,unreachable)")
    args = ap.parse_args()
    keep = {b.strip() for b in args.include.split(",")}

    qdrant = QdrantClient(url=QDRANT_URL, timeout=60)
    encoder = DenseEncoder(autoconfig.encoder_config_for(BGE_M3))

    # corpus text -> doc_id (gold passage lives here under a sequential id, matched by text)
    text2id: dict[str, str] = {}
    off = None
    while True:
        pts, off = qdrant.scroll(QDRANT_COLLECTION, limit=2000, with_payload=True, offset=off)
        for p in pts:
            text2id[_norm(p.payload.get("text"))] = str(p.payload.get("doc_id"))
        if off is None:
            break
    print(f"corpus: {len(text2id)} passages")

    rows = [json.loads(l) for l in open(os.path.expanduser(args.source)) if l.strip()]
    out_rows, missing = [], 0
    dist = {"easy": 0, "medium": 0, "hard": 0, "unreachable": 0}
    ranks: list[int | None] = []
    for r in rows:
        gold_id = text2id.get(_norm(r["source_text"]))
        if gold_id is None:
            missing += 1
            continue  # gold not in corpus -> can't rank fairly, skip (reported below)
        qv = encoder.encode([r["question"]])[0]
        hits = qdrant.query_points(QDRANT_COLLECTION, query=qv.tolist(), limit=args.pool,
                                   with_payload=True).points
        ranked = [str(h.payload.get("doc_id")) for h in hits]
        rank = ranked.index(gold_id) + 1 if gold_id in ranked else None
        ranks.append(rank)
        diff = difficulty_of(rank)
        dist[diff] += 1
        if diff in keep:
            out_rows.append({**{k: r[k] for k in
                                ("source_doc_id", "source_text", "question", "short_answer")},
                             "gold_rank": rank, "difficulty": diff})

    out = Path(os.path.expanduser(args.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in out_rows) + "\n")
    found = [r for r in ranks if r is not None]
    print(f"difficulty distribution (all {len(rows)} rows, pool={args.pool}): {dist}")
    if found:
        print(f"gold-rank when found: min={min(found)} median={sorted(found)[len(found)//2]} "
              f"max={max(found)} | in top-1: {sum(r == 1 for r in found)}")
    print(f"skipped (gold text not in corpus): {missing}")
    print(f"wrote {out}: {len(out_rows)} rows (buckets kept: {sorted(keep)})")


if __name__ == "__main__":
    main()
