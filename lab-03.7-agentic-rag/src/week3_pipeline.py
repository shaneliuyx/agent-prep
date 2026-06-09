"""Week 3 SINGLE-PASS RAG baseline: retrieve -> rerank -> ONE synthesis call.

This is the non-agentic control the Phase-2 harness compares against the agentic
loop. It reuses the lab's tested retrieval/synthesis from baseline_handrolled.py
(BGE-M3 over Qdrant `bge_m3_hnsw` + BGE-reranker + one oMLX synthesis call) - no
grading, no rewrite, no corrective loop. Returns (answer, contexts).

Import:  from week3_pipeline import run_single_pass
"""
from __future__ import annotations

import os
import sys

# make sibling modules (baseline_handrolled) importable when run as `python src/...`
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baseline_handrolled import multi_retrieve, rerank, synthesize  # noqa: E402


def run_single_pass(question: str, top_k: int = 6) -> tuple[str, list[str]]:
    """One retrieval + one synthesis call (the single-pass baseline).

    Returns:
        (answer, contexts) - the synthesized answer string and the list of
        reranked passage texts used as context (for downstream RAGAS scoring).
    """
    hits = multi_retrieve(question)
    reranked = rerank(question, hits, top_k=top_k)
    out = synthesize(question, reranked)
    contexts = [h.get("text", "") for h in reranked]
    return out["answer"], contexts


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What is the main topic of the indexed documents?"
    ans, ctx = run_single_pass(q)
    print(f"Q: {q}\nA: {ans}\n({len(ctx)} contexts)")
