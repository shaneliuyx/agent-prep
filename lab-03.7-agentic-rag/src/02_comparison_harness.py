"""Run the single-pass baseline AND BOTH agentic graphs on the same dev set:
  - single_pass        : Week-3 retrieve -> rerank -> synthesize (always retrieves)
  - agentic_canonical  : LangChain's 5-node graph (agent MAY skip retrieval)  [canonical_agentic_rag.py]
  - agentic_structural : the §2.5.1 fix (retrieval is a structural edge)        [structural_rag.py]

Capture per-query: answer, latency, total LLM calls, retrieved contexts. Quality
(faithfulness / context_*) is scored downstream with RAGAS (02b) over these outputs.

Run from the lab root with the lab's uv venv:
    uv run python src/02_comparison_harness.py
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# sibling modules live in this src/ dir
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from week3_pipeline import run_single_pass            # single-pass baseline
from canonical_agentic_rag import app as canonical_app  # agent-discretion (skip-allowed)
from structural_rag import app as structural_app        # structural retrieval (the fix)

# Dev set is a CLI parameter (falls back to $DEV_SET, then the Week-3 default). Rows carry
# {source_doc_id, source_text, question, short_answer}. --out lets a harder/larger dev set
# write its own artifact instead of clobbering results/comparison_raw.json (so 02b/§2.6 can
# point at it). Example:
#   uv run python src/02_comparison_harness.py \
#       --dev-set data/hard_dev_set.jsonl --out results/comparison_hard.json
_p = argparse.ArgumentParser(description="single-pass + both agentic arms over a dev set")
_p.add_argument("--dev-set", default=os.getenv("DEV_SET", os.path.expanduser(
    "~/code/agent-prep/lab-03-rag-eval/data/dev_set.jsonl")),
    help="path to a .jsonl dev set {source_doc_id, source_text, question, short_answer}")
_p.add_argument("--out", default="results/comparison_raw.json",
                help="where to write the per-query raw results")
_args = _p.parse_args()
DEV_SET = os.path.expanduser(_args.dev_set)
dev = [json.loads(line) for line in open(DEV_SET) if line.strip()]
print(f"dev set: {DEV_SET} ({len(dev)} rows) -> {_args.out}")


def run_agentic(app, question: str) -> dict:
    """Invoke a LangGraph agentic graph, return {answer, latency, llm_calls, contexts}.
    Contexts are the retrieved passages (ToolMessages) so RAGAS can score context_* on this arm.
    recursion_limit guards the rewrite -> retrieve loop on pathological queries."""
    t0 = time.time()
    result = app.invoke({"messages": [("user", question)]}, {"recursion_limit": 25})
    latency = time.time() - t0
    msgs = result["messages"]
    contexts = [p for m in msgs if getattr(m, "type", None) == "tool"
                for p in m.content.split("\n\n") if p.strip()]
    return {
        "answer": msgs[-1].content,
        "latency": latency,
        "llm_calls": sum(1 for m in msgs if getattr(m, "type", None) == "ai"),
        "contexts": contexts,
    }


results = []
for i, q in enumerate(dev):
    qid = q.get("qid") or q.get("source_doc_id") or f"q{i}"
    question = q["question"]

    t0 = time.time()
    sp_answer, sp_contexts = run_single_pass(question)
    sp = {"answer": sp_answer, "latency": time.time() - t0,
          "llm_calls": 1, "contexts": sp_contexts}

    canonical = run_agentic(canonical_app, question)
    structural = run_agentic(structural_app, question)

    results.append({
        "qid": qid, "question": question, "ground_truth": q.get("short_answer", ""),
        "single_pass": sp,
        "agentic_canonical": canonical,
        "agentic_structural": structural,
    })
    print(f"[{i + 1}/{len(dev)}] {qid}: single-pass {sp['latency']:.1f}s / "
          f"canonical {canonical['latency']:.1f}s ({len(canonical['contexts'])} ctx) / "
          f"structural {structural['latency']:.1f}s ({len(structural['contexts'])} ctx)")

out = Path(os.path.expanduser(_args.out))
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(results, indent=2))
print(f"wrote {out} ({len(results)} rows)")
