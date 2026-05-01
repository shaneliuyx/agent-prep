"""Compare GraphRAG vs vector-RAG on multi-hop queries.

Runs each question through:
1. GraphRAG (this lab's `query_graph.answer`)
2. (Optionally) a vector-RAG implementation if a `vector_search(query, k)` callable
   is provided via the VECTOR_SEARCH_MODULE env var (e.g. `mylab.retrieve` and
   inside that module a top-level `search_with_rerank(query, k=5)` returning
   `{"answer": str}`).

If no vector module is configured, GraphRAG-only metrics are reported.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import time
from pathlib import Path

# query_graph lives next to this file in src/. When the script is run via
# `python src/compare.py` from the project root, `src/` is the script's
# directory, so a sibling import ("from query_graph") works without
# additional sys.path manipulation. The previous `from src.query_graph`
# style assumed a package layout this lab does not adopt.
from query_graph import answer as graph_answer


def _load_vector_search():
    """Resolve a vector_search callable from VECTOR_SEARCH_MODULE if set.

    Format: <module_path_relative_or_absolute>:<callable_name>
    Example: VECTOR_SEARCH_MODULE=../lab-02-rerank-compress/src:search_with_rerank

    Returns None if not configured or if the import fails — compare.py then
    runs in GraphRAG-only mode."""
    spec = os.getenv("VECTOR_SEARCH_MODULE")
    if not spec:
        return None
    try:
        mod_path, _, fn_name = spec.partition(":")
        if not fn_name:
            fn_name = "search_with_rerank"
        # Allow specifying a directory to add to sys.path before import.
        if "/" in mod_path or "\\" in mod_path:
            dir_path = str(Path(mod_path).resolve())
            sys.path.insert(0, dir_path)
            mod_name = "retrieve"  # convention; user can override via :name
        else:
            mod_name = mod_path
        mod = importlib.import_module(mod_name)
        return getattr(mod, fn_name)
    except (ImportError, AttributeError, ValueError) as exc:
        print(f"[WARN] could not load VECTOR_SEARCH_MODULE={spec!r}: {exc}", file=sys.stderr)
        return None


def score(answer_text: str, expected_entities: list[str]) -> float:
    """Recall@expected: fraction of expected entities mentioned in the answer."""
    if not expected_entities:
        return 0.0
    at = answer_text.lower()
    return sum(1 for e in expected_entities if e.lower() in at) / len(expected_entities)


def main() -> None:
    eval_set = json.loads(Path("data/eval.json").read_text())
    vector_search = _load_vector_search()
    results: list[dict] = []

    for item in eval_set:
        q = item["q"]
        exp = item["expected_entities"]

        t0 = time.time()
        g = graph_answer(q)
        g_time = time.time() - t0
        g_recall = score(g["answer"], exp)

        vector_record: dict | None = None
        if vector_search is not None:
            try:
                t0 = time.time()
                v = vector_search(q, k=5)
                v_latency = time.time() - t0
                vector_record = {
                    "recall":  score(v["answer"], exp),
                    "latency": round(v_latency, 2),
                }
            except Exception as exc:  # noqa: BLE001 — graceful fallback
                print(f"[WARN] vector_search failed on q={q!r}: {exc}", file=sys.stderr)

        record: dict = {
            "q": q,
            "expected": exp,
            "graphrag": {
                "recall":  g_recall,
                "latency": round(g_time, 2),
                "edges":   g["edges_used"],
            },
        }
        if vector_record is not None:
            record["vectorrag"] = vector_record
            v_recall = vector_record["recall"]
            if g_recall > v_recall:
                record["winner"] = "graph"
            elif v_recall > g_recall:
                record["winner"] = "vector"
            else:
                record["winner"] = "tie"
        results.append(record)

    Path("results").mkdir(exist_ok=True)
    Path("results/comparison.json").write_text(json.dumps(results, indent=2))

    g_avg_r = sum(r["graphrag"]["recall"]  for r in results) / len(results)
    g_avg_t = sum(r["graphrag"]["latency"] for r in results) / len(results)
    print(f"\nGraphRAG  avg recall = {g_avg_r:.2f}   avg latency = {g_avg_t:.2f}s")

    if vector_search is not None and any("vectorrag" in r for r in results):
        v_results = [r for r in results if "vectorrag" in r]
        v_avg_r = sum(r["vectorrag"]["recall"]  for r in v_results) / len(v_results)
        v_avg_t = sum(r["vectorrag"]["latency"] for r in v_results) / len(v_results)
        win_graph  = sum(1 for r in v_results if r.get("winner") == "graph")
        win_vector = sum(1 for r in v_results if r.get("winner") == "vector")
        ties       = sum(1 for r in v_results if r.get("winner") == "tie")
        print(f"VectorRAG avg recall = {v_avg_r:.2f}   avg latency = {v_avg_t:.2f}s")
        print(f"\nWins — Graph: {win_graph}  Vector: {win_vector}  Ties: {ties}")
    else:
        print("(VectorRAG comparison skipped — set VECTOR_SEARCH_MODULE to enable.)")


if __name__ == "__main__":
    main()