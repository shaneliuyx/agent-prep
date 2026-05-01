"""Compare GraphRAG vs vector-RAG (reuses Week 2 pipeline) on multi-hop queries."""
import json
import sys
import time
from pathlib import Path

# query_graph is a sibling module in src/. Running `python src/compare.py`
# from the project root puts src/ on sys.path automatically.
from query_graph import answer as graph_answer

# Week 2 pipeline lives in lab-02-rerank-compress/src/ as `retrieve.py`,
# which exposes search_with_rerank(query, k=5) -> {"answer", "chunks"}.
sys.path.insert(0, "../lab-02-rerank-compress/src")
from retrieve import search_with_rerank  # noqa: E402


def score(answer_text: str, expected_entities: list[str]) -> float:
    """Recall@expected: fraction of expected entities mentioned in the answer."""
    if not expected_entities:
        return 0.0
    at = answer_text.lower()
    return sum(1 for e in expected_entities if e.lower() in at) / len(expected_entities)


def main() -> None:
    eval_set = json.loads(Path("data/eval.json").read_text())
    results = []

    for item in eval_set:
        q = item["q"]
        exp = item["expected_entities"]

        t0 = time.time()
        g = graph_answer(q)
        g_time = time.time() - t0
        g_recall = score(g["answer"], exp)

        t0 = time.time()
        v = search_with_rerank(q, k=5)
        v_time = time.time() - t0
        v_recall = score(v["answer"], exp)

        results.append({
            "q":         q,
            "expected":  exp,
            "graphrag":  {"recall": g_recall, "latency": round(g_time, 2), "edges": g["edges_used"]},
            "vectorrag": {"recall": v_recall, "latency": round(v_time, 2)},
            "winner":    "graph" if g_recall > v_recall else ("vector" if v_recall > g_recall else "tie"),
        })

    Path("results").mkdir(exist_ok=True)
    Path("results/comparison.json").write_text(json.dumps(results, indent=2))

    g_avg_r = sum(r["graphrag"]["recall"]   for r in results) / len(results)
    v_avg_r = sum(r["vectorrag"]["recall"]  for r in results) / len(results)
    g_avg_t = sum(r["graphrag"]["latency"]  for r in results) / len(results)
    v_avg_t = sum(r["vectorrag"]["latency"] for r in results) / len(results)
    win_graph  = sum(1 for r in results if r["winner"] == "graph")
    win_vector = sum(1 for r in results if r["winner"] == "vector")
    ties       = sum(1 for r in results if r["winner"] == "tie")

    print(f"\nGraphRAG  avg recall = {g_avg_r:.2f}   avg latency = {g_avg_t:.2f}s")
    print(f"VectorRAG avg recall = {v_avg_r:.2f}   avg latency = {v_avg_t:.2f}s")
    print(f"\nWins — Graph: {win_graph}  Vector: {win_vector}  Ties: {ties}")


if __name__ == "__main__":
    main()
