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


def _summarize(records: list[dict], label: str) -> None:
    if not records:
        print(f"{label}: (no records)")
        return
    g_avg_r = sum(r["graphrag"]["recall"]   for r in records) / len(records)
    v_avg_r = sum(r["vectorrag"]["recall"]  for r in records) / len(records)
    g_avg_t = sum(r["graphrag"]["latency"]  for r in records) / len(records)
    v_avg_t = sum(r["vectorrag"]["latency"] for r in records) / len(records)
    wins_g = sum(1 for r in records if r["winner"] == "graph")
    wins_v = sum(1 for r in records if r["winner"] == "vector")
    ties   = sum(1 for r in records if r["winner"] == "tie")
    print(f"{label:<22}  n={len(records):>2}  "
          f"Graph={g_avg_r:.2f}/{g_avg_t:.1f}s  "
          f"Vector={v_avg_r:.2f}/{v_avg_t:.1f}s  "
          f"W/L/T={wins_g}/{wins_v}/{ties}")


def main() -> None:
    eval_set = json.loads(Path("data/eval.json").read_text())
    results = []

    for item in eval_set:
        q = item["q"]
        exp = item["expected_entities"]
        q_type = item.get("type", "unknown")

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
            "type":      q_type,
            "expected":  exp,
            "graphrag":  {"recall": g_recall, "latency": round(g_time, 2), "edges": g["edges_used"]},
            "vectorrag": {"recall": v_recall, "latency": round(v_time, 2)},
            "winner":    "graph" if g_recall > v_recall else ("vector" if v_recall > g_recall else "tie"),
        })

    Path("results").mkdir(exist_ok=True)
    Path("results/comparison.json").write_text(json.dumps(results, indent=2))

    print("\n" + "-" * 80)
    print(f"{'CATEGORY':<22}  {'N':>2}  {'GraphRAG R/Lat':<22}  {'VectorRAG R/Lat':<22}  W/L/T")
    print("-" * 80)
    _summarize(results, "ALL")
    print()
    seen_types: list[str] = []
    for r in results:
        if r["type"] not in seen_types:
            seen_types.append(r["type"])
    for t in seen_types:
        bucket = [r for r in results if r["type"] == t]
        _summarize(bucket, f"  {t}")


if __name__ == "__main__":
    main()
