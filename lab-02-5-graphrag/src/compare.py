"""Compare GraphRAG vs vector-RAG (reuses Week 2 pipeline) on multi-hop queries.

Two scoring metrics per question:
  1. **substring** — fraction of expected entities whose lower-cased text
     appears as a substring of the answer. Cheap, deterministic, but
     undercounts semantic equivalents ("Steve Jobs" vs "Steven Paul Jobs",
     "purchased" vs "acquired", "co-founded" vs "co-founder").
  2. **llm_judge** — LLM-judge per-expected-entity decision allowing
     semantic equivalence. Honest about whether the answer captures the
     intended fact regardless of surface form.

Both metrics are recorded per question + averaged per category. The
substring metric stays primary for backward-compat with the frozen
comparison.json baseline (SHA prefix 39d0e96346d54b04, tracked in
shared/parity/pre_refactor.json). The llm_judge metric is the honest
recall number — see RESULTS.md v10d for the side-by-side gap.
"""
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# query_graph is a sibling module in src/. Running `python src/compare.py`
# from the project root puts src/ on sys.path automatically.
from query_graph import answer as graph_answer

# Week 2 pipeline lives in lab-02-rerank-compress/src/ as `retrieve.py`,
# which exposes search_with_rerank(query, k=5) -> {"answer", "chunks"}.
sys.path.insert(0, "../lab-02-rerank-compress/src")
from retrieve import search_with_rerank  # noqa: E402

load_dotenv()
_omlx = OpenAI(base_url=os.getenv("OMLX_BASE_URL"), api_key=os.getenv("OMLX_API_KEY"))
_JUDGE_MODEL = os.getenv("MODEL_SONNET", "gemma-4-26B-A4B-it-heretic-4bit")

_JUDGE_SYSTEM = """You are an evaluation judge for a knowledge-graph QA system. Given:
  - A question
  - A list of expected entities/concepts the correct answer should mention
  - An actual answer

For EACH expected entity, decide whether the answer correctly mentions it OR
a clear semantic equivalent.

Match (semantic equivalents to ACCEPT):
  - "Steve Jobs" matches "Steven Jobs", "Steven Paul Jobs", "S. Jobs"
  - "acquired" matches "purchased", "bought", "took over"
  - "co-founder" matches "co-founded", "co-founding", "founded with"
  - "Tesla" matches "Tesla, Inc.", "Tesla Motors", "Tesla Inc"
  - "Stanford" matches "Stanford University", "Stanford U."
  - "Bill Gates" matches "William Henry Gates III", "William Gates III"

Not match (do NOT accept):
  - "Tesla" does NOT match "SpaceX"
  - "founded" does NOT match "left" or "resigned from"
  - "Stanford" does NOT match "Berkeley"

Output strict JSON only:
  {"matches": {"<expected entity>": true | false, ...}}

Use the EXACT expected-entity strings as keys. Output the entire object
on one line, no markdown, no commentary."""


def score_substring(answer_text: str, expected_entities: list[str]) -> float:
    """Recall@expected: fraction of expected entities mentioned (substring,
    case-insensitive) in the answer. Original eval metric — kept for
    backward-compat with frozen comparison.json hash."""
    if not expected_entities:
        return 0.0
    at = answer_text.lower()
    return sum(1 for e in expected_entities if e.lower() in at) / len(expected_entities)


def score_llm_judge(
    query: str, answer_text: str, expected_entities: list[str]
) -> tuple[float, dict]:
    """LLM-judge per-entity hit detection allowing semantic equivalents.
    Returns (recall, per_entity_matches_dict). Falls back to substring score
    on JSON parse failure or LLM error."""
    if not expected_entities:
        return 0.0, {}
    user_msg = (
        f"Question: {query}\n\n"
        f"Expected entities: {json.dumps(expected_entities)}\n\n"
        f"Actual answer: {answer_text}"
    )
    try:
        resp = _omlx.chat.completions.create(
            model=_JUDGE_MODEL,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=400,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or "{}"
        parsed = json.loads(content)
        matches = parsed.get("matches", {})
        if not isinstance(matches, dict):
            raise ValueError(f"matches not a dict: {type(matches).__name__}")
        hits = sum(1 for e in expected_entities if matches.get(e, False))
        return hits / len(expected_entities), matches
    except Exception as e:
        # Fall back to substring on any judge failure (network, JSON, etc.)
        return score_substring(answer_text, expected_entities), {
            "_fallback": "substring",
            "_error": f"{type(e).__name__}: {e}",
        }


def _summarize(records: list[dict], label: str) -> None:
    if not records:
        print(f"{label}: (no records)")
        return
    g_sub = sum(r["graphrag"]["recall"] for r in records) / len(records)
    v_sub = sum(r["vectorrag"]["recall"] for r in records) / len(records)
    g_jud = sum(r["graphrag"]["recall_judge"] for r in records) / len(records)
    v_jud = sum(r["vectorrag"]["recall_judge"] for r in records) / len(records)
    g_lat = sum(r["graphrag"]["latency"] for r in records) / len(records)
    v_lat = sum(r["vectorrag"]["latency"] for r in records) / len(records)
    wins_g = sum(1 for r in records if r["winner_judge"] == "graph")
    wins_v = sum(1 for r in records if r["winner_judge"] == "vector")
    ties = sum(1 for r in records if r["winner_judge"] == "tie")
    print(
        f"{label:<22}  n={len(records):>2}  "
        f"Graph={g_sub:.2f}|{g_jud:.2f}/{g_lat:.1f}s  "
        f"Vector={v_sub:.2f}|{v_jud:.2f}/{v_lat:.1f}s  "
        f"W/L/T={wins_g}/{wins_v}/{ties}"
    )


def main() -> None:
    eval_set = json.loads(Path("data/eval.json").read_text())
    results = []

    for i, item in enumerate(eval_set):
        q = item["q"]
        exp = item["expected_entities"]
        q_type = item.get("type", "unknown")

        t0 = time.time()
        g = graph_answer(q)
        g_time = time.time() - t0
        g_substr = score_substring(g["answer"], exp)
        g_judge, g_judge_detail = score_llm_judge(q, g["answer"], exp)

        t0 = time.time()
        v = search_with_rerank(q, k=5)
        v_time = time.time() - t0
        v_substr = score_substring(v["answer"], exp)
        v_judge, v_judge_detail = score_llm_judge(q, v["answer"], exp)

        # Winner uses judge metric (more honest — substring-only winner is
        # preserved as `winner_substring` for backward-compat).
        winner_judge = (
            "graph" if g_judge > v_judge
            else ("vector" if v_judge > g_judge else "tie")
        )
        winner_substring = (
            "graph" if g_substr > v_substr
            else ("vector" if v_substr > g_substr else "tie")
        )

        results.append({
            "q":         q,
            "type":      q_type,
            "expected":  exp,
            "graphrag":  {
                "recall":        g_substr,           # primary metric (backward compat)
                "recall_judge":  g_judge,            # LLM-judge metric
                "judge_detail":  g_judge_detail,
                "latency":       round(g_time, 2),
                "edges":         g["edges_used"],
            },
            "vectorrag": {
                "recall":        v_substr,
                "recall_judge":  v_judge,
                "judge_detail":  v_judge_detail,
                "latency":       round(v_time, 2),
            },
            "winner":           winner_substring,    # backward compat — substring-based
            "winner_judge":     winner_judge,        # honest winner — judge-based
        })

        # Per-question progress (no waiting for the summary at end).
        print(
            f"  Q{i+1:02d} [{q_type}] "
            f"G={g_substr:.2f}|{g_judge:.2f} "
            f"V={v_substr:.2f}|{v_judge:.2f} "
            f"-> {winner_judge} :: {q[:60]}"
        )

    Path("results").mkdir(exist_ok=True)
    Path("results/comparison.json").write_text(json.dumps(results, indent=2))

    print("\n" + "-" * 92)
    print(
        f"{'CATEGORY':<22}  {'N':>2}  "
        f"{'GraphRAG sub|jud/Lat':<26}  "
        f"{'VectorRAG sub|jud/Lat':<26}  "
        f"W/L/T (judge)"
    )
    print("-" * 92)
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
