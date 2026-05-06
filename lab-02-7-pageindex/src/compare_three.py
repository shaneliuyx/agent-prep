"""Three-way comparison on Berkshire 2023 annual report:
  - VectorRAG (shared/rag_hybrid against Qdrant collection 'brk_2023_dense')
  - GraphRAG  (lab-02-7-pageindex local copy: query_brk_graph against Neo4j
               BrkEntity nodes + brk_entity_names index — all under the
               default 'neo4j' database, isolated from W2.5's :Entity nodes
               via label namespacing)
  - TreeIndex (query_tree against data/tree.json)

Same eval set, same corpus across all three backends. Validates W2.7's
architectural thesis: tree wins citation + refusal; vector wins factoid
lookup; graph degenerates on single-document corpora (Berkshire's annual
report is one entity-dense document, not a multi-document graph — graph
collapses to a star centered on Berkshire itself).

Pre-reqs (all must complete in order BEFORE running this):
  1. python src/build_tree.py             → data/tree.json
  2. python src/_brk_corpus.py            → data/brk_corpus.json
  3. python src/ingest_brk_to_vector.py   → Qdrant 'brk_2023_dense' collection
  4. python src/build_brk_graph.py        → Neo4j BrkEntity nodes + brk_entity_names index

Reuses scoring helpers from lab-02-5/src/compare.py via cross-lab import.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI
from qdrant_client import QdrantClient

# --- shared/rag_hybrid for the vector backend ---------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "shared"))

from rag_hybrid import (  # noqa: E402
    BGE_M3,
    BGE_RERANKER_V2_M3,
    CrossEncoderReranker,
    DenseEncoder,
    autoconfig,
)

# --- W2.5 scoring helpers (reuse) --------------------------------------------
sys.path.insert(0, str(_REPO_ROOT / "lab-02-5-graphrag" / "src"))
from compare import score_substring, score_llm_judge  # noqa: E402

# --- W2.7-local backend modules ----------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from query_brk_graph import answer as graph_answer  # noqa: E402
from query_tree import answer as tree_answer        # noqa: E402

# -----------------------------------------------------------------------------
# Vector backend (W2.7-specific: queries the Berkshire-only collection
# 'brk_2023_dense' with shared/rag_hybrid encoder + reranker via autoconfig)
# -----------------------------------------------------------------------------
omlx = OpenAI(base_url=os.getenv("OMLX_BASE_URL"), api_key=os.getenv("OMLX_API_KEY"))
SONNET = os.getenv("MODEL_SONNET")

_qd = QdrantClient(url="http://127.0.0.1:6333", timeout=60)
_enc = DenseEncoder(autoconfig.encoder_config_for(BGE_M3))
_rr = CrossEncoderReranker(autoconfig.recommend(BGE_M3, BGE_RERANKER_V2_M3).reranker)
_VECTOR_COLLECTION = "brk_2023_dense"

VECTOR_ANSWER_PROMPT = """Use ONLY the context below. Answer in 1-3 sentences.
If the context does not contain the answer, say exactly: insufficient context.

Context:
{ctx}

Question: {q}
Answer:"""


def vector_answer(q: str, k: int = 5) -> dict:
    """Dense kNN + cross-encoder rerank + answer LLM. Mirrors lab-03 pattern."""
    qv = _enc.encode([q])[0]
    hits = _qd.query_points(
        _VECTOR_COLLECTION,
        query=qv.tolist(),
        limit=30,
        with_payload=True,
    ).points
    _rr._ensure_loaded()
    pairs = [(q, h.payload["text"]) for h in hits]
    scores = _rr._model.predict(pairs, batch_size=_rr.cfg.spec.batch_size)
    top = [h for h, _ in sorted(zip(hits, scores), key=lambda x: -x[1])[:k]]
    ctx = "\n\n".join(h.payload["text"] for h in top)
    r = omlx.chat.completions.create(
        model=SONNET, temperature=0.0, max_tokens=300,
        messages=[{"role": "user", "content": VECTOR_ANSWER_PROMPT.format(ctx=ctx, q=q)}],
    )
    return {
        "question": q,
        "answer": (r.choices[0].message.content or "").strip(),
        "contexts": [h.payload["text"] for h in top],
    }


def run(q: str, retriever, label: str) -> dict:
    t0 = time.time()
    try:
        out = retriever(q)
    except Exception as e:  # noqa: BLE001 — capture for reporting, do not abort
        return {"label": label, "answer": f"[ERROR {type(e).__name__}: {e}]",
                "latency": round(time.time() - t0, 2), "error": True}
    elapsed = time.time() - t0
    return {"label": label, "answer": out["answer"], "latency": round(elapsed, 2)}


def main() -> None:
    eval_set = json.loads(Path("data/eval.json").read_text())
    results = []
    for item in eval_set:
        q, exp, q_type = item["q"], item["expected_entities"], item["type"]
        print(f"\n>>> [{q_type}] {q}")

        v = run(q, lambda x: vector_answer(x, k=5), "vector")
        g = run(q, graph_answer, "graph")
        t = run(q, tree_answer, "tree")

        for r in (v, g, t):
            r["recall_substr"] = score_substring(r["answer"], exp)
            r["recall_judge"], _ = score_llm_judge(q, r["answer"], exp)

        results.append({"q": q, "type": q_type, "expected": exp,
                        "vector": v, "graph": g, "tree": t})
        print(
            f"  V={v['recall_judge']:.2f}/{v['latency']:.1f}s  "
            f"G={g['recall_judge']:.2f}/{g['latency']:.1f}s  "
            f"T={t['recall_judge']:.2f}/{t['latency']:.1f}s"
        )

    Path("results").mkdir(exist_ok=True)
    Path("results/three_way.json").write_text(json.dumps(results, indent=2))

    print("\n--- Aggregate (LLM-judge / latency) ---")
    for backend in ("vector", "graph", "tree"):
        avg_jud = sum(r[backend]["recall_judge"] for r in results) / len(results)
        avg_sub = sum(r[backend]["recall_substr"] for r in results) / len(results)
        avg_lat = sum(r[backend]["latency"] for r in results) / len(results)
        print(f"  {backend:<8}  judge={avg_jud:.2f}  substr={avg_sub:.2f}  lat={avg_lat:.1f}s")

    print("\n--- Per-category (LLM-judge) ---")
    types = sorted({r["type"] for r in results})
    for t in types:
        bucket = [r for r in results if r["type"] == t]
        v = sum(r["vector"]["recall_judge"] for r in bucket) / len(bucket)
        g = sum(r["graph"]["recall_judge"] for r in bucket) / len(bucket)
        tr = sum(r["tree"]["recall_judge"] for r in bucket) / len(bucket)
        print(f"  {t:<25}  V={v:.2f}  G={g:.2f}  T={tr:.2f}")


if __name__ == "__main__":
    main()
