"""Three-way comparison v3 — Vector vs Graph vs Tree-v3 (cluster pre-fetch
+ chunk-level fallback + 2-model split).

Differences from compare_three.py:
- Uses COMBINED eval set (eval.json + eval_v2.json = 16 questions, vs 8 in v1)
- Tree backend uses the v3 retriever (AgenticTreeRetriever with summary_index
  + page_vector_index + entity_index), not the v1 query_tree.answer wrapper
- Optionally dual-scores Q3/Q9/Q11/Q12 with the GT-judge (gt_judge.py) and
  emits both entity-recall AND GT-pass scores per backend
- Output: results/three_way_v3.json

Pre-reqs (same as compare_three.py):
- Qdrant 'brk_2023_dense' collection (3857 points)
- Neo4j BrkEntity nodes + brk_entity_names index
- data/tree.json (multi-pass build)
- data/summary_index.json (cluster index)
- data/page_vectors.npy + .sparse.json (chunk fallback)
- data/eval_ground_truth.json (for the 4 GT-judged Qs)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from qdrant_client import QdrantClient

_LAB = Path(__file__).resolve().parents[1]
_REPO = _LAB.parent
sys.path.insert(0, str(_REPO / "shared"))
sys.path.insert(0, str(_REPO / "lab-02-5-graphrag" / "src"))
sys.path.insert(0, str(_LAB / "src"))

load_dotenv(_LAB / ".env")

from rag_hybrid import (                                # noqa: E402
    BGE_M3, BGE_RERANKER_V2_M3,
    CrossEncoderReranker, DenseEncoder, autoconfig,
)
from compare import score_substring, score_llm_judge   # noqa: E402
from query_brk_graph import answer as graph_answer     # noqa: E402
from gt_judge import score_against_ground_truth, load_ground_truth  # noqa: E402
from tree_index import (                                # noqa: E402
    AgenticTreeRetriever, AGENTIC_SYSTEM_TEMPLATE_V2,
    EntityIndex, TreeIndex,
)
from tree_index.summary_index import SummaryIndex      # noqa: E402
from tree_index.page_vector_index import PageVectorIndex  # noqa: E402

# -----------------------------------------------------------------------------
omlx = OpenAI(base_url=os.getenv("OMLX_BASE_URL"), api_key=os.getenv("OMLX_API_KEY"))
SONNET = os.getenv("MODEL_SONNET", "models/gemma-4-26B-A4B-it-heretic-4bit")
TREE_MODEL = os.getenv("MODEL_TREE", "models/MLX-Qwen3.5-9B-GLM5.1-Distill-v1-8bit")

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
    """Dense kNN + cross-encoder rerank + Gemma synthesis.

    `_rr.rerank()` from shared/rag_hybrid expects ScoredPoint-like objects
    with `.payload["text"]` (NOT raw text strings); pass `hits` directly.
    Returned tuples are `(point, score)` — extract `.payload["text"]`.
    """
    qv = _enc.encode([q])[0]
    hits = _qd.query_points(
        collection_name=_VECTOR_COLLECTION, query=qv.tolist(), limit=k * 4,
    ).points
    reranked = _rr.rerank(q, hits, top_k=k)
    top_text = [point.payload["text"] for point, _score in reranked]
    ctx = "\n\n---\n\n".join(top_text)
    r = omlx.chat.completions.create(
        model=SONNET, temperature=0.0, max_tokens=300,
        messages=[{"role": "user",
                   "content": VECTOR_ANSWER_PROMPT.format(ctx=ctx, q=q)}],
    )
    return {"answer": (r.choices[0].message.content or "").strip()}


# Tree v3 — instantiate once, reuse across questions
def _build_tree_retriever() -> AgenticTreeRetriever:
    tree = json.loads((_LAB / "data" / "tree.json").read_text())
    pdf_path = _LAB / "data" / "brk-2023-ar.pdf"
    pages = [(p.extract_text() or "") for p in PdfReader(str(pdf_path)).pages]

    def page_provider(s: int, e: int) -> str:
        sp, ep = max(0, s - 1), min(len(pages), e)
        return "\n\n".join(f"[page {i+1}]\n{pages[i]}" for i in range(sp, ep))

    def raw_provider(s: int, e: int) -> str:
        sp, ep = max(0, s - 1), min(len(pages), e)
        return "\n\n".join(pages[i] for i in range(sp, ep))

    ti = TreeIndex(tree)
    ei = EntityIndex(ti, page_provider=raw_provider)

    si = None
    try:
        si = SummaryIndex(
            index_path=_LAB / "data" / "summary_index.json",
            tree_path=_LAB / "data" / "tree.json",
        )
        from sentence_transformers import SentenceTransformer
        _bge = SentenceTransformer("BAAI/bge-m3", device="mps")
        si.set_embedder(lambda t: _bge.encode([t], normalize_embeddings=True)[0])
    except Exception as e:                              # noqa: BLE001
        print(f"[tree] SummaryIndex unavailable: {e}", flush=True)

    pvi = None
    try:
        from FlagEmbedding import BGEM3FlagModel
        _bge_hybrid = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)

        def hybrid(text: str) -> dict:
            o = _bge_hybrid.encode([text], return_dense=True, return_sparse=True,
                                    return_colbert_vecs=False)
            return {"dense": o["dense_vecs"][0],
                    "sparse": {int(k): float(v) for k, v in o["lexical_weights"][0].items()}}

        pvi = PageVectorIndex.load(
            _LAB / "data" / "page_vectors.npy", embedder=hybrid,
        )
    except Exception as e:                              # noqa: BLE001
        print(f"[tree] PageVectorIndex unavailable: {e}", flush=True)

    return AgenticTreeRetriever(
        tree=tree, page_provider=page_provider,
        model_client=omlx, model_name=TREE_MODEL,
        system_prompt=AGENTIC_SYSTEM_TEMPLATE_V2,
        tree_index=ti, entity_index=ei,
        summary_index=si, page_vector_index=pvi,
    )


def run(q: str, retriever, label: str) -> dict:
    t0 = time.time()
    try:
        out = retriever(q)
    except Exception as e:                               # noqa: BLE001
        return {"label": label, "answer": f"[ERROR {type(e).__name__}: {e}]",
                "latency": round(time.time() - t0, 2), "error": True}
    elapsed = time.time() - t0
    return {"label": label, "answer": out["answer"], "latency": round(elapsed, 2)}


def main() -> None:
    eval_a = json.loads((_LAB / "data" / "eval.json").read_text())
    eval_b = json.loads((_LAB / "data" / "eval_v2.json").read_text())
    eval_set = eval_a + eval_b
    print(f"[v3] eval set: {len(eval_set)} questions", flush=True)

    tree_retriever = _build_tree_retriever()
    tree_fn = tree_retriever.answer

    # Load GT for ALL 16 questions
    gt_map = load_ground_truth(_LAB / "data" / "eval_ground_truth.json")
    gt_qs = {entry["q"]: entry for entry in gt_map.values()}

    def _gt_str(r):
        if r.get("gt_pass") is True:
            return "PASS"
        if r.get("gt_pass") is False:
            return "FAIL"
        return "—"

    results = []
    for i, item in enumerate(eval_set):
        q, exp, q_type = item["q"], item["expected_entities"], item["type"]
        print(f"\n>>> Q{i+1} [{q_type}] {q[:70]}", flush=True)

        v = run(q, lambda x: vector_answer(x, k=5), "vector")
        g = run(q, graph_answer, "graph")
        t = run(q, tree_fn, "tree")

        if q not in gt_qs:
            print(f"  WARN: no ground truth for Q{i+1}; entity-recall fallback",
                  flush=True)

        for r in (v, g, t):
            # Substring kept for backward visibility; not used in aggregate.
            r["recall_substr"] = score_substring(r["answer"], exp)
            # Entity-recall judge kept for backward comparability with prior runs.
            r["recall_judge"], _ = score_llm_judge(q, r["answer"], exp)
            # GT-judge — PRIMARY metric for all 16 questions
            if q in gt_qs:
                gt = gt_qs[q]
                passed, rationale = score_against_ground_truth(
                    client=omlx, model=SONNET,
                    question=q, gt_answer=gt["gt_answer"],
                    pass_criteria=gt["pass_criteria"],
                    candidate_answer=r["answer"],
                )
                r["gt_pass"] = bool(passed)
                r["gt_rationale"] = rationale[:200]
            else:
                r["gt_pass"] = None
                r["gt_rationale"] = "no ground truth available"

        results.append({"q": q, "type": q_type, "expected": exp,
                        "vector": v, "graph": g, "tree": t})
        # Live screen output now shows GT-judge result first
        print(f"  V={_gt_str(v)}/{v['latency']:.1f}s  "
              f"G={_gt_str(g)}/{g['latency']:.1f}s  "
              f"T={_gt_str(t)}/{t['latency']:.1f}s "
              f"(entity: V={v['recall_judge']:.2f} G={g['recall_judge']:.2f} "
              f"T={t['recall_judge']:.2f})", flush=True)

    Path("results").mkdir(exist_ok=True)
    out_path = _LAB / "results" / "three_way_v3.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}", flush=True)

    print("\n--- Aggregate — PRIMARY GT-judge (binary pass/fail) ---", flush=True)
    for backend in ("vector", "graph", "tree"):
        passed = [r[backend].get("gt_pass") for r in results]
        n_pass = sum(1 for p in passed if p is True)
        n_fail = sum(1 for p in passed if p is False)
        n_skip = sum(1 for p in passed if p is None)
        rate = n_pass / max(1, n_pass + n_fail)
        avg_lat = sum(r[backend]["latency"] for r in results) / len(results)
        print(f"  {backend:<8}  pass_rate={rate:.3f}  "
              f"({n_pass} pass / {n_fail} fail / {n_skip} no-GT)  "
              f"lat={avg_lat:.1f}s", flush=True)

    print("\n--- Per-category (GT-judge pass rate) ---", flush=True)
    for t_cat in sorted({r["type"] for r in results}):
        bucket = [r for r in results if r["type"] == t_cat]
        scores = {}
        for b in ("vector", "graph", "tree"):
            passed = [r[b].get("gt_pass") for r in bucket]
            n_pass = sum(1 for p in passed if p is True)
            n_eval = sum(1 for p in passed if p is not None)
            scores[b] = n_pass / max(1, n_eval)
        print(f"  {t_cat:<28}  V={scores['vector']:.2f}  "
              f"G={scores['graph']:.2f}  T={scores['tree']:.2f}", flush=True)

    print("\n--- Legacy entity-recall (for comparison with prior runs) ---", flush=True)
    for backend in ("vector", "graph", "tree"):
        avg_jud = sum(r[backend]["recall_judge"] for r in results) / len(results)
        avg_sub = sum(r[backend]["recall_substr"] for r in results) / len(results)
        print(f"  {backend:<8}  entity_judge={avg_jud:.3f}  "
              f"substr={avg_sub:.3f}", flush=True)


if __name__ == "__main__":
    main()
