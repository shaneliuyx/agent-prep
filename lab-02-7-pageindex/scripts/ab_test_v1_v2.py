"""A/B harness: v1 (greedy navigate) vs v2 (entity-graph + auto-merge tools).

Runs both retrievers against a combined eval set:
- 8 original questions (data/eval.json)
- 8 NEW questions (data/eval_v2.json) authored against actual tree.json
  to test on unseen prompts and avoid overfitting

Outputs:
- results/ab_v1_v2.json — per-question scores + tool usage
- per-category aggregate table
- net delta (v2 - v1) on judge / substr / latency

The v1 path uses AGENTIC_SYSTEM_TEMPLATE (greedy navigate over tree, no
v2 tools); v2 uses AGENTIC_SYSTEM_TEMPLATE_V2 (TITLE-LITERAL +
entity-graph + auto-merge + convergence rules).

Both share the same model + same tree.json + same PDF — isolating the
prompt + tool-set as the only variable.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

_LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_LAB_ROOT / "src"))
sys.path.insert(0, str(_LAB_ROOT.parents[0] / "shared"))
sys.path.insert(0, str(_LAB_ROOT.parents[0] / "lab-02-5-graphrag" / "src"))

from openai import OpenAI                              # noqa: E402
from pypdf import PdfReader                             # noqa: E402

from compare import score_substring, score_llm_judge   # noqa: E402
from tree_index import (                                # noqa: E402
    AGENTIC_SYSTEM_TEMPLATE,
    AGENTIC_SYSTEM_TEMPLATE_V2,
    AgenticTreeRetriever,
    EntityIndex,
    TreeIndex,
)

omlx = OpenAI(base_url=os.getenv("OMLX_BASE_URL"),
              api_key=os.getenv("OMLX_API_KEY"))
MODEL = os.getenv("MODEL_TREE") or os.getenv("MODEL_SONNET") or ""


def make_page_provider(pdf_path: str):
    pages = [p.extract_text() or "" for p in PdfReader(pdf_path).pages]

    def provider(start: int, end: int) -> str:
        sp = max(0, int(start) - 1)
        ep = min(len(pages), int(end))
        return "\n\n".join(f"[page {i+1}]\n{pages[i]}" for i in range(sp, ep))

    def raw_provider(start: int, end: int) -> str:
        sp = max(0, int(start) - 1)
        ep = min(len(pages), int(end))
        return "\n\n".join(pages[i] for i in range(sp, ep))

    return provider, raw_provider


def build_v1(tree, pdf_path):
    page_provider, _ = make_page_provider(pdf_path)
    return AgenticTreeRetriever(
        tree=tree, page_provider=page_provider,
        model_client=omlx, model_name=MODEL,
        system_prompt=AGENTIC_SYSTEM_TEMPLATE,
    )


def build_v2(tree, pdf_path):
    page_provider, raw_provider = make_page_provider(pdf_path)
    ti = TreeIndex(tree)
    ei = EntityIndex(ti, page_provider=raw_provider)
    return AgenticTreeRetriever(
        tree=tree, page_provider=page_provider,
        model_client=omlx, model_name=MODEL,
        system_prompt=AGENTIC_SYSTEM_TEMPLATE_V2,
        tree_index=ti, entity_index=ei,
    )


def run_eval(retriever, label: str, eval_set):
    rows = []
    for item in eval_set:
        q, exp, ty = item["q"], item["expected_entities"], item["type"]
        t0 = time.time()
        try:
            out = retriever.answer(q)
            ans = out["answer"]
            tools = [tc["tool"] for tc in out.get("tool_calls", [])]
        except Exception as e:                          # noqa: BLE001
            ans = f"[ERROR {type(e).__name__}: {e}]"
            tools = []
        lat = time.time() - t0
        sub = score_substring(ans, exp)
        try:
            judge, _ = score_llm_judge(q, ans, exp)
        except Exception:                                # noqa: BLE001
            judge = 0.0
        rows.append({"variant": label, "q": q, "type": ty,
                     "answer": ans, "sub": sub, "judge": judge,
                     "lat": lat, "tools": tools})
        print(f"  [{label}][{ty[:14]:14s}] {q[:50]:50s} judge={judge:.2f} sub={sub:.2f} lat={lat:.1f}s")
    return rows


def aggregate(rows, label):
    j = sum(r["judge"] for r in rows) / len(rows)
    s = sum(r["sub"] for r in rows) / len(rows)
    l = sum(r["lat"] for r in rows) / len(rows)
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["type"]].append(r["judge"])
    cat = {k: sum(v) / len(v) for k, v in by_cat.items()}
    return {"label": label, "agg_judge": j, "agg_sub": s, "agg_lat": l,
            "per_cat": cat, "rows": rows}


def main():
    tree = json.loads((_LAB_ROOT / "data" / "tree.json").read_text())
    pdf_path = str(_LAB_ROOT / "data" / "brk-2023-ar.pdf")
    eval_a = json.loads((_LAB_ROOT / "data" / "eval.json").read_text())
    eval_b = json.loads((_LAB_ROOT / "data" / "eval_v2.json").read_text())
    full = eval_a + eval_b
    print(f"Eval set: {len(eval_a)} original + {len(eval_b)} new = {len(full)} questions\n")

    print("--- Building v1 (greedy nav, AGENTIC_SYSTEM_TEMPLATE) ---")
    v1 = build_v1(tree, pdf_path)
    print("\n--- Running v1 ---")
    v1_rows = run_eval(v1, "v1", full)
    a1 = aggregate(v1_rows, "v1")

    print("\n--- Building v2 (entity-graph + auto-merge, AGENTIC_SYSTEM_TEMPLATE_V2) ---")
    v2 = build_v2(tree, pdf_path)
    print("\n--- Running v2 ---")
    v2_rows = run_eval(v2, "v2", full)
    a2 = aggregate(v2_rows, "v2")

    print("\n=== A/B Summary ===")
    print(f"  Aggregate (judge):   v1={a1['agg_judge']:.3f}  v2={a2['agg_judge']:.3f}  Δ={a2['agg_judge']-a1['agg_judge']:+.3f}")
    print(f"  Aggregate (substr):  v1={a1['agg_sub']:.3f}  v2={a2['agg_sub']:.3f}  Δ={a2['agg_sub']-a1['agg_sub']:+.3f}")
    print(f"  Aggregate (latency): v1={a1['agg_lat']:.1f}s  v2={a2['agg_lat']:.1f}s  Δ={a2['agg_lat']-a1['agg_lat']:+.1f}s")
    print("\n  Per-category (judge):")
    for cat in sorted(set(a1["per_cat"]) | set(a2["per_cat"])):
        v1c = a1["per_cat"].get(cat, 0)
        v2c = a2["per_cat"].get(cat, 0)
        print(f"    {cat:30s}  v1={v1c:.2f}  v2={v2c:.2f}  Δ={v2c-v1c:+.2f}")

    out = {"v1": a1, "v2": a2,
           "delta_judge": a2["agg_judge"] - a1["agg_judge"],
           "delta_sub": a2["agg_sub"] - a1["agg_sub"],
           "delta_lat": a2["agg_lat"] - a1["agg_lat"]}
    (_LAB_ROOT / "results" / "ab_v1_v2.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {_LAB_ROOT / 'results' / 'ab_v1_v2.json'}")


if __name__ == "__main__":
    main()
