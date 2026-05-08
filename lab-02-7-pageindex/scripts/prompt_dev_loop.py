"""Tiny test harness for fast prompt-iteration on DWQ (or any model).

3 representative questions covering the failure modes we see in v1:
  Q-FACT     — section factoid (must surface canonical number)
  Q-SYNTH    — cross-section synthesis (must fetch ≥ 2 ranges)
  Q-OOD      — out-of-document refusal (must explain + close with phrase)

Each run prints the FULL answer + tool-call sequence so we can SEE what the
model did, not just the score. Use this loop:

  edit shared/tree_index/prompts.py
  → python scripts/prompt_dev_loop.py
  → read output, edit again

Total wall time per iteration: ~30-60s on DWQ (vs 5+ min on full eval).

Usage:
  python scripts/prompt_dev_loop.py            # default model from .env
  python scripts/prompt_dev_loop.py <model>    # override model
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

_LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_LAB_ROOT / "src"))
sys.path.insert(0, str(_LAB_ROOT.parents[0] / "shared"))
sys.path.insert(0, str(_LAB_ROOT.parents[0] / "lab-02-5-graphrag" / "src"))

from dotenv import load_dotenv  # noqa: E402
from openai import OpenAI  # noqa: E402
from pypdf import PdfReader  # noqa: E402

load_dotenv(_LAB_ROOT / ".env")

from compare import score_substring, score_llm_judge  # noqa: E402
from tree_index import (  # noqa: E402
    AGENTIC_SYSTEM_TEMPLATE,
    AGENTIC_SYSTEM_TEMPLATE_V2,
    AgenticTreeRetriever,
    EntityIndex,
    TreeIndex,
)


# Three representative probes — one per failure mode. Keep small.
PROBES = [
    {
        "id": "Q-FACT",
        "type": "section-factoid",
        "q": "What were Berkshire's total revenues in 2023?",
        "expected": ["364", "billion"],
    },
    {
        "id": "Q-SYNTH",
        "type": "cross-section synthesis",
        "q": "What did Buffett write about non-controlled businesses in 2023?",
        "expected": ["Coca-Cola", "American Express", "Apple"],
    },
    {
        "id": "Q-ENTITY",
        "type": "entity-graph (regex semantic gap)",
        "q": "What did Buffett describe as Berkshire's 'not-so-secret weapon' in the 2023 letter?",
        "expected": ["secret weapon", "Charlie", "shareholders", "patient"],
    },
    {
        "id": "Q-OOD",
        "type": "out-of-document",
        "q": "What is Berkshire Hathaway's stock price today?",
        "expected": ["insufficient context", "outside", "does not contain"],
    },
]


def _make_pp(pdf_path: str):
    pages = [p.extract_text() or "" for p in PdfReader(pdf_path).pages]

    def provider(s: int, e: int) -> str:
        sp = max(0, int(s) - 1)
        ep = min(len(pages), int(e))
        return "\n\n".join(f"[page {i+1}]\n{pages[i]}" for i in range(sp, ep))

    def raw(s: int, e: int) -> str:
        sp = max(0, int(s) - 1)
        ep = min(len(pages), int(e))
        return "\n\n".join(pages[i] for i in range(sp, ep))

    return provider, raw


def main() -> None:
    omlx = OpenAI(base_url=os.getenv("OMLX_BASE_URL"),
                  api_key=os.getenv("OMLX_API_KEY"))
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    variant = "v2" if "--v2" in sys.argv else "v1"
    model = args[0] if args else (
        os.getenv("MODEL_TREE") or os.getenv("MODEL_SONNET") or "")
    print(f"[prompt-dev] model={model} variant={variant}\n", flush=True)

    tree = json.loads((_LAB_ROOT / "data" / "tree.json").read_text())
    page_provider, raw_provider = _make_pp(
        str(_LAB_ROOT / "data" / "brk-2023-ar.pdf"))
    if variant == "v2":
        ti = TreeIndex(tree)
        ei = EntityIndex(ti, page_provider=raw_provider)
        retriever = AgenticTreeRetriever(
            tree=tree, page_provider=page_provider,
            model_client=omlx, model_name=model,
            system_prompt=AGENTIC_SYSTEM_TEMPLATE_V2,
            tree_index=ti, entity_index=ei,
        )
    else:
        retriever = AgenticTreeRetriever(
            tree=tree, page_provider=page_provider,
            model_client=omlx, model_name=model,
            system_prompt=AGENTIC_SYSTEM_TEMPLATE,
        )

    rows = []
    for p in PROBES:
        print(f"\n=== [{p['id']}] {p['q']!r}", flush=True)
        t0 = time.time()
        try:
            out = retriever.answer(p["q"])
            ans = out["answer"]
            tools = [tc["tool"] for tc in out.get("tool_calls", [])]
            iters = out.get("iterations", 0)
        except Exception as e:                              # noqa: BLE001
            ans = f"[ERROR {type(e).__name__}: {e}]"
            tools = []
            iters = 0
        lat = time.time() - t0
        sub = score_substring(ans, p["expected"])
        try:
            judge, _ = score_llm_judge(p["q"], ans, p["expected"])
        except Exception:                                    # noqa: BLE001
            judge = 0.0

        print(f"  iters={iters} tools={tools} lat={lat:.1f}s "
              f"judge={judge:.2f} sub={sub:.2f}", flush=True)
        print(f"  --- ANSWER ---\n{ans}\n  --- END ---", flush=True)
        rows.append({"id": p["id"], "q": p["q"], "answer": ans,
                     "tools": tools, "iters": iters, "lat": lat,
                     "sub": sub, "judge": judge})

    avg_j = sum(r["judge"] for r in rows) / len(rows)
    avg_s = sum(r["sub"] for r in rows) / len(rows)
    avg_l = sum(r["lat"] for r in rows) / len(rows)
    print(f"\n=== AGGREGATE judge={avg_j:.2f} sub={avg_s:.2f} lat={avg_l:.1f}s")

    out = _LAB_ROOT / "results" / "prompt_dev_last.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
