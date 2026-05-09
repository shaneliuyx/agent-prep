"""Run ONE variant (v1 or v2) against the combined eval set, write per-variant
results to results/ab_<variant>.json. Used by ab_test_v1_v2_isolated.py which
spawns this script as a subprocess for each variant — avoids same-process
oMLX KV-cache pollution that hit the prior in-process A/B harness.

Usage:
  python scripts/run_one_variant.py v1   # runs v1 (greedy nav, no v2 tools)
  python scripts/run_one_variant.py v2   # runs v2 (entity-graph + auto-merge)
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

from dotenv import load_dotenv                          # noqa: E402
from openai import OpenAI                              # noqa: E402
from pypdf import PdfReader                             # noqa: E402

# Load lab-02-7-pageindex/.env so OMLX_BASE_URL / OMLX_API_KEY / MODEL_TREE
# resolve correctly under subprocess invocation.
load_dotenv(_LAB_ROOT / ".env")

from compare import score_substring, score_llm_judge   # noqa: E402
from tree_index import (                                # noqa: E402
    AGENTIC_SYSTEM_TEMPLATE,
    AGENTIC_SYSTEM_TEMPLATE_V2,
    AgenticTreeRetriever,
    EnsembleTreeRetriever,
    EntityIndex,
    TreeIndex,
)
from tree_index.summary_index import SummaryIndex      # noqa: E402
from tree_index.page_vector_index import PageVectorIndex  # noqa: E402

# Phoenix tracing — auto-instruments OpenAI calls. Lazy/optional: skip silently
# if not installed or server not reachable. View at http://127.0.0.1:6006.
sys.path.insert(0, str(_LAB_ROOT.parents[0] / "shared"))
try:
    from phoenix_tracing import init_phoenix, phoenix_span  # noqa: E402
    _PHOENIX_OK = True
except Exception:                                            # noqa: BLE001
    _PHOENIX_OK = False
    def phoenix_span(label, attrs=None):                     # type: ignore
        from contextlib import nullcontext
        return nullcontext()


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


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "v1"
    if variant not in {"v1", "v2", "ensemble"}:
        print(f"ERROR: variant must be v1, v2, or ensemble; got {variant!r}",
              file=sys.stderr)
        sys.exit(1)

    if _PHOENIX_OK:
        try:
            init_phoenix(project_name=f"lab-02-7-{variant}",
                         server_url="http://127.0.0.1:6006")
            print(f"[{variant}] Phoenix tracing enabled "
                  f"(http://127.0.0.1:6006/projects)", flush=True)
        except Exception as e:                              # noqa: BLE001
            print(f"[{variant}] Phoenix init failed (continuing without): "
                  f"{type(e).__name__}: {str(e)[:80]}", flush=True)

    omlx = OpenAI(base_url=os.getenv("OMLX_BASE_URL"),
                  api_key=os.getenv("OMLX_API_KEY"))
    model = os.getenv("MODEL_TREE") or os.getenv("MODEL_SONNET") or ""
    # Synthesis-tier model for ensemble pick-best step. Defaults to retriever
    # model in single-model setups; in split mode points to MODEL_SONNET.
    synthesis_model = os.getenv("MODEL_SONNET") or model
    print(f"[{variant}] retriever_model={model}  synthesis_model={synthesis_model}", flush=True)

    tree = json.loads((_LAB_ROOT / "data" / "tree.json").read_text())
    pdf_path = str(_LAB_ROOT / "data" / "brk-2023-ar.pdf")
    eval_a = json.loads((_LAB_ROOT / "data" / "eval.json").read_text())
    eval_b = json.loads((_LAB_ROOT / "data" / "eval_v2.json").read_text())
    full = eval_a + eval_b
    print(f"[{variant}] eval set: {len(full)} questions", flush=True)

    page_provider, raw_provider = _make_pp(pdf_path)

    def _load_summary_index():
        """Best-effort SummaryIndex load. Returns None if missing/stale.

        Embedder is BGE-M3 on MPS (reused from prompt_dev / lab-02-3 baseline).
        """
        try:
            si = SummaryIndex(
                index_path=_LAB_ROOT / "data" / "summary_index.json",
                tree_path=_LAB_ROOT / "data" / "tree.json",
            )
            from sentence_transformers import SentenceTransformer
            _bge = SentenceTransformer("BAAI/bge-m3", device="mps")
            si.set_embedder(
                lambda t: _bge.encode([t], normalize_embeddings=True)[0]
            )
            print(f"[{variant}] SummaryIndex loaded: {len(si.clusters)} clusters",
                  flush=True)
            return si
        except (FileNotFoundError, RuntimeError, ValueError) as e:
            print(f"[{variant}] SummaryIndex unavailable: {type(e).__name__}: {e}",
                  flush=True)
            return None

    def _load_page_vector_index():
        """Best-effort PageVectorIndex load. Returns None if missing.

        Hybrid embedder: BGE-M3 dense (1024-dim) + sparse (token weights)
        via FlagEmbedding. Single forward pass returns both per query —
        sparse is essentially free. Used by AgenticTreeRetriever's
        chunk-level fallback (only fires on refusal).
        """
        try:
            from FlagEmbedding import BGEM3FlagModel
            _bge = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)

            def hybrid_embed(text: str) -> dict:
                out = _bge.encode(
                    [text], return_dense=True, return_sparse=True,
                    return_colbert_vecs=False,
                )
                dense = out["dense_vecs"][0]
                sparse_raw = out["lexical_weights"][0]
                sparse = {int(tok): float(w) for tok, w in sparse_raw.items()}
                return {"dense": dense, "sparse": sparse}

            pvi = PageVectorIndex.load(
                _LAB_ROOT / "data" / "page_vectors.npy",
                embedder=hybrid_embed,
            )
            print(f"[{variant}] PageVectorIndex loaded: {pvi.num_pages} pages "
                  f"({'hybrid' if pvi.sparse_embeddings else 'dense-only'})",
                  flush=True)
            return pvi
        except Exception as e:                                 # noqa: BLE001
            print(f"[{variant}] PageVectorIndex unavailable: "
                  f"{type(e).__name__}: {e}", flush=True)
            return None

    if variant == "v1":
        retriever = AgenticTreeRetriever(
            tree=tree, page_provider=page_provider,
            model_client=omlx, model_name=model,
            system_prompt=AGENTIC_SYSTEM_TEMPLATE,
        )
    elif variant == "v2":
        ti = TreeIndex(tree)
        ei = EntityIndex(ti, page_provider=raw_provider)
        si = _load_summary_index()
        pvi = _load_page_vector_index()
        retriever = AgenticTreeRetriever(
            tree=tree, page_provider=page_provider,
            model_client=omlx, model_name=model,
            system_prompt=AGENTIC_SYSTEM_TEMPLATE_V2,
            tree_index=ti, entity_index=ei,
            summary_index=si,
            page_vector_index=pvi,
        )
    else:  # ensemble (best-of-both v1 + v2 with synthesis-time LLM picker)
        ti = TreeIndex(tree)
        ei = EntityIndex(ti, page_provider=raw_provider)
        si = _load_summary_index()
        retriever = EnsembleTreeRetriever(
            tree=tree, page_provider=page_provider,
            model_client=omlx, model_name=model,
            synthesis_model=synthesis_model,
            tree_index=ti, entity_index=ei,
            summary_index=si,
        )

    # Health check the model. Retry up to 3 times on connection error
    # (vMLX/oMLX may have dropped a keepalive while reranker was loading).
    health_content = ""
    health_lat = 0.0
    last_err = None
    for attempt in range(3):
        try:
            t0 = time.time()
            health = omlx.chat.completions.create(
                model=model, temperature=0.0, max_tokens=10,
                messages=[{"role": "user", "content": "Reply with the word OK only."}],
            )
            health_lat = time.time() - t0
            health_content = (health.choices[0].message.content or "").strip()
            break
        except Exception as e:                          # noqa: BLE001
            last_err = e
            print(f"[{variant}] healthcheck attempt {attempt+1} failed: "
                  f"{type(e).__name__}: {str(e)[:120]}", flush=True)
            time.sleep(2.0)
    print(f"[{variant}] healthcheck: {health_content!r} ({health_lat:.2f}s)", flush=True)
    if not health_content:
        print(f"[{variant}] FATAL: healthcheck failed after 3 retries; abort. last_err={last_err}")
        sys.exit(2)

    rows = []
    for q_idx, item in enumerate(full):
        q, exp, ty = item["q"], item["expected_entities"], item["type"]
        t0 = time.time()
        with phoenix_span(label=f"q{q_idx+1:02d}-{ty[:14]}",
                          attrs={"variant": variant, "question": q,
                                 "type": ty, "expected": ",".join(exp)}):
            try:
                out = retriever.answer(q)
                ans = out["answer"]
                tools = [tc["tool"] for tc in out.get("tool_calls", [])]
                iters = out.get("iterations", 0)
                # Retry once on empty (oMLX state-degradation pattern: 0.6s
                # + iter=1 + empty answer).
                if not ans.strip() and iters == 1 and len(tools) == 0:
                    print(f"  [{variant}] EMPTY on q{q_idx+1}, sleeping 3s + retry...",
                          flush=True)
                    time.sleep(3.0)
                    out = retriever.answer(q)
                    ans = out["answer"]
                    tools = [tc["tool"] for tc in out.get("tool_calls", [])]
                    iters = out.get("iterations", 0)
            except Exception as e:                          # noqa: BLE001
                ans = f"[ERROR {type(e).__name__}: {e}]"
                tools = []
                iters = 0
        lat = time.time() - t0
        sub = score_substring(ans, exp)
        try:
            judge, _ = score_llm_judge(q, ans, exp)
        except Exception:                                # noqa: BLE001
            judge = 0.0
        rows.append({"variant": variant, "q": q, "type": ty,
                     "answer": ans, "sub": sub, "judge": judge,
                     "lat": lat, "iters": iters, "tools": tools})
        print(f"  [{variant}][{ty[:14]:14s}] {q[:50]:50s} judge={judge:.2f} sub={sub:.2f} lat={lat:.1f}s iters={iters}",
              flush=True)

    j = sum(r["judge"] for r in rows) / len(rows)
    s = sum(r["sub"] for r in rows) / len(rows)
    L = sum(r["lat"] for r in rows) / len(rows)
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["type"]].append(r["judge"])
    cat = {k: sum(v) / len(v) for k, v in by_cat.items()}
    summary = {"variant": variant, "agg_judge": j, "agg_sub": s, "agg_lat": L,
               "per_cat": cat, "rows": rows}
    out_path = _LAB_ROOT / "results" / f"ab_{variant}.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[{variant}] aggregate: judge={j:.3f} substr={s:.3f} lat={L:.1f}s", flush=True)
    print(f"[{variant}] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
