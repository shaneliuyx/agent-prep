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

sys.path.insert(0, str(_LAB_ROOT / "src"))
from gt_judge import (                                  # noqa: E402
    load_ground_truth, score_against_ground_truth,
)
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


def _reset_vmlx_cache_if_requested(args: list[str]) -> None:
    """Run scripts/reset_vmlx_cache.sh before eval if --reset-cache=soft|hard.

    Reduces cross-run KV cache pollution variance. Soft mode (~5s) evicts
    via LRU; hard mode (~60s) kills + respawns per-model server processes.
    Default off — existing workflows unaffected.
    """
    import subprocess
    mode = None
    for a in args:
        if a.startswith("--reset-cache="):
            mode = a.split("=", 1)[1]
    if mode in ("soft", "hard"):
        script = _LAB_ROOT / "scripts" / "reset_vmlx_cache.sh"
        flag = f"--{mode}"
        print(f"[reset] running {script.name} {flag} ...", flush=True)
        subprocess.run([str(script), flag], check=False)


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "v1"
    if variant not in {"v1", "v2", "ensemble"}:
        print(f"ERROR: variant must be v1, v2, or ensemble; got {variant!r}",
              file=sys.stderr)
        sys.exit(1)

    _reset_vmlx_cache_if_requested(sys.argv[2:])

    if _PHOENIX_OK:
        try:
            init_phoenix(project_name=f"lab-02-7-{variant}",
                         server_url="http://127.0.0.1:6006")
            print(f"[{variant}] Phoenix tracing enabled "
                  f"(http://127.0.0.1:6006/projects)", flush=True)
        except Exception as e:                              # noqa: BLE001
            print(f"[{variant}] Phoenix init failed (continuing without): "
                  f"{type(e).__name__}: {str(e)[:80]}", flush=True)
            # Init failed — rebind phoenix_span to a no-op so per-Q calls
            # below don't crash with 'phoenix_span called before init_phoenix'.
            from contextlib import nullcontext
            globals()["phoenix_span"] = lambda label=None, attrs=None: nullcontext()

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

    # Load ground-truth for GT-judge (binary pass/fail). Falls back to
    # entity-recall judge when no GT entry exists for a question.
    gt_qs: dict[str, dict] = {}
    try:
        gt_map = load_ground_truth(_LAB_ROOT / "data" / "eval_ground_truth.json")
        gt_qs = {entry["q"]: entry for entry in gt_map.values()}
        print(f"[{variant}] GT-judge loaded: {len(gt_qs)} questions covered",
              flush=True)
    except FileNotFoundError as e:
        print(f"[{variant}] GT not loaded ({e}); entity-recall judge for all Qs",
              flush=True)

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
        # Entity-recall judge (legacy) — kept for backward comparability.
        try:
            judge, _ = score_llm_judge(q, ans, exp)
        except Exception:                                # noqa: BLE001
            judge = 0.0
        # GT-judge (PRIMARY) — binary pass/fail when GT entry exists.
        gt_pass: bool | None
        gt_rationale = ""
        if q in gt_qs:
            gt_entry = gt_qs[q]
            try:
                gt_pass, gt_rationale = score_against_ground_truth(
                    client=omlx, model=os.getenv("MODEL_SONNET", ""),
                    question=q, gt_answer=gt_entry["gt_answer"],
                    pass_criteria=gt_entry["pass_criteria"],
                    candidate_answer=ans,
                )
            except Exception as e:                       # noqa: BLE001
                gt_pass = False
                gt_rationale = f"GT-judge error: {type(e).__name__}: {e}"
        else:
            gt_pass = None
        rows.append({"variant": variant, "q": q, "type": ty,
                     "answer": ans, "sub": sub, "judge": judge,
                     "gt_pass": gt_pass, "gt_rationale": gt_rationale[:200],
                     "lat": lat, "iters": iters, "tools": tools})
        gt_str = "PASS" if gt_pass is True else ("FAIL" if gt_pass is False else "—")
        print(f"  [{variant}][{ty[:14]:14s}] {q[:50]:50s} GT={gt_str} judge={judge:.2f} sub={sub:.2f} lat={lat:.1f}s iters={iters}",
              flush=True)

    j = sum(r["judge"] for r in rows) / len(rows)
    s = sum(r["sub"] for r in rows) / len(rows)
    L = sum(r["lat"] for r in rows) / len(rows)
    # GT pass rate (only over Qs with GT)
    gt_evaluated = [r for r in rows if r.get("gt_pass") is not None]
    gt_pass_count = sum(1 for r in gt_evaluated if r["gt_pass"])
    gt_pass_rate = gt_pass_count / max(1, len(gt_evaluated))
    by_cat = defaultdict(list)
    by_cat_gt: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        by_cat[r["type"]].append(r["judge"])
        if r.get("gt_pass") is not None:
            by_cat_gt[r["type"]].append(bool(r["gt_pass"]))
    cat = {k: sum(v) / len(v) for k, v in by_cat.items()}
    cat_gt = {k: sum(1 for x in v if x) / len(v) for k, v in by_cat_gt.items() if v}
    summary = {"variant": variant, "agg_judge": j, "agg_sub": s, "agg_lat": L,
               "agg_gt_pass_rate": gt_pass_rate,
               "gt_evaluated": len(gt_evaluated),
               "gt_pass_count": gt_pass_count,
               "per_cat": cat, "per_cat_gt": cat_gt, "rows": rows}
    out_path = _LAB_ROOT / "results" / f"ab_{variant}.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[{variant}] aggregate (PRIMARY): GT pass_rate={gt_pass_rate:.3f} "
          f"({gt_pass_count}/{len(gt_evaluated)})", flush=True)
    print(f"[{variant}] aggregate (legacy):  entity_judge={j:.3f} substr={s:.3f} "
          f"lat={L:.1f}s", flush=True)
    print(f"[{variant}] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
