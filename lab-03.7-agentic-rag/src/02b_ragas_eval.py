"""Score BOTH arms (single-pass + agentic) from results/comparison_raw.json with RAGAS,
backed by LOCAL models: oMLX as the judge LLM + local BGE-M3 as the judge embeddings.

WHY the local wiring matters: RAGAS defaults to OpenAI for both the judge LLM and the
embeddings. On this local-first stack you MUST wrap oMLX + a local embedder, or every
metric silently tries to call api.openai.com and fails. This mirrors the Week-3 RAGAS
harness (lab-03-rag-eval/src/02b_ragas_eval.py).

PRE-REQ: the agentic arm needs its retrieved `contexts` (context_precision / recall /
faithfulness all read them). Re-run 02_comparison_harness.py after the context-capture
patch so each agentic row carries `contexts`.

    cd ~/code/agent-prep/lab-03.7-agentic-rag
    env -u VIRTUAL_ENV uv pip install -U ragas datasets langchain-huggingface
    env -u VIRTUAL_ENV uv pip uninstall xai_sdk   # instructor eager-imports it; breaks on protobuf>=7
    uv run python src/02_comparison_harness.py     # re-run first if agentic.contexts missing
    uv run python src/02b_ragas_eval.py
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="langchain.*")

from dotenv import load_dotenv
from datasets import Dataset
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# --- compat shim (ragas 0.4.x + langchain-community >=0.3) -------------------
# ragas 0.4.x unconditionally imports the deprecated Vertex classes
# `langchain_community.chat_models.vertexai.ChatVertexAI` and
# `langchain_community.llms.VertexAI`, both REMOVED in langchain-community >=0.3.
# We never use Vertex (we configure ChatOpenAI), so stub them before importing
# ragas. Delete this block once ragas stops the unconditional import.
import types as _types  # noqa: E402
_vx = _types.ModuleType("langchain_community.chat_models.vertexai")
_vx.ChatVertexAI = type("ChatVertexAI", (), {})
sys.modules.setdefault("langchain_community.chat_models.vertexai", _vx)
try:
    import langchain_community.llms as _lcl  # noqa: E402
    if not hasattr(_lcl, "VertexAI"):
        _lcl.VertexAI = type("VertexAI", (), {})
except Exception:
    pass
# ----------------------------------------------------------------------------

from ragas import evaluate  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
from ragas.run_config import RunConfig

load_dotenv(os.path.expanduser("~/code/agent-prep/lab-03.7-agentic-rag/.env"))
sys.path.insert(0, os.path.expanduser("~/code/agent-prep/shared"))
from rag_hybrid import autoconfig  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
# --raw points at the harness output to score; --out where to write scores. Defaults keep the
# canonical flow; pass a custom --raw to score a harder/larger dev set's artifact (§2.6).
import argparse  # noqa: E402
_p = argparse.ArgumentParser(description="RAGAS-score single-pass + both agentic arms")
_p.add_argument("--raw", default=str(ROOT / "results" / "comparison_raw.json"))
_p.add_argument("--out", default=str(ROOT / "results" / "ragas_scores.json"))
_args, _ = _p.parse_known_args()
RAW = Path(os.path.expanduser(_args.raw))
OUT = Path(os.path.expanduser(_args.out))


def ragas_backends():
    """One LOCAL judge LLM (oMLX) + one LOCAL embedder (BGE-M3). RAGAS would hit OpenAI
    for both by default; wrapping them is the whole local-first adaptation."""
    llm = ChatOpenAI(model=os.getenv("MODEL_SONNET"),
                     base_url=os.getenv("OMLX_BASE_URL", "http://localhost:8000/v1"),
                     api_key=os.getenv("OMLX_API_KEY", "not-needed"), temperature=0.0)
    emb = HuggingFaceEmbeddings(
        model_name=os.path.expanduser("~/models/bge-m3"),                 # same BGE-M3 as retrieval
        model_kwargs={"device": autoconfig.probe_system().device},        # mps / cuda / cpu
        encode_kwargs={"normalize_embeddings": True})
    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(emb)


def dataset_for(rows: list[dict], arm: str) -> Dataset:
    """RAGAS expects {question, answer, contexts, ground_truth} per row."""
    return Dataset.from_list([{
        "question": r["question"],
        "answer": r[arm]["answer"],
        "contexts": r[arm].get("contexts", []),
        "ground_truth": r["ground_truth"],
    } for r in rows])


# single-pass + BOTH agentic arms (canonical skip-allowed vs the structural fix, §2.5.1)
ARMS = ("single_pass", "agentic_canonical", "agentic_structural")


def main() -> None:
    rows = json.loads(RAW.read_text())
    for arm in ("agentic_canonical", "agentic_structural"):
        if not rows or arm not in rows[0]:
            sys.exit(f"{arm} missing in comparison_raw.json — re-run 02_comparison_harness.py "
                     "(it now writes single_pass + both agentic arms, §2.4) first.")

    ragas_llm, ragas_emb = ragas_backends()
    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb),
        ContextPrecision(llm=ragas_llm),
        ContextRecall(llm=ragas_llm),
    ]
    rc = RunConfig(timeout=300, max_retries=3, max_workers=2)

    out: dict[str, dict[str, float]] = {}
    for arm in ARMS:
        print(f"\n=== scoring {arm} ({len(rows)} rows) ===")
        scores = evaluate(dataset_for(rows, arm), metrics=metrics,
                          llm=ragas_llm, embeddings=ragas_emb, run_config=rc)
        df = scores.to_pandas()
        out[arm] = {c: float(df[c].mean()) for c in df.columns if df[c].dtype.kind in "fi"}

    OUT.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print("\n=== single-pass vs canonical (skip-allowed) vs structural (fixed) ===")
    print(f"  {'metric':18}{'single-pass':>13}{'canonical':>12}{'structural':>12}")
    for m in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        sp = out["single_pass"].get(m, float("nan"))
        ca = out["agentic_canonical"].get(m, float("nan"))
        st = out["agentic_structural"].get(m, float("nan"))
        print(f"  {m:18}{sp:>13.3f}{ca:>12.3f}{st:>12.3f}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
