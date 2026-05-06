"""Score the baseline pipeline with RAGAS, backed by local oMLX.

Run from project root:

    cd ~/code/agent-prep/lab-03-rag-eval
    set -a; source ../.env; set +a
    python src/02b_ragas_eval.py

v2 (2026-05-06): RAGAS judge embeddings now use shared/rag_hybrid autoconfig
to pick device + fp16 instead of hardcoded `device="mps"`. Same `bge-m3`
weights, but portable across cuda / mps / cpu hosts. The pipeline import
goes through the migrated 02_pipeline.py (autoconfig'd encoder + reranker).

Why this version uses the compatible legacy RAGAS metric classes:
- ragas.metrics.collections requires "modern" embeddings.
- RAGAS native HuggingfaceEmbeddings is abstract/broken in some versions.
- Local BGE-M3 works reliably through LangchainEmbeddingsWrapper.
"""

import json
import os
import sys
import warnings
from pathlib import Path

# Hide warnings that are unavoidable in this local RAGAS + BGE-M3 combination.
# The fully-modern path currently conflicts with local HuggingFace embeddings in
# this environment, so the compatible wrapper path is intentional.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="langchain.*")

from datasets import Dataset
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)

# Bootstrap shared/ for autoconfig probe (device / memory tier).
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "shared"))
from rag_hybrid import autoconfig  # noqa: E402

from src.script_wrap import load

pipeline = load("02_pipeline.py")
run_pipeline = pipeline.run_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing environment variable: {name}. "
            "Run: set -a; source ../.env; set +a"
        )
    return value


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    model_sonnet = require_env("MODEL_SONNET")
    omlx_base_url = require_env("OMLX_BASE_URL")
    omlx_api_key = require_env("OMLX_API_KEY")

    dev_path = PROJECT_ROOT / "data" / "dev_set.jsonl"
    results_path = PROJECT_ROOT / "results" / "ragas_baseline.json"
    debug_path = PROJECT_ROOT / "results" / "ragas_baseline_debug.jsonl"

    if not dev_path.exists():
        raise FileNotFoundError(f"Missing dev set: {dev_path}")

    llm = ChatOpenAI(
        model=model_sonnet,
        base_url=omlx_base_url,
        api_key=omlx_api_key,
        temperature=0.0,
    )
    ragas_llm = LangchainLLMWrapper(llm)

    # Probe device for the RAGAS judge embeddings. autoconfig picks mps / cuda /
    # cpu — replaces the previous hardcoded `device="mps"` which broke on cuda + cpu hosts.
    lc_emb = HuggingFaceEmbeddings(
        model_name=os.path.expanduser("~/models/bge-m3"),
        model_kwargs={"device": autoconfig.probe_system().device},
        encode_kwargs={"normalize_embeddings": True},
    )
    ragas_emb = LangchainEmbeddingsWrapper(lc_emb)

    dev = load_jsonl(dev_path)
    rows = []
    debug_rows = []

    for i, q in enumerate(dev):
        question = q["question"]
        ground_truth = q["short_answer"]

        print(f"  {i + 1}/{len(dev)}: {question[:80]}")
        out = run_pipeline(question)

        rows.append(
            {
                "question": question,
                "answer": out["answer"],
                "contexts": out["contexts"],
                "ground_truth": ground_truth,
            }
        )
        debug_rows.append(
            {
                "source_doc_id": q.get("source_doc_id"),
                "question": question,
                "short_answer": ground_truth,
                "answer": out["answer"],
                "context_ids": out.get("context_ids", []),
                "contexts": out["contexts"],
            }
        )

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with debug_path.open("w", encoding="utf-8") as f:
        for row in debug_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    ds = Dataset.from_list(rows)

    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb),
        ContextPrecision(llm=ragas_llm),
        ContextRecall(llm=ragas_llm),
    ]

    scores = evaluate(
        ds,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_emb,
        run_config=RunConfig(
            timeout=300,
            max_retries=3,
            max_workers=2,
        ),
    )

    print("\n=== BASELINE ===")
    print(scores)

    results_path.write_text(
        json.dumps(scores.to_pandas().to_dict(), indent=2, default=str),
        encoding="utf-8",
    )

    print(f"\nwrote {results_path}")
    print(f"wrote {debug_path}")


if __name__ == "__main__":
    main()
