"""Score the PRE-MIGRATION multi-query fusion pipeline with RAGAS.

This is a one-off script for capturing Run "PreM-MQ" — the pre-migration
multi-query baseline that was missed before the shared/rag_hybrid
migration shipped. After this run, the post-migration `04b_ragas_
multiquery.py` numbers have a within-±0.02 anchor to validate against
(same contract as Run 4 → Run 5 for baseline, Run pre/post for HyDE).

Differences vs `04b_ragas_multiquery.py`:
  - loads `04_multiquery_premigration.py` (not `04_multiquery.py`)
  - writes `results/ragas_multiquery_premigration.json` +
    `results/ragas_multiquery_premigration_debug.jsonl`
  - debug rows DO NOT include `rewrites` field (pre-migration return
    shape didn't expose them)
  - prints `=== MULTI-QUERY (PRE-MIGRATION) ===`

Everything else identical: legacy RAGAS metric path, autoconfig device
probe for judge embeddings (eval-side, not pipeline-side — autoconfig
for judge is a separate axis from the migration), same RunConfig.

Run from project root:

    cd ~/code/agent-prep/lab-03-rag-eval
    set -a; source ../.env; set +a
    python src/04b_ragas_multiquery_premigration.py 2>&1 | tee /tmp/04b_pre.log

After this completes, compare against the post-migration multi-query
run output (`results/ragas_multiquery.json`). If all 4 metrics within
±0.02, multi-query migration is non-regressive — same conclusion shape
as Run 4 → Run 5 for baseline.

This script is dead code after the run. Fine to delete or keep as
historical reproduction artifact.
"""
import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="langchain.*")

from datasets import Dataset
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
from ragas.run_config import RunConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Bootstrap shared/ for autoconfig probe (eval-side judge embeddings —
# this is fine; autoconfig for the judge is a separate axis from the
# pipeline migration we're measuring).
_REPO_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(_REPO_ROOT / "shared"))
from rag_hybrid import autoconfig  # noqa: E402

from src.script_wrap import load  # noqa: E402

# Load pre-migration multi-query module (standalone — no shared lib deps).
mq = load("04_multiquery_premigration.py")
run_pipeline = mq.run_pipeline_mq


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing environment variable: {name}. Run: set -a; source ../.env; set +a")
    return value


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    model_sonnet = require_env("MODEL_SONNET")
    omlx_base_url = require_env("OMLX_BASE_URL")
    omlx_api_key = require_env("OMLX_API_KEY")

    dev_path = PROJECT_ROOT / "data" / "dev_set.jsonl"
    results_path = PROJECT_ROOT / "results" / "ragas_multiquery_premigration.json"
    debug_path = PROJECT_ROOT / "results" / "ragas_multiquery_premigration_debug.jsonl"

    llm = ChatOpenAI(
        model=model_sonnet,
        base_url=omlx_base_url,
        api_key=omlx_api_key,
        temperature=0.0,
    )
    ragas_llm = LangchainLLMWrapper(llm)

    # autoconfig for judge embeddings is fine; not part of pipeline migration.
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
        print(f"  {i + 1}/{len(dev)}: {question[:80]}")
        out = run_pipeline(question)

        rows.append({
            "question": question,
            "answer": out["answer"],
            "contexts": out["contexts"],
            "ground_truth": q["short_answer"],
        })
        debug_rows.append({
            "source_doc_id": q.get("source_doc_id"),
            "question": question,
            "short_answer": q["short_answer"],
            "answer": out["answer"],
            "context_ids": out.get("context_ids", []),
            # Pre-migration: no `rewrites` field in output. Captured as
            # missing in debug too — matches pre-migration return shape.
            "contexts": out["contexts"],
        })

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
        run_config=RunConfig(timeout=300, max_retries=3, max_workers=2),
    )

    print("\n=== MULTI-QUERY (PRE-MIGRATION) ===")
    print(scores)

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(scores.to_pandas().to_dict(), indent=2, default=str), encoding="utf-8")
    debug_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in debug_rows), encoding="utf-8")

    print(f"\nwrote {results_path}")
    print(f"wrote {debug_path}")


if __name__ == "__main__":
    main()
