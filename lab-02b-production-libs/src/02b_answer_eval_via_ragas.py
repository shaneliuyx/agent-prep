"""1:1 with lab-02/src/02b_answer_eval.py — but via RAGAS (RAG eval library).

Same task: judge whether retrieved-context-driven answers are faithful + relevant.
Different code: RAGAS provides typed metrics (faithfulness, answer_relevancy,
context_precision, context_recall) and a one-call evaluate() that returns scores
per-query AND aggregates.

Industry-standard library for RAG evaluation. Used at scale by teams shipping RAG
products. Replaces the hand-rolled LLM-as-judge prompt loop in lab-02 with typed
metrics that are well-tested + reproducible.
"""
import json, os
from pathlib import Path
from datasets import Dataset
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

HOME = os.path.expanduser("~")

# RAGAS needs an LLM (for judge prompts) + an embedding model (for relevancy scoring)
llm = ChatOpenAI(
    model=os.getenv("MODEL_HAIKU", "gpt-oss-20b-MXFP4-Q8"),
    base_url=os.getenv("OMLX_BASE_URL"),
    api_key=os.getenv("OMLX_API_KEY"),
    temperature=0.0,
)
embedding = HuggingFaceEmbeddings(
    model_name=f"{HOME}/models/bge-m3",
    model_kwargs={"device": "mps", "trust_remote_code": True},
)

# --- Build the eval dataset ---
# RAGAS expects: question, answer, contexts (list[str]), [ground_truth optional]
# Reads the 50 sample answers produced by lab-02/02_compress.py + a real answer pass.
# For a self-contained demo, builds 5 toy examples — replace with real outputs in practice.
toy_examples = [
    {
        "question":     "What is the capital of France?",
        "answer":       "The capital of France is Paris.",
        "contexts":     ["Paris is the capital and most populous city of France."],
        "ground_truth": "Paris",
    },
    {
        "question":     "Who wrote Hamlet?",
        "answer":       "Shakespeare wrote Hamlet around 1600.",
        "contexts":     ["Hamlet is a tragedy written by William Shakespeare sometime between 1599 and 1601."],
        "ground_truth": "William Shakespeare",
    },
    # Add more — or load from results/answer_eval_data.json if it exists
]

eval_data = Dataset.from_list(toy_examples)
print(f"evaluating {len(eval_data)} examples via RAGAS (faithfulness + answer_relevancy + context_precision)")

# --- Run RAGAS — one call, multiple metrics ---
result = evaluate(
    dataset=eval_data,
    metrics=[faithfulness, answer_relevancy, context_precision],
    llm=llm,
    embeddings=embedding,
)

# Per-query scores + aggregates
print("\n--- Per-query scores ---")
print(result.to_pandas())

print("\n--- Aggregate ---")
agg = {
    "library": "ragas",
    "faithfulness":      float(result["faithfulness"]),
    "answer_relevancy":  float(result["answer_relevancy"]),
    "context_precision": float(result["context_precision"]),
}
print(json.dumps(agg, indent=2))

Path("results").mkdir(exist_ok=True)
Path("results/ragas_metrics.json").write_text(json.dumps(agg, indent=2))

print("\nCompare to lab-02/src/02b_answer_eval.py — RAGAS replaces the hand-rolled LLM-as-judge prompt.")
print("Bonus: RAGAS gives 4+ metrics (faithfulness, answer_relevancy, context_precision/recall) in one call.")
print("\nIn production: replace toy_examples with rows from your retrieve+answer pipeline output.")
