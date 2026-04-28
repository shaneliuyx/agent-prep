# lab-02b — Production Library Refactor

This lab takes the **same retrieval / rerank / eval tasks** from `lab-02-rerank-compress` and re-implements them using three production-grade Python libraries:

| Library | What it abstracts | Script in this lab |
|---|---|---|
| [`langchain-qdrant`](https://python.langchain.com/docs/integrations/vectorstores/qdrant/) | Vector-store wrapper over Qdrant — `as_retriever()`, `similarity_search()`, integration glue for the LangChain ecosystem | `src/00_hybrid_via_langchain.py` |
| [`rerankers`](https://github.com/AnswerDotAI/rerankers) | Unified cross-encoder API across BGE / ColBERT / Cohere / Jina / etc. — `Reranker(model).rank(query, docs)` | `src/01_rerank_via_rerankers.py` |
| [`ranx`](https://github.com/AmenRa/ranx) | IR metrics library — `evaluate(qrels, run, ["recall@10", "ndcg@10", "mrr@10"])` | `src/02_eval_via_ranx.py` |

## Why this lab exists

`lab-02` (the primitives lab) teaches you *what* hybrid retrieval, two-stage rerank, and IR metrics actually are — by writing them from scratch.

`lab-02b` (this lab) teaches you *which production libraries you'd actually reach for*, what they give you, and what they take away.

**Read both.** The primitives lab is your portfolio + interview answer ("I built it from scratch so I know what each piece does"). This lab is your shipping answer ("In production I'd use these — here's the code").

## Prerequisites

- `lab-02-rerank-compress` already run end-to-end:
  - Qdrant collection `bge_m3_hybrid` populated with 10K MS MARCO docs
  - Qdrant collection `bge_m3_hnsw` populated (lab-01 baseline, used as upstream for rerank)
  - `data/queries.json`, `data/qrels.json`, `data/docs.jsonl` present

## Setup

```bash
cd ~/code/agent-prep/lab-02b-production-libs
source ../.venv/bin/activate
set -a; source ../.env; set +a
mkdir -p src results data

# Reuse lab-02's data (same MS MARCO 10K corpus + qrels)
cp ../lab-02-rerank-compress/data/*.json{,l} data/

# New deps for the production libraries
uv pip install -U \
    langchain-qdrant \
    langchain-huggingface \
    rerankers \
    ranx
```

## What each script demonstrates

### `00_hybrid_via_langchain.py` — vector-store abstraction

Wraps the **existing** `bge_m3_hnsw` collection (built by lab-01) as a LangChain `QdrantVectorStore`. Demonstrates:
- `from_existing_collection()` — connect without re-ingesting
- `as_retriever(search_kwargs={"k": 10})` — get a LangChain Retriever object usable in any chain/agent
- Result: same recall numbers as lab-01's `04_eval.py` (identical encoder + collection, just different API)

**Trade-off vs lab-02:** LangChain hides the `query_points()` call. Plus side: you get drop-in compatibility with chains, agents, and other LangChain components. Minus side: BGE-M3's *sparse* + hybrid mode aren't directly supported (LangChain's `RetrievalMode.HYBRID` uses fastembed/SPLADE sparse, not BGE-M3 sparse) — for true BGE-M3 hybrid you fall back to native client.

### `01_rerank_via_rerankers.py` — unified reranker API

Replaces lab-02's `CrossEncoder` directly with `rerankers.Reranker`. Demonstrates:
- `Reranker("BAAI/bge-reranker-v2-m3", model_type="cross-encoder")` — same model, simpler API
- `ranker.rank(query=q, docs=[...])` — returns a sorted `RankedResults` object
- Single API across BGE, ColBERT, Cohere, Jina, T5, RankZephyr — swap `Reranker(...)` argument, no other code changes

**Trade-off vs lab-02:** Trivially replaceable. The `rerankers` API doesn't take much away — just adds a uniform interface across vendors. Worth adopting in production from day one.

### `02_eval_via_ranx.py` — IR metrics library

Replaces lab-02's hand-rolled recall@K / MRR@K / nDCG@K math with one `ranx.evaluate()` call. Demonstrates:
- `Qrels({qid: {docid: relevance}})` — ground truth as a typed object
- `Run({qid: {docid: retrieval_score}})` — system output as a typed object
- `evaluate(qrels, run, ["recall@10", "mrr@10", "ndcg@10"])` — metrics in one call
- Bonus: `ranx.compare(...)` for paired statistical tests across systems

**Trade-off vs lab-02:** Saves ~30 lines of metric math, removes a class of off-by-one errors (rank starts at 1 not 0, IDCG denominator handling, etc.) but hides the formula. Use after you've internalized what nDCG actually computes.

## Expected results

All three scripts should produce **numbers within ε of lab-02's hand-rolled versions** when run on the same collections + queries. If they diverge by more than rounding error, something's wrong with the library configuration — not a real algorithmic difference.

The headline output is the same `recall@10 ≈ 0.993` (BGE-M3 dense on MS MARCO 10K) you got in lab-01.

## What this lab is NOT

- Not a pitch for adopting LangChain wholesale — for an agent stack at scale, the integration glue is real value; for a single-purpose retrieval pipeline, the dependency cost is high.
- Not a benchmark of the libraries vs each other — they're complementary, not competitive.
- Not the "right" answer — the right answer in production depends on your team's existing dependencies, your latency budget, and your customization needs. The labs together teach you to make that decision deliberately.
