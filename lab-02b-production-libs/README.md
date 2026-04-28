# lab-02b — Production Library Refactor

Re-implements every lab-02 script using production Python libraries instead of primitives. **Strict 1:1 pairing** — each lab-02 script has a counterpart here that does the same task with `langchain-qdrant`, `rerankers`, `ragas`, or `ranx`.

## 1:1 mapping with lab-02

| lab-02 (primitives) | lab-02b (libraries) | Library | Numbers match lab-02? |
|---|---|---|---|
| `00_hybrid_ingest.py` | `00_hybrid_ingest_via_langchain.py` | langchain-qdrant + FastEmbedSparse | ⚠️ different sparse encoder (SPLADE++ vs BGE-M3 sparse) |
| `00_hybrid_eval.py` | `00_hybrid_eval_via_langchain.py` | langchain-qdrant retrievers (3 modes) | dense ✅ identical · sparse/hybrid ⚠️ differ ~3-7pp |
| `00_fiqa_eval.py` | `00_fiqa_eval_via_langchain.py` | langchain-qdrant + datasets | dense ✅ identical · sparse/hybrid ⚠️ differ |
| `01_rerank.py` | `01_rerank_via_rerankers.py` | rerankers | ✅ identical (same model, wrapper API) |
| `01b_latency.py` | `01b_latency_via_rerankers.py` | rerankers | ✅ within ±5% (wrapper overhead) |
| `02_compress.py` | `02_compress_via_langchain.py` | LangChain `ContextualCompressionRetriever + LLMChainExtractor` | ✅ comparable ratio (0.25-0.50 range) |
| `02b_answer_eval.py` | `02b_answer_eval_via_ragas.py` | RAGAS — `faithfulness`, `answer_relevancy`, `context_precision` | ➕ MORE metrics than lab-02's hand-rolled judge |
| `03_chunk_sweep.py` | `03_chunk_sweep_via_langchain.py` | LangChain `RecursiveCharacterTextSplitter` + langchain-qdrant | ⚠️ slightly different boundaries (LangChain handles edge cases differently) |
| `03b_sweep_eval.py` | `03b_sweep_eval_via_ranx.py` | ranx — typed Qrels + Run + bonus paired stat tests via `compare()` | ✅ identical recall@5 · ➕ adds significance tests |

**Read this table as the lab's thesis:** every primitive task has a library counterpart. Some library versions are clean drop-in wins (`rerankers`, `ranx`); others have intentional deltas (`langchain-qdrant` hybrid uses SPLADE not BGE-M3 sparse). The deltas ARE the lesson — they show you what each library hides + what it costs.

## Why this lab exists

`lab-02` (primitives) teaches you *what* hybrid retrieval, two-stage rerank, context compression, and IR metrics actually are — by writing them from scratch.

`lab-02b` (this lab) teaches you *which production libraries you'd actually reach for*, what they give you, and what they take away.

**Read both.** The primitives lab is your portfolio + interview answer ("I built it from scratch so I know what each piece does"). This lab is your shipping answer ("In production I'd use these — here's the code, here's the trade-off table").

## Prerequisites

- `lab-02-rerank-compress` ideally already run end-to-end (so you have lab-02 numbers to compare against)
- Qdrant running on `:6333`
- For most scripts: lab-01's `bge_m3_hnsw` collection populated (used as upstream for rerank + compression scripts)

## Setup

```bash
cd ~/code/agent-prep/lab-02b-production-libs
source ../.venv/bin/activate
set -a; source ../.env; set +a
mkdir -p src results data

# Reuse lab-02's data (same MS MARCO 10K corpus + qrels)
cp ../lab-02-rerank-compress/data/*.json{,l} data/

# All deps for the production-library scripts
uv pip install -U \
    langchain-qdrant \
    langchain-huggingface \
    langchain-openai \
    langchain-text-splitters \
    "langchain[community]" \
    rerankers \
    ranx \
    ragas \
    datasets \
    fastembed
```

## Run order

The scripts have dependencies (some build collections that others read):

```bash
# Phase 1 — Hybrid retrieval
python src/00_hybrid_ingest_via_langchain.py    # builds lc_hybrid collection (~6-8 min)
python src/00_hybrid_eval_via_langchain.py      # evaluates dense/sparse/hybrid (~3-5 min)
python src/00_fiqa_eval_via_langchain.py        # ingests + evals BEIR-FiQA (~15-20 min)

# Phase 2 — Rerank
python src/01_rerank_via_rerankers.py           # two-stage retrieval (~3-5 min)
python src/01b_latency_via_rerankers.py         # latency benchmark (~2 min)

# Phase 3 — Compression + answer eval
python src/02_compress_via_langchain.py         # 50-query compression (~3-5 min)
python src/02b_answer_eval_via_ragas.py         # RAGAS metrics (~2-5 min, depends on judge model)

# Phase 4 — Chunking sweep
python src/03_chunk_sweep_via_langchain.py      # builds 9 lc_sweep_* collections (~25-30 min)
python src/03b_sweep_eval_via_ranx.py           # evaluates + paired stat tests (~5 min)
```

## Where each library wins

| Library | Strongest 1:1 win | What it hides |
|---|---|---|
| `langchain-qdrant` | Reduces collection-create + upsert + query boilerplate by ~80%; gives you a `Retriever` interface usable in any chain or agent | Collection schema details (named vector configs, sparse encoder choice) |
| `rerankers` | Single API across BGE / ColBERT / Cohere / Jina / T5 / RankZephyr — swap one string to A/B vendors | Almost nothing — adopt from day one in production |
| `LLMChainExtractor` (LangChain) | Composes retrieval + compression as one `Retriever` object; can plug into any LangChain chain | Custom prompt control (use `from_llm(llm, prompt=...)` if you need it) |
| `ragas` | Typed metrics with proven LLM-as-judge implementations; gets you 4+ metrics in one `evaluate()` call | The judge prompts (RAGAS picks battle-tested versions) |
| `ranx` | One-call IR metrics + `compare()` for paired stat tests; eliminates off-by-one errors in metric math | Hides the recall/MRR/nDCG formulas (which is also the learning content) |

## What this lab is NOT

- **Not a pitch for adopting any single library wholesale.** The right stack depends on your team's existing dependencies, latency budget, and customization needs.
- **Not a benchmark of libraries vs each other.** They're complementary — `langchain-qdrant` for vector ops, `rerankers` for cross-encoders, `ranx` for metrics, `ragas` for RAG eval.
- **Not the "right" answer.** The correct production choice changes as your scope grows. The lab teaches you to make that choice deliberately, with the trade-off table in your head.

## What you should learn from finishing both labs

After running lab-02 + lab-02b end-to-end, you should be able to defend any of these in an interview:

1. **"Why didn't you just use LangChain?"** — Because lab-02 was about understanding what `langchain-qdrant.from_existing_collection().as_retriever()` actually does. Now I know, and I'd use it for the integration glue in production.
2. **"How do you know if reranking actually helps your retrieval?"** — Run lab-02b's `03b_sweep_eval_via_ranx.py` style: paired statistical comparison via `ranx.compare(stat_test="fisher")`. Eyeballing recall deltas is a way to fool yourself.
3. **"What's the trade-off in 'hybrid retrieval'?"** — Hybrid via BGE-M3 (lab-02): one model, learned dense + sparse. Hybrid via LangChain default (lab-02b): two models, dense + SPLADE. Different recall on hard queries; same RRF fusion logic.
4. **"How would you evaluate a RAG system?"** — RAGAS for the answer-quality side (`faithfulness`, `answer_relevancy`), ranx for the retrieval side (`recall@K`, `nDCG@K`), paired statistical tests when comparing variants.
