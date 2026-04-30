# Lab 02 — Rerank & Context Compression — RESULTS

**Date:** 2026-04-29
**Hardware:** Apple M5 Pro (48 GB unified memory, Metal 4)
**Corpus / tooling:** MS MARCO 10K (10,000 docs / 6,980 dev queries) + BEIR-FiQA-2018 (57,638 docs / 648 test queries)
**Status:** Phase 1 (hybrid retrieval), Phase 2.2 (reranker), Phase 2.3 (latency), Phase 3 (compression + LLM-as-judge), and Phase 4 (chunking sweep) complete

This document synthesizes today's empirical findings. For the full investigation arc — including disproven hypotheses, source-code analysis of failed framework ports, and the moving parts behind each measurement — see `Week 2 - Rerank and Context Compression.md` §1.3.1, §1.4.1, §2.2.1-§2.2.5.

---

## 1. Hybrid Retrieval Lift (MS MARCO 10K — saturated benchmark)

| Mode | Recall@10 | MRR@10 | nDCG@10 | Wall (s) |
|------|-----------|--------|---------|----------|
| Dense (Week 1 baseline) | 0.9933 | 0.9556 | 0.9637 | 205 |
| Sparse | 0.7754 | 0.6587 | 0.6831 | 182 |
| Hybrid (RRF, k=60) | 0.9923 | 0.8768 | 0.9046 | 215 |

- Hybrid lift over dense (recall@10): **−0.001** (essentially tied)
- Hybrid MRR drop: **−7.9pp** (fusion demotes dense's confident rank-1 picks)
- **Caveat:** ceiling effect at recall ≥ 0.99 — see §2 for the meaningful comparison

### DBSF fusion variant tested

| Mode | Recall@10 | MRR@10 | nDCG@10 |
|------|-----------|--------|---------|
| Hybrid (RRF) | 0.9923 | 0.8768 | 0.9046 |
| Hybrid (DBSF) | 0.9872 | 0.8913 | 0.9130 |

DBSF gains 1.5 MRR points but loses 0.5 recall — score-aware fusion preserves dense's confidence at the cost of marginal-candidate breadth. **No fusion algorithm closed the gap to dense alone on this corpus.**

### Recall@100 — does hybrid earn its keep as a candidate generator?

| Mode | Recall@100 | MRR@100 | nDCG@100 |
|------|-----------|---------|----------|
| Dense | 0.9993 | 0.9559 | 0.9653 |
| Sparse | 0.8665 | 0.6624 | 0.7028 |
| Hybrid | 0.9987 | 0.8781 | 0.9069 |

**Hybrid recall@100 is *lower* than dense recall@100 by 4 queries.** On MS MARCO 10K, hybrid doesn't even win as a candidate generator for downstream reranking — the dense baseline is too saturated.

### Cross-lab follow-up: SPLADE++ flips the hybrid story

Lab-02b ran the same evaluation against the same MS MARCO 6,980-query dev set, but swapped BGE-M3's free lexical output for **SPLADE++** sparse via `fastembed`. Result:

| Mode | Lab-02 sparse encoder | Lab-02b sparse encoder | Recall@10 Δ |
|------|----------------------|------------------------|-------------|
| Sparse | BGE-M3 lexical (0.7754) | SPLADE++ (**0.9928**) | **+21.7 pp** |
| Hybrid (RRF) | BGE-M3 hybrid (0.9923 — *loses* to dense) | SPLADE++ hybrid (**0.9977** — *beats* dense by 0.5 pp) | **+0.5 pp; first hybrid win** |

**The reframing:** lab-02's "hybrid is corpus-dependent" finding is half-right. The corpus matters, but **the sparse encoder matters more.** With a learned-sparse retriever (SPLADE++) instead of a dense-encoder side-channel (BGE-M3 lexical), RRF averages two confident voters instead of diluting one with noise — and hybrid finally produces the lift the architecture promised. See runbook §7.6 for the full comparison.

---

## 2. BEIR-FiQA-2018 Cross-Benchmark (ceiling-free)

| Mode | Recall@10 | MRR@10 | nDCG@10 | Wall (s) |
|------|-----------|--------|---------|----------|
| Dense | 0.6775 | 0.4989 | 0.4077 | 21 |
| Sparse | 0.5170 | 0.3288 | 0.2695 | 16 |
| Hybrid (RRF) | 0.6821 | 0.4721 | 0.3911 | 21 |

- **Hybrid finally beats dense on recall** (+0.5pp) — first time across all measurements
- MRR drop persists (−2.7pp) but is much smaller than MS MARCO (−7.9pp) because dense isn't dominating as hard
- Statistical reading: differences > ~3pp at this recall scale (n=648) are meaningful

### Sanity-check vs published BEIR numbers

| System | nDCG@10 (FiQA) |
|--------|----------------|
| BM25 | ~0.236 |
| Our sparse (BGE-M3 lexical) | 0.270 |
| Our dense (BGE-M3) | **0.408** |
| Published BGE-M3 / BGE-large baseline | ~0.408 |
| Cross-encoder rerank target (Phase 2 ext.) | ~0.45-0.50 |

**Our dense nDCG@10 = 0.408 matches the published baseline exactly**, confirming the eval pipeline is correct.

### HNSW corpus-size invariance (operational signal)

| Eval | Corpus size | Queries | Wall (s) | ms/query |
|------|-------------|---------|----------|----------|
| MS MARCO | 10K | 6,980 | 205 | ~29 |
| FiQA | 57.6K | 648 | 21 | ~32 |

5.7× corpus growth → essentially constant per-query latency. HNSW's `O(log n)` traversal working as designed; if it had been brute-force, FiQA would have taken ~120 s instead of 21.

---

## 3. Reranker Lift (PyTorch+MPS+fp16, 6,980 queries)

Best-measured config: `RERANK_GROUP_QUERIES=32, RERANK_BATCH=128, fp16` (validated across 5 runs).

| Pipeline | Recall@5 | nDCG@5 | Per-query latency | Total wall |
|----------|----------|--------|-------------------|------------|
| Dense top-5 (Week 1 baseline) | 0.9881 | 0.9616 | ~5 ms | (encode+search) |
| Dense top-50 → rerank top-5 (PyTorch+MPS+fp16) | **0.9948** | **0.9767** | ~115 ms (cross-encoder) | **27.6 min** |

- Recall@5 lift: **+0.7pp** (reranker recovered ~47 queries from top-50 candidate pool that dense had below rank-5)
- nDCG@5 lift: **+1.5pp** (more meaningful — reranker reorders within top-50 to put gold higher)
- Latency: well under the 200 ms p95 reranker budget on M5 Pro

### Optimization journey (5 measured runs)

| Run | precision | group | batch | sub-batches/call | rerank | total | ms/pair |
|-----|-----------|-------|-------|------------------|--------|-------|---------|
| #1 | fp32 | 32 | 128 | 12.5 | 4,699.5 s | 78.6 min | 13.5 |
| #2 | fp16 | 32 | 128 | 12.5 | **1,642.3 s** | **27.6 min** ← best | **4.7** |
| #3 | fp16 | 1 | 256 | 1 | 2,033.6 s | 34.2 min | 5.8 |
| #4 | fp16 | 8 | 256 | 1.5 | 2,020.1 s | 33.9 min | 5.8 |
| #5 | fp16 | 32 | 256 | 6.25 | 1,890.7 s | 31.8 min | 5.4 |

**Three theories tested across runs #1-#5:**

1. **fp16 conversion** (§2.2.1) — ✓ CONFIRMED: 2.86× speedup, recall@5 identical, nDCG@5 within 1.6 × 10⁻⁴
2. **Cross-query padding penalty** (§2.2.2) — ✗ DISPROVEN: setting `group=1` regressed by 24%
3. **Per-call tokenization overhead** (§2.2.2 hypothesis revision) — ✗ DISPROVEN: setting `group=8` saved only 14 seconds, not the predicted 5+ min
4. **GPU pipelining** (§2.2.3 → §2.2.4) — ✓ CONFIRMED: monotonic relationship between sub-batches/call and ms/pair

### The validated mental model

```
ms/pair vs sub-batches per predict() call:
  ≤2 sub-batches  → 5.8 ms/pair   (group=1 batch=256, group=8 batch=256)
  ~6 sub-batches  → 5.4 ms/pair   (group=32 batch=256)
  ~12 sub-batches → 4.7 ms/pair   (group=32 batch=128) ← validated optimum
```

**The mechanism:** MPS pipelines kernel launches with execution. Below ~10 sub-batches per call, kernel-launch latency is exposed (each kernel waits for the previous). At 12+ sub-batches, MPS overlaps launches with execution, hiding latency.

**Counter-intuitive finding:** doubling `RERANK_BATCH` from 128 to 256 *regressed* by 15% because it halved the sub-batch count. Bigger isn't always better when per-kernel utilization is already saturated; the lever that matters is *how many in-flight kernels the scheduler can pipeline*.

---

## 3.1 Per-query reranker latency (production-shape measurements)

Three configurations benchmarked, isolating each optimization lever:

| Run | precision | batch_size | Per-query latency | Δ vs spec |
|-----|-----------|------------|-------------------|-----------|
| #1 | fp32 | 32 (spec default) | 112.1 ms | 1.0× |
| #2 | fp32 | 128 | 103.0 ms | 0.92× |
| #3 | **fp16** | **128** | **37.8 ms** | **0.34×** ← shipping config |

**fp16 + batch=128 delivers a 2.97× per-query speedup** over the spec defaults. 37.8 ms is **5.3× under the 200 ms p95 budget** for synchronous RAG.

| Stage | Latency at shipping config | Budget (p95 SLA) | Headroom |
|-------|---------------------------|------------------|----------|
| Encode query (BGE-M3) | ~5 ms | < 10 ms | 5 ms |
| Qdrant ANN top-50 | ~10 ms | < 20 ms | 10 ms |
| **Cross-encoder rerank (fp16, batch=128)** | **37.8 ms** | < 100 ms | **62 ms** |
| Pre-LLM total | ~53 ms | < 200 ms p95 | **147 ms** |

### Lever decomposition

```
Spec default (fp32, batch=32):    112.1 ms
                                  ─── batch=32 → 128: -9.1 ms (-8%)
fp32 + batch=128:                 103.0 ms
                                  ─── fp32 → fp16:   -65.2 ms (-63%)
fp16 + batch=128:                  37.8 ms ← shipping config
```

- **fp16 (§2.2.1 lever) is dominant** — 63% of the speedup. Transfers cleanly from offline-eval throughput to per-query latency.
- **batch=128 (production-shape lever) is small** — 8% gain. Reduces kernel launches when only one query is in flight.

### Why batch=128 helps in production but batch=256 didn't help in offline eval

| Shape | Pairs/call | At batch=128 | At batch=256 |
|-------|-----------|--------------|--------------|
| Production (1 query × 50 pairs) | 50 | 1 sub-batch ✓ | 1 sub-batch (no change) |
| Offline eval (32 queries × 50 = 1,600) | 1,600 | 12.5 sub-batches ✓ | 6.25 sub-batches (regressed) |

Same MPS kernel-launch mechanism, opposite optimization direction:
- **Pairs-poor (production)** → reduce kernel-launch count → bigger batch wins
- **Pairs-rich (offline)** → keep enough sub-batches to pipeline → moderate batch wins

See runbook §2.3.2 for the full analysis.

---

## 4. MLX Port Investigation (framework-level)

**Goal:** Apple's native MLX framework typically delivers 1.5-3× speedup over PyTorch+MPS by skipping the PyTorch→MPS abstraction.

**Outcome:** investigated thoroughly, hand-port required, deferred.

| Path tried | Result |
|-----------|--------|
| `mlx-lm` for cross-encoders | ✗ Decoder-LMs only; `Model type xlm-roberta not supported` |
| `mlx-embeddings` direct convert | ✗ 393 parameters not in model — encoder yes, classifier head no |
| Hand-port (subclass + sanitize override) | Estimated 1-2 hrs; not pursued |

**Key finding:** `mlx-embeddings` supports the **architecture family** (XLM-RoBERTa) but not the **task variant** (`XLMRobertaForSequenceClassification`). The hand-port shape (~30-50 lines: classifier head + prefix-strip in `sanitize()`) is documented in §2.2.5 of the runbook for future reference.

**Decision:** ship `PyTorch+MPS+fp16+group=32+batch=128` (the §2.2.4 validated 5-run optimum at 27.6 min). The 1-2 hr hand-port cost vs ~30-60 min savings across realistic usage doesn't justify the investment for this lab. Calculation flips when:
- Productionizing rerank at high QPS
- mlx-embeddings adds `XLMRobertaForSequenceClassification` natively
- Pre-converted weights appear on `mlx-community/*`

---

## 5. Context Compression (Phase 3 — measured)

| Metric | Predicted | Measured | Notes |
|--------|-----------|----------|-------|
| compression_ratio | 0.25-0.50 | **0.142** | 3-4× more aggressive than predicted |
| compressed_words | 65-130 | 35.9 | ~7 sentences kept on average |
| completion_tokens | 100-200 | 60.04 | Thinking mode disabled at oMLX layer |
| per-query latency | 2-5 s | **2.18 s** | ✓ in spec range |
| n_attempted / n_ok | — | 50 / 50 | Clean run, no failures |

**Headline:** 50/50 successful compression at 2.18s per query, ratio 0.14 (3× more aggressive than predicted).

The aggressive ratio could mean either (a) MS MARCO passages have low relevant-content density and the compressor correctly extracts only answer-bearing sentences, or (b) the compressor is over-pruning and dropping context that the synthesizer would benefit from. **Phase 3.2 (LLM-as-judge) is the test that distinguishes these interpretations.**

### Iteration journey to clean run

The first attempts crashed and burned through several script-level workarounds before the right fix emerged at the configuration layer:

| Run | `max_tokens` | Outcome | Lesson |
|-----|--------------|---------|--------|
| #1 | 500 (spec default) | `AttributeError: 'NoneType' has no .split()` | Need defensive None handling |
| #2 | 1500 (after defensive fix) | First query "define preventive" failed with `finish_reason=length` | Reasoning chain exhausted budget |
| #3 | 3000 (workaround) | Better but still slow due to reasoning overhead | Working around symptom, not cause |
| **#4** | 3000 (thinking OFF) | **50/50 clean, 2.18s/query** | **Configuration > script-level tuning** |

The lesson: when LLM API responses have `content=None` and `reasoning_content` populated, the right fix is at the model layer (disable thinking mode in oMLX UI), not the script layer (bigger max_tokens). See runbook Troubleshooting + §3.1.1 for the full investigation.

### Phase 3.2 — measured: compression is safe to ship

Two runs were performed; the second is the calibrated number after a max_tokens fix to the judge.

| Run | judge max_tokens | n_ok / n_fail | compressed_wins | ties | raw_wins | wins+ties | verdict |
|-----|-------------------|---------------|-----------------|------|----------|-----------|---------|
| #1 | 500 | 28 / 2 | 11 (39.3%) | 10 (35.7%) | 7 (25.0%) | **0.750** | ship |
| **#2** | **3000** | **30 / 0** | **11 (36.7%)** | **10 (33.3%)** | **9 (30.0%)** | **0.700** | **ship** |

**Headline (calibrated): 70% wins+ties on a clean 30/30 sample, exceeds the 60% safety threshold.** The aggressive 0.14 compression ratio from §3.1 is empirically validated as safe-to-ship.

Outcome decomposition (Run #2):

| Outcome | Count | Rate | Interpretation |
|---------|-------|------|----------------|
| compressed_wins | 11 | 36.7% | Compression *helps* — filters noise that distracts the synthesizer |
| ties | 10 | 33.3% | Equally good — compression is "free" |
| raw_wins | 9 | 30.0% | Compression hurt — answer-supporting context was in the dropped 86% |

**Methodology note: Run #1's 75% was inflated by survivorship bias.** Both originally-failed queries (qid=9083, qid=68095) became `raw_wins` when properly judged in Run #2. The judge failures correlated with case difficulty (harder queries → more reasoning → exhausted 500-token budget) — so filtering them inflated the favorable rate. The honest production number is **70%, not 75%**.

This is a generalizable lesson: when LLM-eval failures correlate with case difficulty, `n_ok`-only metrics paint a rosier picture than the full sample. Always bump max_tokens until `n_failed=0` before reporting.

| Metric | Value |
|--------|-------|
| total_wall_sec | 199.9 (3.3 min for 30 queries) |
| per_query_sec | 6.66 (4 LLM calls per query: compress + 2 answers + judge) |
| Both judge failures resolved by | judge `max_tokens=500 → 3000` |

---

## 6. Chunking Sweep (Phase 4 — measured)

9-cell grid (chunk_size × overlap) on parent-level recall@5, 50 queries:

| chunk_size \ overlap | 0     | 64    | 128                  |
|----------------------|-------|-------|----------------------|
| **256**              | 0.686 | 0.727 | **0.786** ← winner   |
| 512                  | 0.494 | 0.507 | 0.541                |
| 1024                 | 0.493 | 0.493 | 0.493 ← flat in overlap |

Heatmap: `results/chunk_heatmap.png`. Raw JSON: `results/chunk_sweep.json`.

**Headline:** chunk_size=256 + overlap=128 wins at 0.786 recall@5, beating chunk_size=1024 by **24.5 percentage points** — a 50% relative gain just from re-chunking. Going from 256 → 512 cuts recall by ~25 pp; going further to 1024 changes nothing.

### What broke my expectation

I expected a U-shape — small chunks too narrow to hold the answer, large chunks too noisy, with a sweet spot in the middle. The data is **monotonic**: smaller is strictly better in the swept range. The "small chunks miss context" failure mode didn't materialize at 256 tokens on MS MARCO; the "big chunks dilute" failure mode is fully in charge by 512.

Overlap also turned out to be **size-dependent**, not uniformly good. It's worth +10 pp at chunk_size=256 (boundary fragmentation healer when ~100-token answers straddle 256-token boundaries) and zero at chunk_size=1024 (answers fit cleanly inside one chunk; overlap is wasted). At chunk_size=1024 the spec's 9 cells could have collapsed to 3.

### Why this happens — averaging vs concatenating

A chunk's BGE-M3 embedding is a learned average over its tokens. The relevant span's contribution to the pooled vector is roughly inversely proportional to chunk size — a 100-token answer is ~40% of a 256-token chunk's signal but only ~10% of a 1024-token chunk's. **Retrieval embeddings *average*; they don't *concatenate*.** Bigger chunks dilute the relevant direction with surrounding tokens that the cosine score has to average over.

### Single reflection — what to remember

- **Smaller chunks dominate on MS MARCO-style corpora.** Copy-pasting `chunk_size=512` from a tutorial would have cost ~25 pp of recall.
- **chunk_size and overlap interact.** Rule of thumb: overlap ≈ relevant_span_length / 2. At chunk_size=1024 with overlap=64, no real span fits in the overlap region.
- **Sweep before committing.** Build cost ~30 min, eval cost ~5 min — cheap. Not sweeping is the expensive option.

---

## What I learned

**On hybrid retrieval (Phase 1):** The "hybrid is the cheap-first alternative to reranking" framing is corpus-dependent. On MS MARCO with dense already saturated at 99.3% recall@10, hybrid has no headroom to add value — it actually *regressed* MRR by 8 points because RRF's rank-egalitarian arithmetic dilutes a strong retriever's confident picks with a weak retriever's noise. On FiQA where dense lands at 67% recall@10, hybrid finally produced a real (small) lift. The takeaway is that hybrid's value is *inversely proportional to how saturated the dense baseline already is.*

**On reranker tuning (Phase 2.2):** Five measurements, three theories tested, one confirmed. The investigation arc demonstrated that simple causal models ("padding penalty," "per-call overhead") fail empirically when applied to heterogeneous compute (GPU + CPU + tokenizer + memory hierarchy). The factor that actually mattered — GPU pipelining via sub-batches per call — only became visible after running the falsifying experiments for the simpler theories. The validated mental model (10+ sub-batches per `predict()` call) is now both empirically supported and mechanistically explained.

**On framework selection (Phase 2.2 + MLX detour):** Running `mlx_lm.convert` and getting `Model type xlm-roberta not supported` is a different *kind* of engineering result than "the experiment regressed." It's an infrastructure capability gap — the framework I wanted to use doesn't implement the model class I needed. Diagnosing this required reading mlx-embeddings' source code to confirm the encoder is supported but the classifier head is not. The cost-benefit calculation (1-2 hr hand-port vs ~30-60 min runtime savings) made the deferral defensible. Documented as a forward marker — when ecosystem support matures, this is where to revisit.

**On chunking (Phase 4):** I expected a U-shape; I got a monotonic dropoff. Smaller is strictly better in the swept range, by 24.5 percentage points end-to-end. The mechanistic reading is clean: embeddings average rather than concatenate, so a 100-token relevant span's signal share collapses from ~40% (chunk_size=256) to ~10% (chunk_size=1024). Overlap is not a uniformly good knob either — it's a boundary-fragmentation healer, completely inert at chunk_size=1024 because answers don't straddle boundaries that wide. The takeaway worth carrying out: for any real RAG corpus, **sweep chunk_size × overlap rather than copy-pasting from a tutorial** — the build cost is ~30 minutes, the recall delta can be ~25 pp.

---

## Bad-case journal

| Failure | Root cause | Fix / lesson |
|---------|-----------|--------------|
| FiQA upsert `httpx.ReadTimeout` after 31 min | Default 5s `QdrantClient` timeout breaks once HNSW indexing competes with upserts (after `indexing_threshold=10000`) | Bump to `timeout=60`, add `_upsert_with_retry` with exponential backoff, replace empty-or-not gate with resumability check (`start = (points_count // BATCH) * BATCH`) |
| FiQA re-run silently skipped ingest | The `points_count == 0` gate treats partial state as "done" | Replace with resumability check; round down to nearest batch boundary; re-upserts with same id are idempotent |
| RRF hybrid lost MRR vs dense on MS MARCO (-7.9pp) | Cormack-style RRF (`1/(60+rank)`) treats both retrievers as equally informative; weaker sparse votes diluted dense's rank-1 picks | Disproven by empirical recall@100 measurement — hybrid regression isn't a fusion-tuning issue, it's a corpus regime issue (dense already saturated) |
| Predicted "group=1 → 5-8 min total" but got 34 min (24% regression) | Padding-penalty theory was incomplete — per-call CPU overhead competes with the padding savings | §2.2.2 documents the disproof; revised hypothesis was ALSO falsified by §2.2.3 |
| Predicted "group=8 sweet spot at 22-26 min" but got 33.9 min | Per-call-overhead theory also wrong — overhead is ~2 ms not 50 ms | §2.2.3 documents the disproof; revised hypothesis (GPU pipelining) was confirmed by §2.2.4 |
| MLX port: `Model type xlm-roberta not supported` from `mlx-lm` | mlx-lm scoped to decoder LMs only | Investigated mlx-embeddings as alternative; found encoder supported but classifier head missing; documented hand-port shape and deferred |

---

## Infra bridge (for interview narrative)

- **Reranker = feature-engineering layer.** Bi-encoder retrieves top-50 cheaply (~5 ms); cross-encoder rescores those 50 with full self-attention (~115 ms on M5 Pro fp16). The two-stage funnel achieves cross-encoder accuracy at near-bi-encoder latency.
- **fp16 conversion = lossy-serialization codec choice.** Like picking Snappy vs Zstd for a Parquet column — same data, different precision/throughput trade-off, validated by measurement (recall identical, nDCG within rounding noise, 2.86× speedup).
- **GPU pipelining = the same scheduling problem as I/O queue depth.** Database connection pools, network packet sizing, GPU sub-batches — all governed by "the scheduler needs N in-flight requests to hide latency." Once you see the pattern, it transfers across domains.
- **Framework-port investigation = the same as evaluating an external library.** Run the import, capture the error, read the source, decide whether to integrate / fork / write-from-scratch. The empirical cycle (try → fail → diagnose → estimate → defer-or-implement) is identical.

---

## Forward markers

- **mlx-embeddings adds `XLMRobertaForSequenceClassification`** → swap `01_rerank.py` to MLX, re-measure (estimated 10-15 min total)
- **`mlx-community/bge-reranker-v2-m3` published** → 5-line script change to load from HF
- **Bigger chunking sweep** → extend to chunk_size ∈ {128, 192, 256} to find where small-chunk gain saturates
- **High-QPS production deployment** → re-evaluate hand-port; the cost calculation flips at high run count
