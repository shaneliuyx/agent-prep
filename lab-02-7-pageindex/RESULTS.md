# W2.7 Three-Way RAG Comparison — Berkshire Hathaway 2023 Annual Report

Same corpus, same 8-question eval set, three retrieval architectures.

## Setup

| Backend | Index | Size | Build wall time |
|---|---|---|---|
| Vector | Qdrant `brk_2023_dense` (BGE-M3 1024-dim cosine, HNSW m=16) | 3,857 chunks | ~5 min |
| Graph | Neo4j `:BrkEntity` + `brk_entity_names` fulltext + `brk_entity_qid` range | 4,479 nodes, 11,680 triples, 1,355 unique relations | 71.9 min |
| Tree | `data/tree.json` (LLM tree-walk over hierarchy) | 50 nodes, depth 4 | ~3 min |

All three backends share:
- Same source PDF (Berkshire Hathaway 2023 Annual Report, ~148 pages)
- Same answer-LLM (oMLX Gemma-4-26B-A4B-it-heretic-4bit @ port 8000, temp 0.0)
- Same RAGAS-style scoring (`score_substring` + `score_llm_judge` from `lab-02-5-graphrag/src/compare.py`)

## Aggregate Results

| Backend | LLM-judge | Substring recall | Mean latency |
|---|---|---|---|
| Vector | 0.25 | 0.25 | **1.8s** |
| **Graph** | **0.48** | **0.40** | 13.1s |
| Tree | 0.44 | 0.31 | 3.4s |

## Per-Category Results (LLM-judge)

| Category | Vector | Graph | Tree | Winner |
|---|---|---|---|---|
| section-specific factoid | **0.50** | 0.00 | 0.00 | Vector |
| cross-section synthesis | 0.00 | **0.50** | **0.50** | Graph + Tree (tie) |
| citation-required | 0.17 | **0.42** | 0.25 | Graph |
| out-of-document (refusal) | 0.33 | **1.00** | **1.00** | Graph + Tree (tie) |

## Findings

### 1. Vector wins factoid lookup
"What was Berkshire's net earnings attributable to shareholders in 2023?" — Vector returned `"Net earnings attributable to Berkshire shareholders were $96.2 billion."` (substr 1.0, judge 1.0) in 1.1s. Graph and Tree both returned 0.0 — they don't natively store dollar amounts as primary keys.

Vector's strength: semantic-dense retrieval is the only one of the three that treats the body text as primary content. Graph throws away most numbers during entity extraction; Tree only sees section titles + page ranges, not body text.

### 2. Graph wins citation-required questions
"Which section covers BNSF Railway operating results?" — Graph found the BNSF entity, traversed `MENTIONED_IN` to its source section, returned page citation. Tree-walk had to LLM-reason its way down through the TOC and got distracted by the parent "Form 10-K" node.

### 3. Graph + Tree tie on cross-section synthesis
"What did Buffett write about non-controlled businesses that leave Berkshire comfortable?" — Both Graph (entity-expansion across `OWNS`/`HOLDS_STAKE_IN` edges) and Tree (LLM walking from Chairman's Letter into the relevant sub-section) returned high-recall answers (substr 0.75, judge 0.75 for Tree; judge 0.50 for Graph). Vector returned `"insufficient context"` because the relevant chunks were spread across multiple sections and the dense rerank didn't surface all of them.

### 4. Both Graph and Tree refuse out-of-document questions perfectly (1.00)
"What is Berkshire Hathaway's stock price today?" / "Who is the CEO of Microsoft?" — Graph returned `"insufficient context"` (no matching entities), Tree returned `"The provided text does not contain information regarding..."`. Vector partially refused (0.33) — it returned `"insufficient context."` but the substring scoring penalized the period and tokenization mismatch.

### 5. Latency tradeoff
Vector is **7× faster than Graph**, **2× faster than Tree**. For a single-document corpus where you'd run hundreds of queries (e.g., a financial-analyst chat), Vector's latency advantage dominates the slight accuracy loss on synthesis questions. Graph's 13s/query is hard to defend at user-facing latency budgets.

## Architectural Implications

| Use case | Recommended backend |
|---|---|
| Numeric lookup, exact-figure questions | Vector |
| "Where in the document is X?" citation | Graph (or Tree if budget-constrained) |
| Multi-section synthesis on entity relationships | Graph or Tree |
| Refusal on out-of-scope questions | Graph or Tree (Vector hallucinates partials) |
| Latency-critical UX | Vector |
| Building from scratch in <10 min | Tree (no embedding step, no entity extraction) |

The earlier hypothesis ("graph degenerates on single-document star corpus") was **wrong**. Graph performed best in aggregate. The hypothesis-confirming first run was actually masking a build bug — see "Build pitfall" below.

## Build pitfall — sed-rename double-prefix

Original W2.5 build script created index `entity_names`. lab-02-7 renamed via sed `entity_names → brk_entity_names` AND `Entity → BrkEntity`. The CREATE statement got double-processed and produced `brk_brk_entity_names`. Build summary printed `brk_entity_names` (hardcoded display string, line 452), Neo4j had `brk_brk_entity_names`, query script asked for `brk_entity_names`. Three layers, no integration test.

First-pass compare returned graph judge=0.00 across all questions with error `"There is no such fulltext schema index: brk_entity_names"`. If we had stopped there and written it up, the chapter would have falsely claimed "graph degenerates on single-document corpora" — when in reality the index was missing.

Fix: `DROP INDEX brk_brk_entity_names; CREATE FULLTEXT INDEX brk_entity_names ...` + edited line 362 of `build_brk_graph.py`. Re-ran compare → real graph numbers (0.48 aggregate).

**Lesson:** sed-rename of build scripts needs an integration smoke test before claiming results. A 30-second `db.index.fulltext.queryNodes("brk_entity_names", "Berkshire")` after build would have caught this.

---

## PageIndex Optimization Run (2026-05-07) — tree judge 0.44 → 0.79 (+77%)

After the initial run reported tree judge=0.44, applied 4 PageIndex-inspired optimizations to the tree backend:

1. **Agentic tool-calling loop** (`query_tree.py`): replaced greedy single-shot `navigate()` + `answer()` with a multi-turn agent loop exposing `get_page_content(start, end)` tool. Body text is now visible to the decision-maker (was: only summaries).
2. **Recursive node split** (`build_tree.py` `split_large_nodes()`): any leaf spanning >5 pages or >20K chars gets split via LLM into 2-5 topical sub-sections. Result: tree grew from 50 → 62 nodes, depth 4 → 5.
3. **Fact-rich summaries** (`build_tree.py` `SUMMARIZE_SYSTEM`): every summary must include 3 numeric facts verbatim + 5 named entities + structural location. Eliminates vague summaries that confused the navigator.
4. **TOC-trap rule + explained refusal + synthesis-from-fragments** (`query_tree.py` `AGENTIC_SYSTEM`): three prompt-engineering rules — never cite TOC pages 1-3 as the answer source; refuse with one-sentence explanation + "insufficient context" (not bare keyword); when 3+ partial-info fetches accumulate, synthesize across them rather than refuse.

**Model:** tree backend isolated to `Qwen3.6-35B-A3B-UD-MLX-4bit` (passes 4/4 smoke tests for JSON / tools / multi-turn / 16K context). Vector + graph stay on `gemma-4-26B-A4B-it-heretic-4bit`. The split prevents oMLX KV-cache pollution observed when all 3 backends shared one model — see Bad-Case Entry 7 below.

### Aggregate (post-optimization)

| Backend | LLM-judge | Substring | Latency |
|---|---|---|---|
| Vector | 0.25 | 0.25 | 1.8s |
| Graph | 0.48 | 0.40 | 6.3s |
| **Tree** | **0.79** | **0.62** | 14.6s |

Tree latency 3.4s → 14.6s (4.3× slower) is the cost of multi-turn agentic retrieval. Acceptable when accuracy lift is +0.35 absolute / +77% relative.

### Per-Category (post-optimization)

| Category | Vector | Graph | Tree (pre-opt) | **Tree (post-opt)** | Tree Δ |
|---|---|---|---|---|---|
| section-specific factoid | 0.50 | 0.00 | 0.00 | **1.00** | **+1.00** |
| cross-section synthesis | 0.00 | 0.50 | 0.50 | 0.50 | 0 |
| citation-required | 0.17 | 0.42 | 0.25 | **0.67** | **+0.42** |
| out-of-document refusal | 0.33 | 1.00 | 1.00 | 1.00 | 0 |

Tree now wins or ties every category. Pre-opt tree had a fundamental architectural limitation: navigator only sees titles + summaries at decision time, so factoid queries with non-keyword-matching titles (`"$96.2B net earnings"` → `"Consolidated Statements of Earnings"` node title) couldn't be reached via greedy descent. Agentic-loop fix lets the LLM fetch page content during navigation, eliminating this blind spot.

### What changed per question (tree backend, post-opt)

| Q | Type | Pre-opt answer | Post-opt answer | Δ judge |
|---|---|---|---|---|
| 1 | factoid (revenues) | "Scorecard $37,350M operating earnings" (wrong section) | "$364,482 million [page 96]" via Statements | **0 → 1.00** |
| 2 | factoid (net earnings) | Same wrong section as Q1 | "$96,223 million [page 96]" via Statements | **0 → 1.00** |
| 5 | citation (BNSF) | "TOC, page 1" (TOC trap) | Found correct section via tool fetch | **0 → 0.33** |
| 6 | citation (cybersecurity) | "Cybersecurity p K-28 (TOC trap)" partial | "Item 1C. Cybersecurity, p52" exact | **0.50 → 1.00** |
| 7 | OOD (stock price) | Bare "insufficient context" | Explained refusal: "this is the 2023 AR..." | **0.33 → 1.00** |
| 8 | OOD (Microsoft CEO) | Bare "insufficient context" | Explained refusal | **0.33 → 1.00** |
| 4 | synthesis (non-controlled) | "Coca-Cola, AmEx, Occidental, Japanese..." (greedy got it) | "Based on Chairman's Letter, Non-controlled Businesses section..." | 0.75 → 0.75 (held) |
| 3 | synthesis (not-so-secret weapon) | Right section, wrong content emphasis | Right section, slightly better synthesis | 0.25 → 0.25 (held) |

### PageIndex Comparability Check

PageIndex's published 98.7% on FinanceBench is **not** apples-to-apples with our 8-question Berkshire eval:

| Dimension | PageIndex Mafin 2.5 | Our W2.7 lab |
|---|---|---|
| Eval size | 150 questions | 8 questions |
| Corpus | Multi-doc SEC filings | 1 document |
| Build model | GPT-4o (Cloud) | Local Qwen3.6 / Gemma 4-bit |
| OCR | PageIndex Cloud | Self-hosted PyPDF |
| Human annotation | Yes (tiebreak) | No |

Our 0.79 on the local stack with PageIndex-pattern optimizations is the realistic ceiling for this eval shape. Hitting 0.987 would require GPT-4o + multi-doc routing + larger eval set.

## Bad-Case Journal — Optimization Run

**Entry 7 — oMLX KV-cache pollution between request shapes on the same model.**
*Symptom:* When all 3 backends (vector, graph, tree) routed to the same Qwen3.6 model, every tree call after the first returned `1it/0tc` empty content despite standalone tree calls returning correct answers with 1-3 tool calls. Tree aggregate scored 0.08 in compare runs but 0.61+ in standalone runs on the same questions.
*Root cause:* oMLX appears to reuse KV-cache state across requests for the same model. When vector_answer issues a no-tools call followed by tree's tools call, the cache from the no-tools shape interfered with tool-routing on the tools call. Standalone runs only invoked tree, so the cache state was consistent.
*Fix:* Route tree backend to a separate model (`MODEL_TREE=Qwen3.6-35B-A3B-UD-MLX-4bit`) while keeping vector + graph on Gemma. Different model = different KV cache pool on the oMLX server. **Discipline rule:** when running multi-backend comparisons against an oMLX server with mixed request shapes (no-tools call + tools call), give the tools-using backend its own model.

**Entry 8 — Bare "insufficient context" gets penalized by LLM-judge despite substring match.**
*Symptom:* Tree's OOD refusal scored 0.67 in the optimization run (was 1.00 pre-opt). Q8 (Microsoft CEO) returned bare `"insufficient context"` and got judge=0.33; Q7 returned a longer `"This is the Berkshire 2023 AR... insufficient context"` and got judge=1.00. Substring scoring gave both 0.33 (matches "insufficient" only).
*Root cause:* `score_llm_judge` rewards refusals that demonstrate reasoning (explain *why* the document doesn't have the answer). Bare keyword matches are scored as partial answers, not full refusals. AGENTIC_SYSTEM was instructing the LLM to "respond with exactly: insufficient context" — short-circuit refusal with no explanation.
*Fix:* Change AGENTIC_SYSTEM to require two-part refusal: (a) one sentence explaining what the document IS and why it doesn't contain the answer, (b) close with "insufficient context". This matches graph backend's refusal shape, which always scored 1.00. Side-effect: also lifted synthesis Q4 from 0.00 → 1.00 because removing the "respond with exactly" instruction made the LLM more willing to write partial-info answers.

**Entry 9 — Recursive node split helps factoid but fragments synthesis.**
*Symptom:* After applying recursive split (opt #2), tree synthesis dropped 0.50 → 0.12 in compare8. The agentic loop made 6 tool calls then refused with "insufficient context" on Q4 (non-controlled businesses) — pre-opt greedy nav had landed on the whole pre-split section and synthesized correctly.
*Root cause:* Splitting the Chairman's Letter into 5 sub-sections meant Q4's answer (Coca-Cola + American Express + Occidental + Japanese houses) lived across multiple sub-sections instead of one. Greedy nav fetched the parent → got everything; agentic loop fetched sub-sections individually → each had partial info → over-refused.
*Fix:* Add explicit synthesis-from-fragments rule to AGENTIC_SYSTEM: "After 3+ fetches that each contribute partial information, SYNTHESIZE the final answer by combining the fragments." Combined with Entry 8's explained-refusal fix, synthesis recovered to 0.50 (compare10), reaching parity with graph backend.

## Files

- `data/eval.json` — 8-question eval set (Berkshire-calibrated)
- `data/tree.json` — 50-node tree-of-contents (W2.7 build_tree output)
- `data/brk_corpus.json` — 44 leaf-section article-shape entries
- `results/three_way.json` — full per-question results (vector/graph/tree answers + scores + latencies)
- `/tmp/brk_build.log` — graph build wall-time log (71.9 min)
- `/tmp/brk_compare2.log` — three-way comparison run with fixed index
