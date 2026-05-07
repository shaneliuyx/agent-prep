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

## Files

- `data/eval.json` — 8-question eval set (Berkshire-calibrated)
- `data/tree.json` — 50-node tree-of-contents (W2.7 build_tree output)
- `data/brk_corpus.json` — 44 leaf-section article-shape entries
- `results/three_way.json` — full per-question results (vector/graph/tree answers + scores + latencies)
- `/tmp/brk_build.log` — graph build wall-time log (71.9 min)
- `/tmp/brk_compare2.log` — three-way comparison run with fixed index
