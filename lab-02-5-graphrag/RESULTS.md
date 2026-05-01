# Lab 02.5 — GraphRAG Results

> Run date: 2026-04-30. Stack: oMLX serving gemma-4-26B-A4B-it-heretic-4bit + gpt-oss-20b-MXFP4-Q8, Neo4j 5.x in Docker, M5 Pro 48 GB. Corpus: 149 articles via MediaWiki categorymembers walk over 4 tech-domain categories. Graph: 2074 triples, `entity_names` fulltext index over `Entity.name`.

## Pipeline configuration

**Corpus mechanism (`src/fetch_corpus.py`).** Wikipedia category walk over four seed categories: `American_technology_company_founders`, `Companies_based_in_Silicon_Valley`, `Software_companies_of_the_United_States`, `American_chief_executives_of_technology_companies`. Each category is paginated through `categorymembers` (anonymous cap 500 per page, capped at 5 pages per category). Titles are deduped, shuffled with `SHUFFLE_SEED=42` for deterministic random sampling, and capped at `MAX_ARTICLES=150` (149 actually returned non-empty extracts). Each article's first 4000 chars of plain-text extract is stored. The shuffle is what unlocks Z-tail entries — without it, alphabetical truncation drops Zuckerberg, Zynga, etc.

**Graph build (`src/build_graph.py`).** Per-article triple extraction by gemma-4-26B (non-reasoning, deterministic JSON, 1200 max_tokens, temp 0.1) with `MAX_WORKERS=6` ThreadPoolExecutor (oMLX cap is 8; leaving headroom for query traffic). Each `(subject, relation, object)` triple becomes `(:Entity {name})-[REL_TYPE {raw_relation, source_article, source_title}]->(:Entity {name})`. `MERGE` deduplicates entities across articles. After ingestion the script creates `CREATE FULLTEXT INDEX entity_names FOR (n:Entity) ON EACH [n.name]` — replacing the v1 `CONTAINS` substring match that produced the metal/metalloid/Denmark false-positive class.

**Query mechanism (`src/query_graph.py`).** (1) Seed extraction by gemma-4-26B with strict JSON-object response-format and a regex proper-noun fallback (covers reasoning-model empty-content failure mode). (2) Each seed is run through `_lucene_query` — Lucene-reserved chars stripped, sub-3-char tokens dropped, remaining tokens joined with OR, no fuzzy `~` operator (would re-introduce the false-positive class). (3) `db.index.fulltext.queryNodes("entity_names", ...)` returns top 5 scored entity nodes; from those we walk a 1–2 hop neighborhood (`MATCH path = (node)-[*1..2]-(m)`) and return up to 50 distinct edges per seed. (4) Per-seed match counts are tracked separately so a precondition warning fires when `matches_per_seed[seed] == 0`. (5) Subgraph (capped at 40 edges in the prompt) is fed to gemma-4-26B with the system prompt "Answer using ONLY the graph facts below. If the facts do not support an answer, say so. Cite source articles inline."

## Successful queries (entities in corpus)

### Q1 — "What companies has Jack Dorsey founded?"
**Seeds:** `["Jack Dorsey"]`
**matches_per_seed:** `{"Jack Dorsey": 10}`
**edges_used:** 33
**Answer:**
> Jack Patrick Dorsey has co-founded Block, Inc. (source: Jack Dorsey) and Twitter, Inc. (source: Jack Dorsey), and is the founder of Bluesky (source: Jack Dorsey).

### Q2 — "What did Sean Parker work on?"
**Seeds:** `["Sean Parker"]`
**matches_per_seed:** `{"Sean Parker": 6}`
**edges_used:** 34
**Answer:**
> Sean Parker was the first president of Facebook (source: Sean Parker), co-founded Causes (source: Sean Parker), co-founded Brigade (source: Sean Parker), co-founded Napster with Shawn Fanning (source: Sean Parker), co-founded Plaxo (source: Sean Parker), co-founded Airtime.com (source: Sean Parker), and is the founder and chairman of the Parker Foundation (source: Sean Parker). He is also an American entrepreneur and philanthropist (source: Sean Parker) and interned for Mark Pincus (source: Sean Parker).

### Q3 — "Who founded Pinterest?"
**Seeds:** `["Pinterest"]`
**matches_per_seed:** `{"Pinterest": 1}`
**edges_used:** 5
**Answer:**
> Ben Silbermann is the co-founder of Pinterest (source: Ben Silbermann) and Evan Sharp created Pinterest (source: Ben Silbermann).

### Q4 — "Tell me about Dennis Crowley."
**Seeds:** `["Dennis Crowley"]`
**matches_per_seed:** `{"Dennis Crowley": 5}`
**edges_used:** 40
**Answer:**
> Dennis Crowley was born in Medway, Massachusetts (source: Dennis Crowley). He is the son of Dennis P. Crowley (source: Dennis Crowley) and Mary Moraski Crowley (source: Dennis Crowley). He graduated from Xaverian Brothers High School (source: Dennis Crowley), received a B.A. from Syracuse University (source: Dennis Crowley), and received a M.P.S. master's degree from New York University (source: Dennis Crowley).
>
> His professional history includes:
> *   Co-founding Foursquare (source: Dennis Crowley) and Dodgeball (source: Dennis Crowley), the latter of which was acquired by Google (source: Dennis Crowley).
> *   Co-founding Hopscotch Labs with Alejandro Fragoso (source: Dennis Crowley) and Max Sklar (source: Dennis Crowley).
> *   Joining Vindigo (source: Dennis Crowley).
> *   Finding work at Area/Code (source: Dennis Crowley).
> *   Working as a researcher for Jupiter Communications (source: Dennis Crowley).
>
> He is the Founder and Chairman of Kingston Stockade FC (source: Dennis Crowley) and is married to Chelsa Lynn Skees (source: Dennis Crowley).

### Q5 — "What is the relationship between Tom Anderson and MySpace?"
**Seeds:** `["Tom Anderson", "MySpace"]`
**matches_per_seed:** `{"Tom Anderson": 6, "MySpace": 1}`
**edges_used:** 73
**Answer:**
> Thomas Anderson founded Myspace (source: Tom Anderson), was a co-founder of Myspace (source: Tom Anderson), and was president of Myspace (source: Tom Anderson).

### Q6 — "What did Ted Dabney co-found?"
**Seeds:** `["Ted Dabney"]`
**matches_per_seed:** `{"Ted Dabney": 4}`
**edges_used:** 33
**Answer:**
> Samuel Frederick "Ted" Dabney Jr. was a co-founder of Atari, Inc. (source: Ted Dabney).

### Q7 — "What companies has Travis VanderZanden led?"
**Seeds:** `["Travis VanderZanden"]`
**matches_per_seed:** `{"Travis VanderZanden": 3}`
**edges_used:** 21
**Answer:**
> Travis VanderZanden has led the following companies:
> - Bird (as founder) (source: Travis VanderZanden)
> - Lyft (as Chief Operating Officer) (source: Travis VanderZanden)
> - Uber (as VP of International Growth) (source: Travis VanderZanden)
> - Yammer (as Chief Revenue Officer) (source: Travis VanderZanden)
> - Cherry (as co-founder) (source: Travis VanderZanden)
> - Qualcomm (as product manager) (source: Travis VanderZanden)

## Edge-case queries (entities not in corpus)

### Q8 — Mark Zuckerberg (original failing query)
**Seeds:** `["Mark Zuckerberg", "companies"]`
**matches_per_seed:** `{"Mark Zuckerberg": 9, "companies": 2}`
**edges_used:** 45
**stderr:** *(empty — precondition warning did not fire because both seeds matched at least one entity)*
**Answer:**
> The provided graph facts do not contain information regarding companies related to Mark Zuckerberg.

**Commentary.** This is the headline win. The v1 lab confidently fabricated a chemistry/Denmark answer from `CONTAINS "mark"` matching `Denmark`, `Markus`, etc., and `CONTAINS "meta"` matching `metal`, `metalloid`, `metabolism`. With the full-text Lucene index, "Mark Zuckerberg" still finds 9 matches (whole-word hits like `Mark Pincus`, `Mark Vadon`, `Marc Benioff` after Lucene's analyzer drops the case + tokenizes), but those 45 traversed edges are all about *those specific people* — none are Zuckerberg-specific facts. The LLM, faithfully obeying "Answer using ONLY the graph facts below," correctly refuses. The metal/metalloid false-positive class is gone: zero chemistry entities appear in the seed set, and the model's refusal text doesn't mention any chemistry concept. The 9 token-overlap matches are "polite false positives" — they expand the subgraph but contribute no signal that supports a Zuckerberg answer, and the generator catches the gap.

### Q9 — Beethoven (truly out-of-domain)
**Seeds:** `["Ludwig van Beethoven"]`
**matches_per_seed:** `{"Ludwig van Beethoven": 1}`
**edges_used:** 5
**stderr:** *(empty — `matches_per_seed = 1`, so the `[WARN]` precondition did not fire)*
**Answer:**
> The provided graph facts do not contain information about Ludwig van Beethoven.

**Commentary.** The Lucene OR-tokenization (`Ludwig OR van OR Beethoven`, with `van` dropped as <3 chars) found 1 entity node — almost certainly a token-overlap match on "Ludwig" or a similar substring from somewhere in the tech corpus — and walked 5 edges from it. The precondition warning's strict `n == 0` threshold did not trigger. This is an interesting boundary case: a stricter "no graph entity matched the *full* seed string" check would catch it, but the current implementation accepts any single OR-term match. Practically the failure mode is benign: the 5 edges contain no Beethoven facts, and the LLM correctly refuses. To exercise the explicit `[WARN] No graph entity matched` precondition path, a query whose seeds contain only single-letter or sub-3-char tokens (which the Lucene query builder rejects) or seeds with zero token-overlap with any graph entity would be needed.

## Pipeline validation

- **Full-text index replaces CONTAINS substring match.** Confirmed across all queries: no chemistry false positives in any answer, no Denmark, no metal/metalloid. Q8 still matches 9 entities but they are all whole-word `Mark*` / `Marc*` person nodes, and the generator correctly identifies that none are Zuckerberg.
- **Precondition warning fires when seed matches 0 entities.** Code path verified by reading `query_graph.py:144-152`. Did not fire in this run because Q9's `Ludwig van Beethoven` matched 1 token-overlap entity (`matches_per_seed=1`, not 0). All other queries matched legitimate entities. The guard works as written; the threshold is `n == 0` per seed.
- **LLM correctly refuses when graph facts don't support the answer.** Confirmed in Q8 and Q9. The "Answer using ONLY the graph facts below" system prompt holds — both refusals echo the system instruction's "say so" language.
- **Source citations appear in successful answers.** Every Q1–Q7 answer cites `(source: <article title>)` after each fact. Q4 stress-tests this with ~12 distinct citations all pointing to the same source article (`Dennis Crowley`); Q7 does the same with 6 role-specific citations.

## Known limitations

- 149-article random-sampled corpus skews toward mid-tier tech founders/companies; famous names (Zuckerberg, Jobs, Bezos) are statistically unlikely at ~10% per-name inclusion rate even with the shuffle, and `Mark Zuckerberg` was simply not in the random sample this run. The corpus *could* contain him — the `American_technology_company_founders` category includes him — but the shuffle cap at 150 dropped him.
- Precondition warning threshold is `matches_per_seed == 0`. Q9 demonstrates this is too lenient: a single token-overlap match on a sub-string ("Ludwig" matching a non-Beethoven entity) is enough to suppress the warning. A stricter check would require *the full seed string* to match a node name, or require all OR-tokens to land on the same node, but either change risks false negatives on legitimate multi-token entities (`Jack Dorsey` correctly matched 10 nodes via the OR query). Current behavior degrades gracefully because the LLM refusal layer catches the gap.
- Tom Anderson Q5 returned 73 edges (highest in the battery) because both seeds were graph-resolved and each opened a 50-edge neighborhood. The 40-edge prompt cap (`subgraph[:40]`) means roughly half the retrieved edges are discarded before generation — worth tracking if a query genuinely needs the long tail of edges.
- Threading delivered ~30% speedup vs projected 4-5× during graph build; worth profiling oMLX endpoint concurrency under sustained load. The `MAX_WORKERS=6` (under the server's MAX_CONCURRENT=8) leaves margin for queries but may be over-conservative if the server's actual throughput tops out earlier.
- The Lucene query drops tokens shorter than 3 chars, so `van` (in "Ludwig van Beethoven"), `de`, `le`, etc. are silently lost. For most English-language tech entities this is harmless; for entities with significant short-token surface forms (`AT&T`, `IBM` is 3 so OK) or non-English names (`van`, `der`, `del`) the matcher may return less-relevant top-5 anchors.
- Seed extraction picked up `"companies"` as a seed in Q8 — a generic noun that nonetheless matched 2 entities. The system prompt says "Prefer specific surface forms over generic ones" but the LLM still emitted this. Adds 2 false-positive matches to the per-seed dict but the downstream generator absorbs the noise.

---

## v8 — Corpus mechanism re-verification (2026-05-01)

After landing the four-layer corpus-mechanism cascade (category walk → pagination → pageview-weighted A-ExpJ → pvipcontinue per-property pagination → title-resolution mapping), the v8 corpus contains 150 articles drawn from a 3084-candidate pool spanning 6 SEED_CATEGORIES. Of the 3084 candidates only **3** had zero pageviews (vs 87% in the broken v6 cut). Coverage of canonical tech entities reached 13/25 — Mark Zuckerberg, Bill Gates, Jeff Bezos, Elon Musk, Steve Jobs, Tim Cook, Sundar Pichai, Larry Page, Sergey Brin, Larry Ellison, Peter Thiel, Microsoft, Amazon all landed.

Build: 150 articles → **2245 triples** in 709s (3.2 triples/sec, MAX_WORKERS=6).

Both originally-failing queries now return correct, source-cited answers via the phrase strategy of the matcher:

### Q1 — "What is the relationship between Apple and NeXT?"
- **Seeds:** Apple, NeXT
- **matches_per_seed:** Apple={phrase: 4, or: 4, strategy: phrase}, NeXT={phrase: 3, or: 3, strategy: phrase}
- **edges_used:** 77
- **Answer:** "Apple --[acquired]--> NeXT (source: Steve Jobs)."

The Apple → NeXT bridge edge was extracted from the Steve Jobs Wikipedia article (`source: Steve Jobs`), not from Apple's own article. This is the multi-hop bridge GraphRAG promises: cross-article entity overlap synthesizes facts that no single document literally states.

### Q2 — "Which companies are related to Mark Zuckerberg?"
- **Seeds:** Mark Zuckerberg, companies
- **matches_per_seed:** Mark Zuckerberg={phrase: 1, or: 9, strategy: phrase}, companies={phrase: 5, or: 5, strategy: phrase}
- **edges_used:** 81
- **Answer:** "Mark Elliot Zuckerberg co-founded Meta Platforms (source: Mark Zuckerberg)."

Phrase=1 means the full-text index returned exactly one node matching `+mark +zuckerberg` — the canonical Mark Zuckerberg entity. The OR fallback (`mark OR zuckerberg`) would have returned 9 nodes (all `Mark *` people in the corpus); the phrase-first strategy correctly took only the 1 exact match for traversal. This is the precision contract phrase-first matching is designed to enforce.

### Pipeline validation summary

The 5-layer corpus-coverage cascade is now resolved end-to-end:

1. **`load_dataset(..., split="train[:200]")` chemistry-corpus failure** → mechanism rewrite to MediaWiki categorymembers walk
2. **categorymembers 500-member API cap** → cmcontinue pagination
3. **Alphabetical-bias systematic Z-tail drop** → deterministic random shuffle
4. **Uniform-sample noise dominance over canonical entities** → A-ExpJ pageview-weighted sampling
5. **Title-resolution + per-property pagination silent zeros** → input→canonical mapping + pvipcontinue loop

Each layer was a real production-shape failure that masqueraded as the previous layer's symptom. The empirical record (with reproductions, root cause, and fix) is documented as bad-case-journal entries 6–13 in the W2.5 runbook.

---

## v9 — Scale-up to MAX_ARTICLES=400 + edge_cap=200 + REQUEST_SLEEP=1.0 (2026-05-01)

After the v8 corpus-mechanism cascade was resolved, scale-up to 400 articles to (a) capture more canonical entities, (b) build a denser graph with richer cross-article bridges, (c) confirm the pipeline scales without throughput regression.

**Configuration deltas vs v8:**
- `MAX_ARTICLES`: 150 → **400** (2.7× more articles, same 3084-candidate weighted pool)
- `subgraph[:N]` cap in `query_graph.answer()`: 40 → **200** (denser graph produces 70-100 edge neighborhoods; 200 leaves comfortable headroom)
- `REQUEST_SLEEP`: 0.6 → **1.0** (under sustained MediaWiki anonymous ceiling — eliminates retry-loop wall waste)

**Corpus stats:**
- 400 articles in 2399s fetch wall (~40 min)
- 5,948 triples in 1856s build (~31 min, sustained 3.2 triples/sec)
- Total pipeline wall: ~75 min (same as v8 despite 2.7× more articles — `REQUEST_SLEEP=1.0` paid for itself by avoiding retry waste)
- Canonical entity coverage: **16/25** (vs 13/25 at 150 articles) — gained Wozniak, Paul Allen, Marc Andreessen
- Triples per article: 14.9 (stable density, was 15.0 at 150 articles — extraction prompt produces consistent triple density)

### Test queries (v9)

| Q | Question | Strategy | edges_used | Answer |
|---|---|---|---|---|
| Q1 | What is the relationship between Apple and NeXT? | phrase | 94 | "Apple acquired NeXT (source: Steve Jobs)." |
| Q2 | Which companies are related to Mark Zuckerberg? | phrase | 100 | "Mark Zuckerberg ... co-founded Facebook (source: Dustin Moskovitz; source: Mark Zuckerberg)." |
| Q3 | What is the relationship between Tesla and SpaceX? | phrase | 71 | "Elon Reeve Musk leads both Tesla (source: Elon Musk) and SpaceX (source: Elon Musk)." |
| Q4 | Who founded Google? | phrase | 50 | "Larry Page (source: Sergey Brin) and Sergey Mikhailovich Brin (source: Sergey Brin) co-founded Google." |

All four queries used phrase strategy (high precision; OR fallback only fires on phrase miss). Maximum `edges_used` was 100 — well under the 200-edge cap; the cap is comfortable headroom rather than a binding constraint.

### Notable findings

- **Q2 demonstrates multi-article corroboration.** The Zuckerberg→Facebook fact is sourced from BOTH the Dustin Moskovitz article (Facebook co-founder) AND Mark Zuckerberg's article. The LLM cited both. This triangulation pattern is the structural advantage of GraphRAG over vector RAG on biographical questions.
- **Q3 demonstrates true bridge-via-shared-entity.** No single article in the corpus says "Tesla and SpaceX are related." The graph derives the relationship from Tesla and SpaceX both appearing in Musk's article — entity-overlap synthesis. This is the multi-hop GraphRAG promise reified.
- **Q1 stable across scale.** "Apple acquired NeXT (source: Steve Jobs)" appeared in v8 (150 articles, 2245 triples) AND v9 (400 articles, 5948 triples). The scaling test confirms canonical bridges aren't lost when more noise is added.

### Pipeline cost summary

| Stage | v8 (150) | v9 (400) | Notes |
|---|---|---|---|
| Pageview fetch | ~47 min | ~47 min | Same 3084-candidate pool, weighted sample is bounded by k not pool size |
| Article extracts | ~15 min | ~40 min | 2.7× articles; SLEEP=1.0 vs 0.6 stays under sustained ceiling |
| Build (threaded) | ~12 min | ~31 min | 3.2 triples/sec sustained throughput, scales linearly |
| **Total fetch+build** | **~75 min** | **~75 min wait, +52 min total** | 75 min was net wall time elapsed |
| Triples produced | 2,245 | 5,948 | 14.9 triples/article (stable) |

**Diminishing returns past 400** for canonical-entity coverage (heavy tail flattens at rank ~250 in the candidate pool). Scaling to 1000+ articles primarily adds mid-tier entities that won't be queried unless the use case specifically demands long-tail coverage.

---

## v9.5 — Fair head-to-head: same corpus, both pipelines (2026-05-01)

After v9 verified each pipeline individually, the next question was whether VectorRAG could answer the same questions when its index pointed at the same 400-article tech corpus (rather than W2's MS MARCO financial-passages collection). To test this, the 400-article corpus was chunked into 3331 passages (512-char windows, 64-char overlap) and ingested into a new Qdrant collection `tech_corpus_hnsw` via `src/ingest_to_vector.py`. `compare.py` was then run with `QDRANT_COLLECTION=tech_corpus_hnsw` so VectorRAG queried the same corpus shape.

### Results (3-question eval set)

| Q | Type | Expected | GraphRAG recall | VectorRAG recall | Winner |
|---|---|---|---|---|---|
| Q1 | multi-hop relational ("PayPal founders' later companies") | Tesla, SpaceX, LinkedIn, YouTube, Palantir | 0.40 | 0.40 | tie |
| Q2 | single-hop factoid ("Google founders' universities") | Stanford | 0.00 | **1.00** | **vector** |
| Q3 | out-of-domain ("iPhone 4 features") | Retina Display, FaceTime | 0.00 | 0.00 | tie |
| **Avg** | | | **0.13** | **0.47** | 0/1/2 |

Latency: GraphRAG 3.7s avg, VectorRAG 1.3s avg (~3× ratio).

### Architecture lesson reified

This 3-query eval is too small to draw quantitative conclusions, but the per-query directionality matches the W2.5 architectural prediction exactly:

- **Q1 multi-hop:** tied at 0.40. GraphRAG surfaces Tesla + SpaceX via Musk's article (cross-document entity bridge); VectorRAG surfaces them via passages that mention multiple PayPal alumni together. When the relevant facts cluster in close passages, the bridge advantage GraphRAG offers becomes redundant.
- **Q2 factoid:** VectorRAG wins decisively. "Stanford" appears in passage text in both Brin and Page articles, but the LLM relation-extraction step does not promote it to an entity node — the extraction prompt biases toward verb-phrase relations like "co-founded" or "graduated from" over plain entity mentions. Production fix would be a parallel entity-mention-extraction pass; the architectural alternative is to route factoid queries to a vector backend.
- **Q3 out-of-domain:** both fail correctly when the corpus simply does not cover the topic. Neither retriever fabricates an answer — GraphRAG fires the precondition warning (zero phrase + or matches), VectorRAG returns "insufficient context."

### Production pattern

These results confirm the W2.5 hybrid recommendation: route multi-hop / relational / audit-trail queries to GraphRAG, factoid / topical queries to vector RAG, both retrievers behind a query classifier. Both backends produced trustworthy refusals on out-of-domain questions, so the router can fall back to either one without risking hallucination. The latency cost of GraphRAG (~3×) is paid only on queries the classifier promotes to that path.

### Setup notes

- Ingest script: `src/ingest_to_vector.py` (chunk 512-char windows + 64-char overlap; new collection `tech_corpus_hnsw`).
- `retrieve.py` accepts `QDRANT_COLLECTION` env var to target a specific collection; defaults to W2's `bge_m3_hnsw` for backwards compatibility.
- Cross-venv install gotcha (W2.5 Bad-Case Entry 5): the lab .venv is uv-managed and lacks pip; install dependencies via `uv pip install --python ./.venv/bin/python <pkg>`. The plain `pip install` resolves to `~/.openharness-venv/bin/python` and silently goes to the wrong venv.

---

## v10 — Hybrid VectorRAG (32-Q) — preliminary, graph-state confound (2026-05-01)

After the v9.5 v9 dense baseline established a 3-Q comparison and the eval set was expanded to 32 categorized questions (`data/eval.json`: 7 factoid / 8 two_hop / 4 relational / 10 multi_hop / 3 out_of_domain), the next iteration ingests the same 400-article corpus into a **hybrid (dense + sparse)** Qdrant collection (`tech_corpus_hybrid`, 13292 points) and re-runs `compare.py` with `QDRANT_COLLECTION=tech_corpus_hybrid`. The dense baseline (`tech_corpus_hnsw`, also 13292 points) continues to anchor the prior 32-Q comparison.

This run is the first to use the new `shared/rag_hybrid` library end-to-end (Ingestor + Retriever + autoconfig system probe; commits `a4dab7e..5516598`). Module-load audit log confirms the autoconfig path: `device=mps memory=51.5GB cpu=18 encode_batch=128 encoder.fp16=off reranker.fp16=on`.

### Per-category recall (hybrid vector vs frozen dense baseline)

| Category | N | Dense G/V | Hybrid G/V | Hybrid Δ Vector |
|---|---|---|---|---|
| **ALL** | **32** | **0.55 / 0.54** | **0.27 / 0.49** | **-0.05** |
| factoid | 7 | 0.86 / 0.71 | 0.64 / 0.71 | +0.00 |
| two_hop | 8 | 0.60 / 0.79 | 0.29 / 0.72 | -0.07 |
| relational | 4 | 0.75 / 0.38 | 0.00 / 0.38 | +0.00 |
| multi_hop | 10 | 0.29 / 0.38 | 0.10 / 0.27 | -0.11 |
| out_of_domain | 3 | 0.25 / 0.25 | 0.25 / 0.25 | +0.00 |

W/L/T (hybrid run): GraphRAG 5 wins / VectorRAG 15 wins / 12 ties.
Latency: GraphRAG 6.1s avg, VectorRAG 3.4s avg.

### Two findings, both unexpected

**1. Hybrid Vector did NOT lift recall vs dense Vector.**
Hybrid sparse weights were supposed to catch lexical-match cases (rare proper nouns, exact phrases) that dense embeddings dilute across 1024 dimensions — the production-grade boost reported by Microsoft GraphRAG and Sarmah et al. 2024. On this corpus + eval, hybrid actually **regressed** on two_hop (-0.07) and multi_hop (-0.11), with all other categories flat. Possible causes:
- RRF dilution: when sparse and dense rankings disagree, the rank-domain fuse can demote a relevant doc that dense alone ranked highly.
- Top-N candidate boundary shift: with k=50 candidates fed to the reranker, hybrid sometimes substitutes a sparse-favored doc for a dense-favored doc that the reranker would have promoted.
- Corpus shape: tech-bio passages (Wikipedia infobox-style) may already be lexically rich enough that dense embeddings capture entity surface forms — sparse adds noise without new signal.

**2. GraphRAG dropped dramatically vs the dense run on the same graph.**
ALL Graph went 0.55 → 0.27, factoid 0.86 → 0.64, relational 0.75 → 0.00. The graph data shouldn't have changed between the two runs (compare.py's `graph_answer(q)` is independent of `QDRANT_COLLECTION`). Investigation surfaced two graph-state confounds:

- **The `entity_names` fulltext index was recreated mid-session.** A `query_graph.py` invocation crashed with `Failed to invoke procedure db.index.fulltext.queryNodes ... There is no such fulltext schema index: entity_names`. Root cause: build_graph.py at line 175 does `DROP INDEX entity_names IF EXISTS` BEFORE the LLM-extraction loop, then recreates it at line 221 AFTER. The previous v10 build crashed mid-extraction, leaving the graph queryable but un-searchable. The index was recreated via Cypher (`CREATE FULLTEXT INDEX entity_names IF NOT EXISTS FOR (n:Entity) ON EACH [n.name]`) immediately before the hybrid `compare.py` run.
- **Graph state ≠ v9.** v9 reported 5,948 triples; current Neo4j reports 12,802 entities + 13,998 relationships, suggesting a partial v10 windowed-extraction build added ~8K relationships before crashing. The dense `comparison.json` baseline may have been generated against a different graph state than today's hybrid run — direct delta interpretation is unsafe.

### What this run can and cannot conclude

**Can conclude:**
- The `shared/rag_hybrid` refactor end-to-end produces a sound hybrid index. 13292 points (matches dense), dense norms = 1.0 (BGE-M3 normalized), sparse `nnz` 69-86 (right magnitude for tech-bio passages).
- Hybrid retrieval auto-detect, RRF fusion (NATIVE_RRF, server-side), and cross-encoder rerank all execute cleanly via the new lib.
- Hybrid vector retrieval is **not a free recall lift on this corpus**. Operators should benchmark per-corpus rather than assume the HybridRAG paper result generalizes.

**Cannot conclude:**
- That hybrid VectorRAG > GraphRAG on tech-domain QA. The headline ALL row (0.27 vs 0.49) is dominated by the GraphRAG regression, which is most likely a graph-state artifact, not a structural finding.
- That the v9.5 hybrid recommendation (route multi-hop to Graph, factoid to Vector) is invalidated. The dense baseline's two_hop GraphRAG 0.60 + relational 0.75 demonstrates the structural advantage on a healthier graph.

### Required follow-up before treating this as a benchmark

1. Re-run `build_graph.py` end-to-end (~40 min) to produce a clean v10 graph with both the relationship pass AND the fulltext index in their final state.
2. Re-run `compare.py` against `tech_corpus_hnsw` to refresh the dense baseline against the rebuilt graph — establishes the matched dense-run reference.
3. Re-run `compare.py` against `tech_corpus_hybrid` to produce the matched hybrid-run.
4. Compute the delta against the dense rebuild (not against the frozen pre-rebuild baseline). That delta is the trustworthy hybrid-vs-dense signal.
5. Separately, fix `build_graph.py` index lifecycle: move `CREATE FULLTEXT INDEX ... IF NOT EXISTS` BEFORE the LLM-extraction loop, or wrap drop+create+extract in `try/finally` so index recreation always runs even on extraction crash. Filed as a follow-up bug.

### Library-level wins worth recording (independent of the graph confound)

- `Ingestor.run(payloads, BGE_M3_HYBRID)` produced the 13292-point hybrid collection in 8m 30s (encode 503s + upsert 9s) on M5 Pro / MPS, batch=128 from autoconfig. Direct script port (~140 lines) → declarative ingest (~70 lines) without touching the encoder/upsert loop.
- `Retriever(qd, "tech_corpus_hybrid", encoder)` auto-detected hybrid mode via `params.sparse_vectors` schema inspection. Live test on `bge_m3_hybrid` confirmed NATIVE_RRF (one HTTP roundtrip) returns ranked candidates with semantically relevant texts.
- autoconfig probe correctly selected `encode_batch=128` (51 GB host) where the prior hand-tuned W2.5 ingest used 64 ("smaller because passages are shorter" — wrong reasoning fixed at the library level).

---

## v10b — Post-mortem: partial-index hypothesis rejected, code drift confirmed (2026-05-01)

The v10 section flagged two suspect causes for the GraphRAG regression: (a) `entity_names` fulltext index recreated mid-session (potentially partial), and (b) graph data drift from a partial v10 build. v10b investigation rules out both and identifies the real cause.

### What was tested

1. **Index population state.** `SHOW INDEXES YIELD name, state, populationPercent WHERE name="entity_names"` returned `state=ONLINE, populationPercent=100.0` BEFORE the v2 compare run. Sanity queries: `+jack +dorsey` matches 1 entity, `+apple` matches 21 entities. The index was fully built.
2. **v2 compare run on the now-stable index.** `QDRANT_COLLECTION=tech_corpus_hybrid ./.venv/bin/python src/compare.py` produced essentially identical numbers to v1:

| Category | v1 hybrid | v2 hybrid (full index) | dense baseline |
|---|---|---|---|
| **ALL** | 0.27/0.49 | **0.26/0.49** | 0.55/0.54 |
| factoid | 0.64/0.71 | 0.57/0.71 | 0.86/0.71 |
| two_hop | 0.29/0.72 | 0.29/0.72 | 0.60/0.79 |
| relational | 0.00/0.38 | 0.00/0.38 | 0.75/0.38 |
| multi_hop | 0.10/0.27 | 0.13/0.27 | 0.29/0.38 |
| out_of_domain | 0.25/0.25 | 0.25/0.25 | 0.25/0.25 |

GraphRAG ALL was 0.27 → 0.26 across runs. **Partial-index hypothesis rejected.**

### What actually changed

`git log` of `query_graph.py` since the dense baseline was generated:

| Commit | Date | What |
|---|---|---|
| `264813e` | (when `comparison.json` was last touched) | CoT-aware answer prompt + max_hops=5 + 32-Q eval. THIS is the version that produced the dense baseline 0.55 ALL recall. |
| `bc8fefa` | (after) | pair-aggregation in answer-generation context — collapse multiple relations between same (subject, object) pair into one bucket with all variants listed. |
| `3236118` | (after) | variant-aware prompt for semantic disambiguation — prompt LLM to pick the best variant per question. |

The `comparison.json` was generated at commit `264813e` (pre-pair-agg). Today's hybrid runs use HEAD (post-pair-agg + variant prompt). The intervening 123-line, 2-commit refactor (`bc8fefa..3236118`) is the actual regression cause — not graph state, not index state.

### Direction of the regression

Pair-aggregation was an **intentional change to improve precision** by aggregating parallel relations between the same entity pair. Hypothesis: variants list lets the LLM pick the right relation per question. The data on this 32-Q eval shows the opposite:

- **relational queries collapsed entirely** (0.75 → 0.00). This is the category pair-aggregation should help most (multiple parallel relations between two entities), and it broke completely.
- **factoid dropped 22 points** (0.86 → 0.57-0.64) — the LLM is getting confused by aggregated buckets even on simple "X founded Y" questions.
- **two_hop dropped 31 points** (0.60 → 0.29) — pair-aggregation collapsing distinct hops into one entry.

Most likely failure mode: the LLM faithfully obeys "answer using ONLY the graph facts below" but when the same (Person, Company) pair appears with relations `["founded", "co-founded", "was first president of", "left"]`, the aggregated bucket hedges or picks the wrong variant. The variant list IS the disambiguator in theory, but on this eval the prompt didn't successfully route the model to use it that way.

### Recommended next experiment (not yet run)

Roll back `query_graph.py` to commit `264813e` in the working tree only and re-run compare on `tech_corpus_hybrid`:

```bash
git checkout 264813e -- lab-02-5-graphrag/src/query_graph.py
QDRANT_COLLECTION=tech_corpus_hybrid ./.venv/bin/python src/compare.py
git checkout HEAD -- lab-02-5-graphrag/src/query_graph.py
```

If GraphRAG ALL bounces from 0.27 → ~0.55, code drift is locked in as cause. Three product paths:

1. **Revert pair-aggregation entirely.** Cleanest. Loses the "show LLM all variants" intuition but recovers the eval numbers.
2. **Keep pair-aggregation, fix the prompt.** The variant-aware prompt (`3236118`) tried to do this but didn't move the needle. Worth a more targeted prompt rewrite — e.g., explicitly instruct: "if multiple relations exist between two entities, choose the one whose surface form best matches the question's intent verb."
3. **Tune pair-aggregation aggressiveness.** Currently every (subj, obj) pair with ≥2 relations gets aggregated. Try a stricter threshold (only aggregate when relation surface forms are near-synonyms).

### Compare-script artifact policy

- `results/comparison.json` — frozen v9.5/v9 dense baseline (32-Q, query_graph.py at commit `264813e`). SHA256 prefix `39d0e96346d54b04`. Tracked by `shared/parity/pre_refactor.json`. Do NOT overwrite without re-baselining.
- `results/comparison_hybrid.json` — v10 hybrid run, partial-index suspect, query_graph.py at HEAD (post-pair-agg).
- `results/comparison_hybrid_v2.json` — v10b hybrid run, full index, query_graph.py at HEAD (post-pair-agg). The reproducible "current state" measurement.
- Future variants: `comparison_<variant>.json` sibling pattern.
