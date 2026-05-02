# GraphRAG Lab Results

## Hardware / Stack

| | |
|---|---|
| Machine | MacBook Pro M5 Pro, 48 GB unified memory |
| Extraction model | Gemma-4-26B via MLX (oMLX server) |
| Query / judge model | Gemma-4-26B via MLX |
| Neo4j | 5.x, APOC plugin |
| Corpus | 200 Wikipedia articles (tech founders / companies) |
| Graph build | ~38-40 min first run, ~38 min warm-cache rebuild |

---

## v12 Baseline (2026-05-01, pre-fix)

Graph built with:
- QID-keyed entity resolution (`apoc.merge.node`)
- Sliding-window extraction (WINDOW_CHARS=3000, OVERLAP=400)
- Fulltext index: `entity_names FOR (n:Entity) ON EACH [n.name]` ← **name only**
- Extraction prompt: active-voice rule, 10-15 triples/window, corporate + biographical categories

| Type | Avg recall_judge | N |
|---|---|---|
| factoid | 1.000 | 7 |
| two_hop | 0.750 | 8 |
| relational | 0.750 | 4 |
| multi_hop | 0.650 | 10 |
| out_of_domain | 1.000 | 3 |
| **Overall** | **0.797** | **32** |

### Known failures (baseline)

| Q | Type | Score | Root cause |
|---|---|---|---|
| What companies has Jeff Bezos founded? | two_hop | 0.0 | Fulltext index on `name` only; canonical node `name="Jeffrey Preston Bezos"` — common-name seed "Jeff Bezos" returned 12 edges (wrong entity) |
| Relationship between Tesla and SpaceX? | relational | 0.0 | Same alias gap: "Elon Musk" seed hits noise nodes before Tesla Inc. canonical node |
| Stanford alumni companies | multi_hop | 0.667 | Jensen Huang not found via "Nvidia" path; aliases not indexed |

---

## v12.1 Fix (2026-05-02)

### Changes

**1. Fulltext index extended to aliases**

```python
# Before (build_graph.py line 288-289)
"CREATE FULLTEXT INDEX entity_names IF NOT EXISTS "
"FOR (n:Entity) ON EACH [n.name]"

# After
"CREATE FULLTEXT INDEX entity_names IF NOT EXISTS "
"FOR (n:Entity) ON EACH [n.name, n.aliases]"
```

Root cause: QID-resolved canonical nodes store `name = first-seen surface form` (e.g. `"Jeffrey Preston Bezos"` from Wikipedia title). Common-name variants like `"Jeff Bezos"` accumulate in `aliases`. The old index only searched `name`, missing all alias-only matches. Neo4j indexes list properties element-by-element, so adding `n.aliases` makes every variant individually searchable via BM25.

Live DB fix applied before rebuild: `DROP INDEX entity_names` + recreate covering both properties. Confirmed with `answer("What companies has Jeff Bezos founded?")` returning Amazon, Blue Origin, Cadabra, Altos Labs (edges 12 → 400+).

**2. EXTRACT_SYSTEM prompt: three noise-category rules**

Audit of `data/wikidata_qid_cache.json` found 18,407 null-QID entries (53% of 34,637 total).

| Noise category | Count | Rule added |
|---|---|---|
| Lowercase sentence fragments | 5,656 | Proper-noun constraint: every entity must start with capital letter; reject monetary amounts, percentages, job titles as standalone entities, publication titles |
| Comma-separated entity lists | 516 | Decomposition rule: emit one triple per entity in a list — never use a comma-list as a single subject or object |
| Article-prefixed names | ~200 | Article-dropping rule: "New York Times" not "The New York Times" — prevents split nodes when QID is same but string is different |

These rules affect future corpus rebuilds; the live graph still contains pre-fix noise nodes (would require full rebuild to clean).

---

## v12.1 Results (2026-05-02)

| Type | v12 baseline | v12.1 | Delta |
|---|---|---|---|
| factoid | 1.000 | 1.000 | = |
| two_hop | 0.750 | 0.917 | **+0.167 ↑** |
| relational | 0.750 | 1.000 | **+0.250 ↑** |
| multi_hop | 0.650 | 0.612 | -0.038 ↓ |
| out_of_domain | 1.000 | 1.000 | = |
| **Overall** | **0.797** | **0.858** | **+0.061 ↑** |

### Per-question breakdown (v12.1)

| Question | Type | recall_judge | edges |
|---|---|---|---|
| Who founded Microsoft? | factoid | 1.0 | 400 |
| Who is the CEO of Apple? | factoid | 1.0 | 254 |
| Who founded Tesla? | factoid | 1.0 | 320 |
| Who founded SpaceX? | factoid | 1.0 | 319 |
| What university did Mark Zuckerberg attend? | factoid | 1.0 | 343 |
| Who founded Oracle Corporation? | factoid | 1.0 | 321 |
| Where did Bill Gates go to university? | factoid | 1.0 | 378 |
| What companies did Steve Jobs co-found? | two_hop | 0.667 | 277 |
| What companies has Elon Musk founded or led? | two_hop | 1.0 | 358 |
| What companies has Mark Zuckerberg founded? | two_hop | 1.0 | 343 |
| What companies has Jeff Bezos founded? | two_hop | **1.0** | **276** ← fixed |
| What companies did Peter Thiel co-found? | two_hop | 1.0 | 282 |
| What companies has Marc Andreessen founded? | two_hop | 0.667 | 261 |
| What companies did Sergey Brin co-found? | two_hop | 1.0 | 321 |
| What companies did Larry Page co-found? | two_hop | 1.0 | 282 |
| Relationship: Apple and NeXT? | relational | 1.0 | 502 |
| Relationship: Tesla and SpaceX? | relational | **1.0** | **443** ← fixed |
| Relationship: Apple and Pixar? | relational | 1.0 | 502 |
| Relationship: Microsoft and Paul Allen? | relational | 1.0 | 534 |
| Companies founded by PayPal founders? | multi_hop | 0.6 | 311 |
| Companies founded by Stanford alumni? | multi_hop | 0.333 | 340 |
| Apple acquisitions involving Steve Jobs? | multi_hop | 0.5 | 356 |
| Companies founded by Harvard dropouts? | multi_hop | 1.0 | 335 |
| Companies founded by people who later led Tesla? | multi_hop | 0.333 | 338 |
| YouTube co-founders from PayPal? | multi_hop | 0.75 | 268 |
| Palantir co-founders' other companies? | multi_hop | 1.0 | 397 |
| Andreessen Horowitz co-founders? | multi_hop | 0.5 | 273 |
| Reid Hoffman investments / co-foundations? | multi_hop | 0.5 | 257 |
| Stanford alumni tech founders? | multi_hop | 0.6 | 453 |
| Boiling point of helium? | out_of_domain | 1.0 | 112 |
| Who composed Symphony No. 9? | out_of_domain | 1.0 | 108 |
| Capital of France? | out_of_domain | 1.0 | 184 |

### Known remaining failures

| Q | Score | Root cause |
|---|---|---|
| Steve Jobs co-found? (NeXT, Pixar miss) | 0.667 | Content gap: NeXT/Pixar edge density low in 200-article corpus |
| Marc Andreessen (Loudcloud miss) | 0.667 | Content gap: Loudcloud not in Wikipedia seed articles |
| Stanford alumni companies (0.333) | 0.333 | BM25 noise regression: new alias tokens may have shifted seed matching; also Jensen Huang→Nvidia traversal missing specific "founded" edge |
| Tesla→people who later led Tesla (0.333) | 0.333 | Traversal direction: query requires reverse path (Tesla→CEO→company), needs `infer_reverse_edges.py` output to be queried |
| multi_hop ceiling | ~0.61 | Graph density + extraction fragmentation; full rebuild with v12.1 extraction rules expected to improve |

---

## Remaining TODOs

- [ ] Full corpus rebuild with v12.1 extraction prompt (cleans noise nodes, applies comma-list and article-dropping rules)
- [ ] Re-run eval after rebuild to confirm multi_hop recovers to ≥ 0.65
- [ ] Investigate BM25 noise issue on "Stanford alumni" query (seed entity noise nodes scoring above canonical nodes)
- [ ] Add QID-priority re-ranking in `fetch_subgraph` to prefer canonical nodes over noise-BM25 matches
