# W3.5.96 — results: a memory-augmented agent over GBrain MCP

All numbers are from a real run on this machine: GBrain 0.42.25.0 (Postgres engine
on Docker/OrbStack pgvector), embeddings + agent brain on local **oMLX :8000**
(`ollama:nomicai-modernbert-embed-base-bf16`, 768d), agent = **smolagents
`CodeAgent`** on `Qwen2.5-Coder-14B`. No cloud, no metered API.

## TL;DR

A **future agent** (smolagents, *not* Claude Code) uses GBrain as its memory layer
over **MCP**, end-to-end:

| stage | tool | result |
|-------|------|--------|
| connect | Python MCP client → `gbrain serve` (stdio) | **~70 tools** exposed; 5 needed present |
| read | `read_sources()` (local tool) | 2 raw files (email thread + transcript), any shape |
| extract | `extract_pages()` (oMLX, local tool) | raw → structured GBrain pages (JSON) |
| store | `put_page` (MCP) | **10 pages** written by the agent |
| wire | `gbrain extract links` | **11 typed edges** (self-wiring graph) |
| answer | `query` (MCP) | `deals/acme-seed` @ **score 0.93**, correctly cites `[[people/sam-okafor]]` anchoring |

**Headline finding:** the framework + MCP wiring is the easy part. The graph only
materializes if the **extraction prompt hard-mandates `[[wikilinks]]`** — without
that, the 14B writes prose and `extract links` produces **0 edges**; with it, **11**.
Graph quality = extraction quality, not plumbing.

## Pipeline (`src/ingest_agent.py`)

Idiomatic smolagents — **thin agent, fat tools**. The agent's own code is ~4 lines:
```python
raw = read_sources()
pages = extract_pages(raw)
for p in pages: put_page(slug=p["slug"], content=p["content"])
final_answer(query(query="Who is anchoring the acme-seed round and on what terms?"))
```
The hard work lives in tools: `read_sources` (file I/O), `extract_pages` (the oMLX
raw→structured call), and GBrain's MCP tools (`put_page`, `query`).

## Measured output

- **Probe** (`src/probe_mcp.py`): a plain Python MCP client spawned `gbrain serve`
  and listed ~70 tools; `put_page, add_link, add_timeline_entry, query, search` all present.
- **Run 1 (no wikilink mandate):** agent wrote 5 pages, but `extract links` →
  **`Links: created 0`** — pages were prose ("Alice Chen, founder of Acme AI"), no
  `[[wikilinks]]`. Self-wiring graph empty.
- **Run 2 (wikilink-mandate prompt + few-shot):** 10 pages, `extract links` →
  **`Links: created 11`**. `deals/acme-seed` now: *"Seed round for `[[companies/acme-ai]]`…
  `[[people/sam-okafor]]` is anchoring the remainder."*
- **query** "who is anchoring acme-seed?" → top hit `deals/acme-seed` score **0.926**, answer correct.
- **graph-query** `deals/acme-seed` → multi-hop typed traversal: `--invested_in->`,
  `--works_at->`, `--mentions->` across people/companies (depth 1–5).

## Run process

```bash
# prereqs: GBrain initialized (Postgres engine on Docker, oMLX embeddings) — see
# Week 3.5.96 Phase 1. Raw samples under ~/brain/sources/.
cp .env.example .env   # fill GBRAIN_DATABASE_URL, OLLAMA_*/LLM_* (oMLX), key
uv sync

uv run python src/probe_mcp.py      # 1. prove MCP client ↔ gbrain serve
uv run python src/ingest_agent.py   # 2. agent: read → extract → put_page → query

# wire + inspect the graph the agent built
export PATH="$HOME/.bun/bin:$PATH"
gbrain extract links --source db && gbrain stats
gbrain graph-query deals/acme-seed
```

## Bad-Case Journal (real, observed)

| # | symptom | root cause | fix |
|---|---------|-----------|-----|
| 1 | `extract links` → `Links: 0`; pages are prose | one-shot 14B extraction ignored the wikilink rule; wrote names as plain text + used `<!-- timeline -->` not `---` | extraction prompt: few-shot example WITH `[[wikilinks]]` + hard rule "a page with zero wikilinks is invalid" → Links 0→11 |
| 2 | fully-autonomous `CodeAgent` failed (naive regex extractor, hardcoded dates) + `InterpreterError: import pathlib not allowed` | asked a 14B to read files AND write an extractor AND compose markdown in one code loop; CodeAgent sandbox blocks `pathlib`/`json` | **thin agent, fat tools** — move file I/O + extraction into `@tool`s; agent only orchestrates |
| 3 | 14B confused / huge prompt | GBrain exposes ~70 MCP tools; loading all into the agent drowns it | filter `ToolCollection.from_mcp` to the ~5 needed (`put_page`, `query`, …) |
| 4 | `ModuleNotFoundError: mcpadapt` on `ToolCollection.from_mcp` | smolagents MCP support is an extra | depend on `smolagents[mcp]` |
| 5 | spawned `gbrain serve` can't reach oMLX/DB | an MCP server is a separate process — no shell-env inheritance | pass `GBRAIN_DATABASE_URL`/`OLLAMA_*` via `StdioServerParameters(env=…)` |
| 6 | oMLX emits `<code>` smolagents mis-parses (issue #1851) | oMLX has no native tool_calls | `CodeAgent(use_structured_outputs_internally=True)` |

## Interview soundbite (principle-level)

> The framework and MCP plumbing took an afternoon; the graph quality came down to
> one prompt rule. A capable-but-small local model will happily store well-written
> prose pages and silently produce a zero-edge "graph" — because it dropped the
> wikilinks the extractor needed. The lesson for wiring an LLM into a structured
> memory system: the *contract* the extraction prompt enforces (every entity is a
> typed link) is what makes the graph real, not the storage layer. Measure edges,
> not pages.

## Phase 6 — keyword vs vector vs hybrid-RRF benchmark (2026-06-04)

Corpus scaled 2 → 8 raw sources → **19 pages**. Reproduce:

```bash
# 1. ingest the expanded sources/ tree (Phase-3 agent)
cd ~/code/agent-prep/lab-03-5-96-gbrain && python3 src/ingest_agent.py

# 2. materialize the graph — wikilinks in put_page TEXT are not edges until this runs.
#    Pages written over MCP live in Postgres, so point extraction at the DB:
export PATH="$HOME/.bun/bin:$PATH" \
  GBRAIN_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/gbrain \
  OLLAMA_BASE_URL=http://localhost:8000/v1 OLLAMA_API_KEY=<key>
cd ~/brain && gbrain extract links --source db   # 11 → 45 links (created 34 from 19 pages)

# 3. benchmark at the ENGINE layer (CLI search==query, same handler — cannot A/B)
cd ~/code/agent-prep/lab-03-5-96-gbrain && bun src/bench_strategies.ts
```

Result (oMLX `nomicai-modernbert` 768-d, k=3):

| strategy | recall@3 | MRR | nDCG@3 |
|---|---|---|---|
| keyword (tsvector FTS) | 0.600 | 0.500 | 0.526 |
| **vector (HNSW)** | **0.900** | **0.917** | **0.900** |
| hybrid (RRF) | 0.900 | 0.783 | 0.813 |

**Finding (refutes the projected 83→95 RRF lift):** on this small, semantic-heavy
corpus pure **vector wins outright**. Keyword FTS missed all four purely-semantic
queries (no lexical overlap); RRF matched vector's recall but *lost* MRR/nDCG because
fusing the dead keyword arm demoted strong vector hits. RRF helps only when both
arms are competitive + complementary — not a free upgrade.

## Bad-Case Journal — Phase 6 additions

| # | symptom | root cause | fix |
|---|---------|-----------|-----|
| 7 | 19 pages but Links stuck at 11; ~68 wikilinks in text | self-wiring is a batch pass, not a `put_page` side-effect; MCP-written pages live only in PG, bare `extract links` wants a brain dir | `gbrain extract links --source db` → 45 links. Don't gate on `links_extracted_at` (file-source only) |
| 8 | `gbrain search` ≡ `gbrain query` byte-identical; A/B shows no lift | both CLI subcommands fall through to one handler (`cli.ts:771-772`); no pure-keyword CLI path | benchmark at engine layer via `eval.ts:runEval()` (`strategy: keyword/vector/hybrid`) |
| 9 | hybrid-RRF recall 0.90 but MRR 0.78 < pure vector 0.92 | keyword arm missed all semantic queries; RRF folded dead arm in, demoting good vector hits | prefer pure vector on small semantic corpora; RRF needs exact-term-heavy traffic to earn its arm |

## Phase 7 — Ground-Truth Hierarchy A/B (memory-os principle) (2026-06-05)

Leverages ClaudioDrews/memory-os's **Ground-Truth Hierarchy**: injected memory is
authoritative; the "memory-zero" anti-pattern re-establishes context every turn.
5-turn chained conversation over the live brain, chat via VibeProxy→Haiku 4.5.

```bash
export OPENROUTER_BASE_URL=http://localhost:8317/v1 OPENROUTER_API_KEY=vibeproxy \
  GBRAIN_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/gbrain \
  OLLAMA_BASE_URL=http://localhost:8000/v1 OLLAMA_API_KEY=<key>
python3 src/ground_truth_ab.py
```

| mode | retrievals | retr. ctx tok | LLM prompt tok |
|---|---|---|---|
| memory-zero | 5 | 11,167 | 22,254 |
| **ground-truth** | **1** | **2,233** | 23,001 |

**Finding — correctness, not just cost.** memory-zero FAILED 3/5 turns: Q2/Q4
coreference ("he", "that investor") had no antecedent; Q3 retrieval *drifted* to the
wrong cluster (Quanta/Ridgeline instead of Acme/Northstar) because the standalone
query had no anchor. Ground-truth nailed all 5 by resolving coreference from the
injected subgraph. Token nuance: ground-truth does NOT win on total LLM prompt
tokens (accumulating history ≈ repeated per-turn context); it wins on retrieval
(1 vs 5 calls, 80% fewer retrieval-context tokens) AND on answer correctness.

## Bad-Case Journal — Phase 7 additions

| # | symptom | root cause | fix |
|---|---------|-----------|-----|
| 10 | retrieved "context" is just slugs + a one-line snippet; LLM says "doesn't include the actual content" | `gbrain query --json` returns ranked snippets, not page bodies | fetch full bodies with `gbrain get <slug>` for the ranked slugs before injecting |
| 11 | model refuses ("I'm Claude Code, I can't help with questions about people"); system prompt ignored | VibeProxy injects a Claude-Code identity that overrides the `system` role | put the instruction + grounding in the USER message as a document-Q&A task; don't rely on `system` |

## Reconcile wired into the agent + large-corpus resumable ingest (2026-06-05)

**reconcile in ingest_agent.py.** `gbrain extract links --source db` is no longer a
manual step — `reconcile_graph()` runs it deterministically in `main()` AFTER the
agent's put_page writes and BEFORE the query (two-phase: WRITE → reconcile → READ),
so the query sees the wired graph (observed `backlink_boost` on the top hit). It is
infra, NOT an agent tool (must not depend on the LLM remembering). Why required:
MCP `put_page` is a *remote* caller → GBrain skips inline auto-link
(operations.ts `skipped:'remote'`), and inline auto-link only wires already-existing
targets, so forward refs in a single-pass ingest are dropped regardless.

**30s sandbox timeout fixed.** The ~60s oMLX extraction exceeded smolagents' 30s
per-step code limit → every step timed out and re-extracted (6 wasted steps).
Fix: warm `extract_pages` ONCE outside the sandbox (module-level cache) → ingest
now 1 step, 7.5s, 0 timeouts.

**Large-corpus variant `resumable_ingest.py`.** Warm-once doesn't scale (all files
in one prompt = context wall). Per-file streaming instead: extraction is DRIVER-side
(one small file, no sandbox), the AGENT writes each file's pages to a staging
namespace (`staging/<file>/<entity>`), checkpoint per file (`~/brain/.ingest_files.json`,
resumable), then a final `merge_pass()` consolidates cross-file entities, then
reconcile. Measured (8 files):

```
8 files, per-file, 0 timeouts
merge_pass: 5 promoted, 14 merged from 19 entities   # 14 entities spanned >1 file
reconcile graph: 23 pages, 63 links
query -> people/sam-okafor (top hit)
```

## Bad-Case Journal — ingest hardening additions

| # | symptom | root cause | fix |
|---|---------|-----------|-----|
| 12 | every agent step "Code execution exceeded 30 seconds", re-extracts, never finishes | ~60s oMLX extraction runs inside smolagents' 30s per-step sandbox; agent re-runs its whole block each step | warm `extract_pages` once outside the sandbox (cache); agent's call returns instantly → 1-step ingest |
| 13 | `UnicodeDecodeError` / stray 0-page units in checkpoint | `read_sources`/`_files` walked `.DS_Store` (binary) and `.omc-state/` (dotted dirs OMC wrote under sources) | skip any dotted PATH PART + catch `UnicodeDecodeError`, not just dotted filenames |
| 14 | large corpus can't be ingested in one shot | warm-once extraction concatenates all files → context-window wall, un-resumable | per-file streaming + per-file checkpoint + staging-namespace writes + final merge_pass (resumable_ingest.py) |

## resumable_ingest.py v2 — disk staging (embed each entity ONCE) (2026-06-05)

v1 staged into GBrain (one put_page per file-variant) → the store embedded every
variant then threw it away at merge: ~71% wasted embedding (46 staging embeds for
19 final pages on the 8-file run). Embedding is the throughput ceiling, so staging
must not touch the embedded store.

v2: stage on DISK (`~/brain/.ingest_stage/<file>.json`, no embedding), merge from
disk (driver-side), then the AGENT writes only the CANONICAL pages via put_page in
bounded batches — each entity embedded EXACTLY ONCE. Teaching point intact: the
agent still writes canonical pages + queries over MCP; only the throwaway
intermediate left the store.

```
staged 8 files to disk (no embedding)
merge_from_disk: 18 canonical (14 merged from >1 file)
write batch 1/2/3: 8 + 8 + 2 = 18 pages (embedded once each)
reconcile graph: 23 pages, 73 links
staging_in_db = 0          # staging never embedded
query -> people/sam-okafor
```

Embedding calls: **65 → 18** (the 46 wasted staging embeds eliminated). 0 timeouts.
Disk stage JSONs are the resumability artifact (re-run skips staged files;
`rm -rf ~/brain/.ingest_stage` to restart).

## Write checkpoint — resume re-embeds only un-written pages (2026-06-05)

The v2 disk-staging only checkpointed EXTRACTION (per file); the write phase was
idempotent-but-uncheckpointed, so a crash mid-write made a resumed run re-embed
ALL canonical pages — "embed once" held only for an uninterrupted run.

Fix: a write checkpoint `~/brain/.ingest_written.json` (written canonical slugs).
The write loop skips already-written slugs and marks each batch after it lands.
Oversized pages (> BIG_PAGE_CHARS) are written driver-side (no 30s sandbox), since
one such page's single embed could approach the agent's per-step limit.

Proven (stage JSONs present → extraction skipped both runs):
```
run #1: 18 canonical, 0 already written, 18 to write   → written.json = 18 slugs
run #2: 18 canonical, 18 already written (resume), 0 to write   → 0 write batches, 0 re-embeds
```

Resume model now: TWO disk checkpoints — extraction (`.ingest_stage/<file>.json`) +
writes (`.ingest_written.json`). Restart: `rm -rf ~/brain/.ingest_stage ~/brain/.ingest_written.json`.

## Big-file chunking — intra-file resume (2026-06-05)

Per-file staging assumed "one file fits the extract context." For a file too big,
`stage_all` now splits it into deterministic, line-aligned chunks `<file>#0/#1/…`
(by `CHUNK_CHARS`), each its own staging unit — so resume works at CHUNK
granularity using the SAME "skip if the JSON exists" check (no separate
line-offset bookkeeping). Cross-chunk entities are reunited by `merge_from_disk`
(same path as cross-file).

Tested with a synthetic 13.9 KB file (`_chunk_text` → 3 chunks [5937, 5941, 1984],
≤budget, lossless, line-aligned, deterministic):
```
RUN 1 (fresh):  staged #0:13, #1:13, #2:10 pages
delete #1 (simulate crash)
RUN 2 (resume): skip #0, staged #1, skip #2     # only the lost chunk re-extracts
entities spanning >1 chunk: 13 (sam-okafor/lin-zhao/… in chunks [0,1,2]) → merged
```

Design note: chose deterministic indexed chunks over a `{file: last_line}` offset —
chunk-file existence is an atomic checkpoint (can't be half-written), reuses the
per-file resume mechanism, and "many files" + "one huge file" collapse to one
concept (a chunk; a small file = 1 chunk).

## Verify-then-mark — write checkpoint can't claim a page that isn't there (2026-06-05)

The write checkpoint marked a whole batch right after `agent.run` returned —
trusting the agent loop. A silently-failed `put_page` (agent catches the error,
still returns) would mark a slug whose page never landed → resume skips it → page
lost. Tightened the invariant you'd want: **a slug is checkpointed IFF its page is
verified present in GBrain.**

`_verify_written(slugs)` queries `pages` (deleted_at IS NULL) — existence == a
successful embed+upsert. The write loop marks only the verified subset; un-landed
pages stay un-checkpointed and retry on resume. Gate tested:
```
_verify_written([real_slug, "people/__does_not_exist__"]) -> [real_slug]
```
Order + judgment now both correct: stage file exists but slug absent from
checkpoint ⇒ (re-)embed; only a confirmed-present page enters the checkpoint.

## Merge cache + the 4-layer recovery model (2026-06-05)

Resume re-read all stage chunks and RE-RAN the merge (its per-entity LLM calls)
every time — correct but wasteful. Cached `merge_from_disk` to
`~/brain/.ingest_merged.json`, keyed by a stage fingerprint (each chunk's
name+mtime+size). Unchanged staging → cache HIT, skip the re-merge; any
re-extracted chunk changes the fingerprint → MISS → re-merge + re-cache.
Tested (single-variant fixtures, 0 LLM): call1 caches, call2 HIT, stage-change → MISS.

**The pipeline is 4 derived layers, each a rebuildable cache of the one above:**
```
source files (ground truth, ~/brain/sources)   ← only true loss is losing this
  → stage chunks (.ingest_stage/<file>#<idx>.json)   ← gone? re-extract from source
    → merged canonical (.ingest_merged.json)         ← gone/stale? re-merge from stage
      → embedded (GBrain pages)                       ← gone? re-write from canonical
```
Correction to an earlier note: "stage gone + not embedded" is NOT a dead end — the
file is re-extracted from its source (`stage_all` already does this). The only
unrecoverable state is losing the source corpus itself. Each layer is regenerated
from the layer above; checkpoints just let resume skip the layers already built.

## Repeatable test suite for the resume machinery (2026-06-05)

`tests/test_resumable_ingest.py` (+ `tests/conftest.py` puts `src/` on path) — 9
tests, deterministic, no LLM (monkeypatched temp dirs + stubbed `_merge`), one
DB-gated integration test:
- chunker: small=1 chunk; big splits ≤budget, lossless, line-aligned, deterministic; never mid-line
- `_files` skips dotted path parts (`.DS_Store`, `.omc-state/` dirs)
- write checkpoint roundtrip + resume filter (writes only un-checkpointed slugs)
- oversized-page partition (BIG_PAGE_CHARS)
- merge: cross-chunk entities grouped + merged (LLM called once for the 2-variant), singletons pass through
- merge cache: HIT on unchanged stage, MISS after a chunk changes (fingerprint invalidation)
- `_verify_written` excludes a nonexistent slug (skip-gated on live gbrain-pg)

Run: `uv run --with pytest python -m pytest tests/test_resumable_ingest.py -v`
(plain `uv run pytest` resolves the wrong interpreter; `python -m pytest` in the
project venv has smolagents/mcp/openai). Result: **9 passed**.

## Phase 9 — corpus-adaptive search policy from a real golden eval set (2026-06-06)

A self-tuning **measure → decide → apply** loop. After every ingest, `src/policy_eval.ts`
scores keyword/vector/hybrid (engine-layer `runEval`) against a **fixed golden set of
real, labeled questions** (`data/golden_eval.json`), writes the winning arm to
`results/search_policy.json`, and `src/query_policy.ts` routes subsequent agent queries
through it (stock `gbrain query` is hybrid-only, so the actuator governs the agent's
path, not the CLI). `run_auto_eval()` (after reconcile in both ingest drivers) prefers
the golden eval; the known-item proxy `src/auto_eval.ts` is only a cold-start fallback.

**Golden set** (18 real questions, version-controlled): 12 `tenk` = W2.7's labeled
Berkshire-10-K questions (out-of-document refusal Qs dropped — no gold section); 6
`entity` = hand-written from the W3.5.96 fixtures. Gold = `expected_entities` substrings
a correct answer-bearing section must contain (grounding@K; no per-slug labels needed).

**Why NOT known-item proxy (the rejected first cut):** auto-generating queries from page
titles selected `keyword` on the 10-K (recall@3 0.72), but on the real golden questions
keyword grounds at **0.19** vs vector/hybrid **0.95** — the proxy picked the *worst* arm
for real queries. A policy is only as good as its eval-query representativeness.

**Drift experiment (live, isolated `gbrain_brk` DB) — grounding@5 on the 18 golden Qs:**

| phase | corpus | keyword | vector | hybrid | policy |
|---|---|---|---|---|---|
| A | 10-K only (44 pp) | 0.167 | **0.667** | 0.667 | `vector` |
| B | + entity data → mixed (59 pp) | 0.222 | 0.944 | **0.972** | `hybrid` |

Per-domain (Phase B): `g@5 tenk` keyword 0.250 / vector 0.958 / hybrid 0.958; `g@5
entity` keyword 0.167 / vector 0.917 / **hybrid 1.000**. The policy **changed on its own**
(vector → hybrid) when the ingest re-fired the eval — data-justified: entity proper-noun
questions revive the keyword arm, both arms become competitive, RRF earns its weight.
`tenk` grounding held at 0.958 across both phases (distractor growth didn't degrade 10-K
retrieval). Honest note: in Phase A vector and hybrid *tied* (0.667; entity data absent),
so v1 was a tie broken by order — the decisive signal is Phase B.

**vs W2.7** (same 10-K; W2.7 scored *answer quality* across *index types*): shapes
converge — dense vector strongest at 10-K factoid retrieval in both; grounding 0.96 ≫
W2.7's vector answer-judge 0.25 ⇒ the W2.7 bottleneck was generation, not retrieval.

Tests: `bun test tests/auto_eval.test.ts` → **15 passed**; `pytest tests/` → **24 passed**.
Bug found + fixed: CLI `gbrain put` titles from frontmatter, not `# heading` (exact recall
0.000 → 0.786). Isolated DB `gbrain_brk` now holds the mixed 59-page corpus.
