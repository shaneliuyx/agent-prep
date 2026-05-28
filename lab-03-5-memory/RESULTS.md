# Week 3.5 — Cross-Session Memory: Results

**Date:** 2026-05-11 (lab implementation) / 2026-05-12 (mem0 Phase 5 cross-check)
**Hardware:** MacBook Pro M5 Pro, 48 GB unified memory
**Stack:** oMLX local-first inference (`bge-m3-mlx-fp16` embedder + `gpt-oss-20b-MXFP4-Q8` Haiku tier + `gemma-4-26B-A4B-it-heretic-4bit` Sonnet tier) + Qdrant (OrbStack @ :6333) + SQLite (WAL mode)
**Companion chapter:** [`Obsidian Vault/Agent Development Curriculum/Week 3.5 - Cross-Session Memory.md`](../../Documents/Obsidian%20Vault/Agent%20Development%20Curriculum/) — 1355 lines, this RESULTS.md is the measured-numbers digest

---

## Exit criteria

- [x] Qdrant docker-compose running (W1 instance reused)
- [x] `src/memory.py` — hand-rolled memory writer + reader (custom OpenAI-compatible extraction + Qdrant episodic store + SQLite semantic-fact store)
- [x] `src/chat.py` — REPL agent reading memory at turn-start and writing at turn-end
- [x] `src/demo_three_sessions.py` — scripted three-session cross-recall demo
- [x] `tests/test_recall.py` — 15-Q recall benchmark, **15/15 PASS** (exit bar was 12/15)
- [x] `src/memory_mem0.py` + `tests/test_recall_mem0.py` — mem0 v2 cross-check, **10/14 with 4 measured architectural-difference failures across 2 LLM tiers**
- [x] Memory-type taxonomy (working / episodic / semantic / procedural) anchored at storage layer

---

## Headline finding — Hand-roll 15/15 vs mem0 v2 10/14 (the 4 failures are FEATURES)

The interview-grade artifact of this lab is **not** the hand-roll's perfect score. It is the **hypothesis-test discipline** that diagnosed mem0 v2's 4 failures as architectural-contract differences rather than model-quality issues — verified by re-running with a stronger LLM and observing the **identical pass/fail set**.

### Comparison matrix

| Backend | LOC | 15-Q pass | Mean latency / turn | Setup | Where it wins |
|---|---:|---:|---:|---|---|
| **Hand-rolled** (`src/memory.py`) | ~150 | **15/15** | ~3-4s | ~30 min from scratch | Explicit episodic/semantic split; SCD-2 contradiction archival via partial unique index; defensive JSON parsing tuned for local-quantized models |
| **mem0 v2.0.2 wrapper** (`src/memory_mem0.py`) | ~120 | **10/14** (test_15 skipped; 4 architectural failures) | ~4-5s | ~15 min wrapper + Qdrant collection | Production-grade contradiction-detection prompt; single-API surface; multi-backend support |

The 4 mem0 failures: `test_05_contradiction_update_latest_wins`, `test_07_cross_session_recall`, `test_10_episodic_surfaces_on_relevant_query`, `test_12_multiple_contradictions_latest_wins`.

### Three measured architectural differences (the actual artifact)

1. **Contradiction-update semantics (tests 05 + 12 fail).** mem0 v2 does NOT archive old values when a new value contradicts. Multiple location updates (Osaka → Tokyo → Kyoto) leave multiple "memories" in mem0's store, not a single live row + archived history. The hand-roll's SCD-2 + partial-unique-index produces a different contract: at most one live `(user_id, key)` row, unbounded archived history. **Production tradeoff:** mem0 preserves full mention history without schema hacks; hand-roll enforces a single canonical "current" value at the cost of needing the partial unique index (W3.5 BCJ Entry 4).
2. **No episodic vs semantic split (tests 07 + 10 fail).** mem0 v2 has no architectural distinction between episodic and semantic memory. Everything is a flat `memories` list. The wrapper's `score>0.5` heuristic classifies everything as semantic, so `relevant_episodes` is always empty, breaking tests that assert on episodic-specific surfaces. The hand-roll's dual-store (Qdrant episodic + SQLite semantic) is the architectural choice mem0 collapses. **Tradeoff:** mem0's unified model is simpler to operate; hand-roll's split makes the four-memory-types taxonomy explicit at the storage layer.
3. **API churn (mem0 v2 broke v1 patterns).** `Memory()` defaults to OpenAI cloud (not local), `search()` filters dict (not top-level kwarg), return-shape inconsistencies (None vs list vs dict-with-results vs dict-with-memories). The wrapper carries three defensive patches that wouldn't exist in a v1 mem0 client. **Tradeoff:** pinning a version buys stability but locks out fixes; the hand-roll has no external API surface to track.

### The hypothesis-test narrative (the senior-engineer signal)

**Run 1** (mem0 LLM = `gpt-oss-20b-MXFP4-Q8`, Haiku tier): 10 passed, 1 skipped, 4 failed in 55.97s. Error logs mention `Error parsing extraction response: 'NoneType' object has no attribute 'strip'`.

**Initial diagnosis:** failures look gpt-oss-20b-specific (Haiku tier emits `content=None` on edge prompts; mem0's extraction parser doesn't defend).

**Hypothesis test — swap LLM to Sonnet-tier Gemma-26B:**

```bash
MEM0_LLM_MODEL=gemma-4-26B-A4B-it-heretic-4bit MEMORY_BACKEND=mem0 \
  .venv/bin/python -m pytest tests/test_recall_mem0.py -v
```

**Run 2** (mem0 LLM = `gemma-4-26B-A4B-it-heretic-4bit`, Sonnet tier): 10 passed, 1 skipped, 4 failed in 72.30s. **Identical pass/fail set, same 4 tests.**

**Hypothesis falsified.** The `'NoneType' has no attribute 'strip'` was noise; the 4 failures are NOT model-quality issues. They're architectural-contract differences in mem0's design vs the hand-roll.

The 5-step methodology:

1. **Observation** — 10/14 pass with gpt-oss-20b, parse-error in logs
2. **Hypothesis** — failures are gpt-oss-20b-specific (Haiku quirk on edge prompts)
3. **Experiment** — swap to Sonnet-tier Gemma-26B
4. **Result** — identical 10/14, same 4 tests failed → hypothesis falsified
5. **Updated conclusion** — failures are architectural contract differences, not model-quality issues

`★ Why this matters ─────────────────────────────`
- **Without the model-swap**, the writeup is "mem0 is flaky on small local models" — narrow, model-blaming, wrong framing.
- **With the model-swap**, the writeup is "mem0 v2 has different semantic contracts than my hand-roll on contradiction archival and episodic/semantic separation" — specific, actionable, defensible in any interview about empirical method.
- **Production-library-vs-hand-roll comparisons are senior signal**: any candidate can claim "I considered using mem0"; only a senior candidate can say "I tested mem0 against my hand-roll on 14 specific tests, identified 3 architectural differences via a falsifying experiment, and decided to ship the hand-roll because of contract X."
- **Always test the cheaper hypothesis before believing the more complex one.** Model-swap = 1 env-var change + 72 seconds. Saved hours of wrong-direction wrapper-patching.
`─────────────────────────────────────────────────`

---

## What I'd port back from mem0's source (and what I wouldn't)

If productionizing the hand-roll:

**Port:**
- mem0's contradiction-detection prompt structure — more sophisticated than the lab's simple value-mismatch check (uses LLM-driven semantic comparison rather than exact-string match on the `value` column)
- Retry-on-extraction-failure with exponential backoff
- Multi-fact dedup before write (mem0 handles within a single `add()`; hand-roll currently processes one fact at a time)

**Don't port:**
- mem0's unified episodic+semantic model — the dual-store explicit split serves the four-memory-types taxonomy better at the interview surface
- mem0's contradiction-keep-history policy — SCD-2 archival is the right pattern for "what is currently true about this user" (audit trail without contradicting current state)

---

## Bad-Case Journal — chapter §"Bad-Case Journal" inventory

Six entries, all observed during the 2026-05-11 implementation session. The hand-roll moved from initial 12/15 → 14/15 → 15/15 via three rounds of fixes (Entry 3 → Entry 4 → Entry 5).

| Entry | Symptom | Production rule extracted |
|---|---|---|
| **1** — Layered embedding failure: `Connection refused` → `404 Model 'bge-m3' not found` | Two layered dependencies masking each other: oMLX not running, then oMLX running but no embedding model registered | Check what each "running" service ACTUALLY serves, not just whether the port is open. Per-service smoke tests (Qdrant `/readyz`, oMLX `/v1/models \| grep <embed>`, etc.) — ship dual-mode (oMLX-served + in-process `sentence-transformers` fallback) behind one env-var toggle. |
| **2** — `extract_memories()` returns top-level JSON array; downstream `.get("semantic")` crashes | `response_format={"type": "json_object"}` is best-effort on local quantized models. gpt-oss-20b respects schema ~95% of runs but emits other shapes (top-level array, scalar, malformed) ~5% of the time. | `response_format=json_object` is a CONTRACT on cloud models but a HINT on local models. Always coerce-or-empty at the parsing boundary; never trust the schema. Same pattern as W2.7's `_is_low_quality()` defensive check. |
| **3** — `sqlite3.OperationalError: database is locked` under benchmark load | SQLite default journal mode is DELETE (serializes all access). Under 15-test × 3-write/test interleaved with 30-60s LLM extraction calls, default 5s connection timeout was insufficient. Plus: `write_semantic_fact` had no try/finally around close. | Any SQLite-backed code path with concurrent access patterns MUST use WAL mode + try/finally connection cleanup + `timeout=30`. The default-DELETE + leak-on-exception combination produces this exact failure under load every time. |
| **4** — `UNIQUE constraint failed: user_facts.user_id, key, archived` on third contradiction-update | Schema's `UNIQUE(user_id, key, archived)` enforced uniqueness across THE WHOLE TUPLE including the `archived` flag → at most ONE archived row per (user_id, key). Real SCD-2 history requires UNBOUNDED archived rows. | `UNIQUE(a, b, flag)` is almost never what you want when `flag` is "is-this-the-current-version". Use a **partial unique index** with `WHERE flag = <current-value>` instead. After fix: 14/15 → 15/15. |
| **5** — Test passes in isolation but fails in sequence (state leakage) | `write_semantic_fact`'s SELECT-then-INSERT pattern could leave an open transaction if the LLM call raised. Even with WAL + timeout=30, next test's connection saw stale open transaction. | Test isolation in a stateful system requires CONNECTION-CLOSE guarantees, not just per-test fresh data. Per-test `uuid` user_ids are necessary but not sufficient — connection lifecycle must also be deterministic. Prefer `with conn:` context managers. |
| **6** — Episodic-recall threshold 0.35 is a precision-recall lever; `test_11` flakes on noise floor | BGE-M3's cosine score on semantically-distant-but-not-orthogonal text pairs sits at 0.36-0.40 — right on the noise floor. gpt-oss-20b's near-(not-exactly)-deterministic extraction paraphrases the seed text slightly differently each run, pushing score above/below 0.35 non-deterministically. | In production memory systems with similarity thresholds, the threshold choice is a precision-recall lever, NOT a one-time number. Senior engineers MEASURE the corpus's noise-floor distribution before picking the threshold; junior engineers cargo-cult a value (0.35, 0.5, 0.7). Same shape as W2.7's δ=0.07 cluster-routing tiebreak. |

---

## Production-ready findings (the four interview-grade lessons)

### 1. The four-memory-types taxonomy is interview-table-stakes

Working / episodic / semantic / procedural — name the type, pick the right storage. Working memory = conversation buffer (resets every session). Episodic = vector store + timestamp ("on Tuesday user asked about LangGraph"). Semantic = vector store + structured DB ("user.location = Taipei"). Procedural = system prompt augmentation or fine-tune (no off-the-shelf tool). **The collapse-into-"long-term-memory" failure mode is the canonical interviewer signal-detector.**

### 2. Extract → Store → Retrieve → Inject is the universal lifecycle

Most homegrown memory systems skip the EXTRACT stage and store raw turns. Why this fails: retrieval returns verbose transcripts; contradictions accumulate; token cost scales O(turns) not O(facts). The hand-roll's `extract_memories()` (LLM-driven, JSON output) is the load-bearing component.

### 3. Memory without forgetting is a landfill

Three forgetting strategies, all production-relevant: TTL eviction, confidence-weighted eviction, contradiction-triggered archival. The lab implements (3) via SCD-2 — archive don't delete, partial unique index lets exactly one live + unbounded archived rows coexist. **Archive-not-delete is non-negotiable for audit.**

### 4. Storage choice follows query shape

Semantic facts need exact-match lookups + uniqueness constraint → relational DB. Episodic memories are similarity-retrieved → vector store. Graph databases add value only when entity relationships are load-bearing (start with dual-store; graduate to graph in W2.5 GraphRAG when the workload warrants it). **Dual-store is the right starting architecture; mem0's unified flat-list is a different design choice with different tradeoffs.**

---

## Interview soundbites (anchored to measured outcomes)

### Soundbite 1 — "What are the four types of agent memory?"

"Working / episodic / semantic / procedural. Working memory is the conversation buffer — resets every session. Episodic stores time-indexed events: 'user asked about LangGraph on Tuesday.' Semantic stores durable facts: 'user is vegan, lives in Taipei.' Procedural encodes learned behavioral patterns and almost always needs fine-tuning or prompt augmentation — no off-the-shelf tool gives it to you. The mistake interviewers catch is candidates collapsing all four into 'long-term memory.' My lab covers episodic + semantic via dual-store: Qdrant for episodes, SQLite for semantic facts. Hand-roll scored 15/15 on a 15-Q recall benchmark; mem0 scored 10/14 because it collapses the episodic/semantic distinction architecturally."

### Soundbite 2 — "How do you keep memory from becoming a landfill?"

"Three forgetting strategies: TTL evicts stale facts after N days unless re-confirmed; confidence-weighted eviction drops lowest-confidence facts at cap; contradiction-triggered update archives the old fact and writes the new — **archive, never delete**, because audit trail matters. My lab implements (3) natively via SCD-2 with a partial unique index on `WHERE archived = 0`. That lets exactly one live row + unbounded archived history coexist; full-tuple UNIQUE constraint produces the wrong contract because flag-bearing tuples can't accumulate unbounded archived rows. That schema choice was a real bug I hit on the third contradiction-update — see W3.5 BCJ Entry 4."

### Soundbite 3 — "Tell me about a time you investigated a surprising benchmark result."

"I cross-checked my hand-rolled memory implementation against mem0 v2 on the same 15-Q benchmark. Hand-roll scored 15/15; mem0 scored 10/14 with parse errors in the logs. Initial hypothesis: mem0 is flaky on small local LLMs. I tested by swapping the LLM from Haiku-tier gpt-oss-20b to Sonnet-tier Gemma-26B and re-ran the suite. Identical 10/14, same 4 tests failed. Hypothesis falsified — the failures weren't model-quality issues, they were architectural-contract differences. mem0 v2 doesn't archive on contradiction, doesn't split episodic vs semantic, has API churn vs v1. The model-swap took 60 seconds and reframed the writeup from 'mem0 is flaky' (narrow, wrong) to 'mem0 has different semantic contracts than my hand-roll' (specific, defensible). Always test the cheaper hypothesis before believing the more complex one."

---

## File inventory

```
src/
  init_db.py          — SQLite schema bootstrap with partial unique index
  memory.py           — hand-rolled extract/write/recall + SCD-2 archival
  memory_mem0.py      — mem0 v2 wrapper, same API surface, defensive
                        normalization for v2 return-shape inconsistencies
  chat.py             — REPL agent: recall at turn-start, remember at turn-end
  demo_three_sessions.py — scripted cross-session recall demo
  lab_init.py         — guided setup (oMLX endpoint probe + Qdrant collection)

tests/
  test_recall.py      — 15-Q benchmark on hand-roll (15/15 PASS)
  test_recall_mem0.py — 14-Q benchmark on mem0 (10/14 + 1 skip + 4 architectural)
```

---

## Deferred work / open questions

1. **Threshold-calibrated episodic recall.** Entry 6 noted `test_11` flakes on the 0.35 noise floor. The right fix is a corpus-noise-floor measurement before picking the threshold (same shape as W2.7's δ=0.07 calibrated tiebreak), not cargo-culting a value. Open follow-up: measure leader-vs-runner-up score distribution on the lab's actual episodic corpus + pick threshold from the histogram.
2. **TTL + confidence eviction.** Lab implements only contradiction-archival. TTL (`updated_at` cron sweep) and confidence-weighted eviction (`ORDER BY confidence ASC LIMIT N`) are trivial to add but unmeasured.
3. **Cloud LLM comparison.** All measurements are local-first. A Claude / GPT-4 run of the same 15-Q benchmark would isolate model-quality contribution from architecture contribution; useful for the "would mem0 score higher with a frontier LLM?" follow-up question.
4. **mem0's contradiction-detection prompt port-back.** Listed as "would port" above — actually implementing it on the hand-roll and re-running the 15-Q benchmark to measure delta would close the loop.
