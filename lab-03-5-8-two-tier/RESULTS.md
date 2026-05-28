# Week 3.5.8 — Two-Tier Memory Architecture: Results

**Date:** 2026-05-28 (latest sync)
**Hardware:** MacBook Pro M5 Pro, 48 GB unified memory
**Stack:** `mathomhaus/guild` (Go MCP, brew install) + EverCore (Python + Postgres via docker-compose @ :1995) + Qdrant (OrbStack @ :6333) + oMLX local-first inference (@ :8000)
**Companion chapter:** [`Obsidian Vault/Agent Development Curriculum/Week 3.5.8 - Two-Tier Memory Architecture.md`](../../Documents/Obsidian%20Vault/Agent%20Development%20Curriculum/) — 7023 lines, this RESULTS.md is the measured-numbers digest

---

## Exit criteria

- [x] guild server running locally (`brew install mathomhaus/tap/guild`, `guild --version` passes)
- [x] EverCore docker-compose stack up (Postgres + EverCore service at `localhost:1995`)
- [x] `src/tiered_memory.py` (SQLite backend) + `src/tiered_memory_qdrant.py` (Qdrant backend) — both wrappers honor the same `imprint/query_context/consolidate` contract
- [x] `src/consolidation.py` — batch job with idempotency + ordering + failure isolation
- [x] `src/demo_two_agent_shared_knowledge.py` — agent A completes task in session 1, agent B in session 2 has cross-session context
- [x] **LongMemEval `oracle` subset comparison** — N=100, judge-controlled, vs published EverCore 83%
- [x] Six-model compose-LLM sweep on clean Qdrant collections
- [x] Architectural payoff named: data-shape-bound lifecycle, volume-buffered extraction, commitment vs hedge eval bias
- [x] **4-way benchmark on the W3.5 15-Q probe set** — measured 2026-05-28. Results in §"4-way benchmark" below.

---

## 4-way benchmark on W3.5 15-Q probe set (2026-05-28)

**Harness:** `scripts/run_four_way_bench.py` (retrieval-only, no LLM compose) → `results/four_way_bench_15q.json`. Probes lifted from `lab-03-5-memory/tests/test_recall.py` (7 single-agent recall) + 8 multi-agent-flavored variants (deployment, deadlines, dependencies, incidents, conventions, roadmap). Pass criterion: expected keyword (case-insensitive) appears in concat of retrieved memory text.

**Substitution:** EverCore replaced by Qdrant in the `evercore_only` slot per chapter §5.3.1 architectural-equivalence claim. Strict-EverCore version would require ~12hr wall (30-100s per imprint × 15 probes × consolidation pipeline); Qdrant variant runs in seconds and preserves the "semantic tier without operational tier" contract.

### Measured matrix (predicted vs actual)

| Backend | Predicted (chapter §5.2) | **Measured** | Wall | Delta |
|---|---:|---:|---:|---|
| `no_memory` | ~10% (assumed LLM compose) | **0/15 (0.0%)** | 0.0s | retrieval-only methodology; LLM-compose would add baseline hallucination |
| `guild_only` | ~55% aggregate | **15/15 (100.0%)** | 0.59s | +45 pts vs prediction |
| `semantic_only` (Qdrant) | ~60% aggregate | **13/15 (86.7%)** | 2.2s | +26 pts vs prediction |
| `two_tier` (full) | ~85% aggregate | **15/15 (100.0%)** | 60.4s | +15 pts vs prediction |

### Reading the surprise — the methodology gap

The chapter's predicted differentials assumed **LLM-compose downstream of retrieval**, scoring against a per-question gold answer string. This harness scores **keyword-in-retrieved-text** directly. Two consequences:

1. **`guild_only` aces the bench because raw scrolls contain expected keywords verbatim.** No semantic-gap to bridge for these 15 probes — scroll text was written WITH the keyword by construction. The chapter's predicted 55% guild_only assumed the LLM compose step would fail to extract the keyword from cluttered raw scroll text; bare keyword-search beats that prediction.
2. **`two_tier` doesn't visibly beat `guild_only` on this harness.** Both 100%. The chapter's predicted 20-30pt two-tier advantage requires the LLM compose step (which can pick clean semantic atoms over noisy raw scrolls). Without compose, raw-scroll union is unbeatable when the keyword is in the scroll.

The honest result: **this benchmark validates that all three memory backends store and retrieve the seeded facts, but doesn't differentiate them on retrieval QUALITY without an LLM-compose step.** Same lesson as W3.5.8 §5.3.5's commitment-bias finding — the instrument shapes the measured differential.

### Where `semantic_only` lost 2 probes (the architectural signal)

| Probe | Query | Expected | Why missed |
|---|---|---|---|
| P04 | "what's my hobby?" | `bicycle` | Seed framed bicycle as transport ("I ride my bicycle to work every day"), not hobby. bge-m3 cosine pulled other top-3 candidates. |
| P05 | "where do I live?" | `taipei` | Seed had 2 facts ("I'm vegan AND I live in Taipei"). Top-3 retrieval pulled OTHER location-flavored seeds (Osaka P01, Seoul P08) before this multi-fact one. |

Both misses are precision-recall-lever signals: bumping k from 3 to 5 would likely fix both. **This is the same noise-floor calibration question as W3.5 BCJ Entry 6** — the threshold/k choice is a tuning knob, not a fixed value. Senior-engineer move: measure the leader-vs-runner-up gap on representative queries before fixing k.

### Two-tier wall-time decomposition (60.4s)

The two-tier 60.4s wall is dominated by `consolidate()`'s LLM-summarize step:

| Stage | Count | Wall |
|---|---:|---|
| Guild post/claim/complete (15 quests) | 15 | <1s |
| Consolidate (15 scrolls → 12 imprints, 3 SKIPPED) | 15 | ~58s |
| Qdrant query (15 probes, k=3) | 15 | ~1s |
| Scroll fetch + union retrieval | 15 | <1s |

**The 3 skipped scrolls reveal the same scenario-binding issue as BCJ Entry 16** — `SUMMARIZE_PROMPT` is tech-biased, so conversational probes (location, diet, hobby) get filtered out at consolidation time. The two_tier still passes 15/15 because the guild scroll fallback catches what consolidation dropped. **This is the chapter §5.3.1 direct-imprint lesson visible empirically: the §3.x cascade is the wrong tool for conversational data; the guild-scroll-union fallback is what saves the score.**

`★ Insight ─────────────────────────────────────`
- **The chapter's predicted matrix isn't wrong — it's measured under a different methodology.** Predicted assumes LLM-compose downstream of retrieval, scoring against gold strings (canonical multi-question recall eval). Measured here is retrieval-only, keyword-in-text. Both are defensible methodologies; they answer different questions. Document the gap honestly rather than pick one and claim the other is "wrong."
- **The benchmark VALIDATES the architecture without DIFFERENTIATING it on this harness.** All three memory backends (guild_only / semantic_only / two_tier) successfully store + retrieve the seeded facts. The two-tier advantage shows up at LLM-compose time (semantic atoms outscore raw scrolls because the composer doesn't have to wade through scroll boilerplate). Adding an LLM-compose extension to the harness is the next-step measurement to make the differential visible.
- **The 3/15 SKIPPED at consolidation is the load-bearing finding.** Even on this small bench, the chapter §3.x SUMMARIZE_PROMPT's tech-bias dropped 20% of probes. Two-tier still scored 100% because of guild-scroll-union fallback — but a hypothetical "pure semantic" backend (Qdrant-only-no-guild-fallback) on the same corpus would inherit those skips. This confirms BCJ Entry 16's data-shape-bound finding from a DIFFERENT angle than the LongMemEval N=100 run.
`─────────────────────────────────────────────────`

---

## Headline finding — Commitment vs hedge (N=100, judge-controlled)

The interview-grade result of this lab is **not** the absolute accuracy number. It is the discovery that **LongMemEval structurally rewards commitment over calibration**, and the measurement design that confirmed this is what makes the finding defensible.

### The N=100 board (judge held constant: `claude-opus-4-7` across all answer-sets)

| Compose model | no-atomise | + read-time atomise | Δ vs EverCore (83%) |
|---|---:|---:|---:|
| **Qwen3.5-27B-Claude-Opus-distill** (4-bit MLX) | **77%** | **77%** | −6 pts |
| Claude Opus 4.7 (full precision, via proxy) | **68%** | **69%** | −15 pts |
| Claude Sonnet 4.6 (via proxy) | **60%** | — | −23 pts |

**Sources:** `results/longmemeval_qwen_distill_n100_rejudged.json` (77/100), `results/longmemeval_opus47proxy_n100_rerun_rejudged.json` (68/100), `results/longmemeval_opus47proxy_atomise_n100_rejudged.json` (69/100), `results/longmemeval_sonnet46proxy_n100.json` (60/100).

### Why a 4-bit 27B distillation out-scores Claude Opus 4.7 by 9 points

Three explanations were considered. Two were ruled out by direct test before the result was trusted:

1. **Judge confound (ruled out).** The eval runner defaults `MODEL_JUDGE` to the compose model — so initially each model graded its own answers. Re-judging all four answer-sets with one fixed judge moved every number by ≤1 point. The judge is **not** the explanation.
2. **Parser bug (ruled out).** Opus 4.7 is an extended-thinking model; a plausible bug is the `<answer>` extractor grabbing a CoT fragment. Categorising Opus 4.7's 32 wrong answers: 0 empty, 0 sentinel, 1 CoT-leak, 4 explicit abstentions — **27/32 are well-formed answers**. Parser is fine.
3. **Commit vs hedge (the real cause).** The 27 well-formed-but-wrong Opus 4.7 answers all share a *hedging* shape: "you mentioned X but the context doesn't confirm Y, so I can't calculate Z." LongMemEval scores against a concrete gold string — a hedge **never** matches a concrete gold answer, even when correct in spirit. Asymmetry:

| Situation | Commit | Hedge |
|---|---|---|
| Answer **is** in context (buried, needs reasoning) | often **CORRECT** | **INCORRECT** — never stated |
| Answer **genuinely not** in context | INCORRECT (guessed wrong) | INCORRECT ("can't tell") |

**Committing never scores worse than hedging, and scores better whenever the answer was derivable.** A model that always commits dominates a model that hedges, *on this eval*.

The Qwen distillation obeys `COMPOSE_SYSTEM`'s "default to answering, don't hedge" literally. Opus 4.7 — well-calibrated — overrides the prompt when it judges the context insufficient. Sonnet 4.6 confirms the trait is Claude-family-wide, not Opus-specific.

`★ Production caveat ─────────────────────────────`
- **The 77-vs-68 gap measures fit to commitment bias, NOT which model is safer to ship.** In production, a hedging model is the right answer for most assistant workflows — it does not confidently hallucinate "white Adidas sneakers" when the user never mentioned shoes. The distillation wins the benchmark; Opus 4.7 is arguably the better production choice.
- **Interview framing**: "A 4-bit local model beat frontier Opus on my LongMemEval run — and the reason is the benchmark, not the model. LongMemEval scores a confident wrong guess and an honest abstention identically, so it structurally rewards commitment. The lesson: a benchmark number is only as trustworthy as the behaviour it rewards."
- See chapter §5.3.5 for the full breakdown including Sonnet contamination-corrected numbers (60% raw → ~62% after dropping 2 proxy-injection misfires).
`─────────────────────────────────────────────────`

---

## N=20 six-model compose-LLM matrix (clean Qdrant collections)

The §5.3.2 matrix in the chapter was originally measured on a **contaminated** Qdrant collection — a reused `longmemeval-{qid}` namespace accumulated cross-run residue and scrambled numbers by up to ±40 pts. The clean re-runs (per-run unique namespaces; see chapter Production Considerations §"Measurement-harness discipline"):

| Rank | Compose + Judge model | Accuracy | Source file | Trade-off |
|---|---|---:|---|---|
| 🥇 1 | **Qwen3.5-27B-Claude-Opus-distill** v2 (dense, 4-bit) | **80%** | `longmemeval_clean_Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit_v2.json` | Best accuracy. Distillation transferred "commit-with-evidence" trait from Opus. |
| 🥇 1' | Qwen3.5-27B-Claude-Opus-distill v1 (no atomise) | 70% | `longmemeval_qwen27opus.json` | Pre-v2-prompt baseline |
| 🥈 2 | Qwen3.5-27B + read-time atomise (v3) | 75% | `longmemeval_qwen27opus_v3atomise.json` | +5 pts uniform lift (see §"Atomisation lifecycle" below) |
| 🥉 3 | Qwen3.5-35B-A3B-Opus-Reasoning-distill v4 (MoE, 4-bit) | 65% | `longmemeval_clean_MLX-Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled-4bit_v4.json` | Reasoning trace + distill, didn't help on this slice |
| 4 | Gemma-4-26B-A4B-heretic (dense, 4-bit) | 60% | `longmemeval_clean_gemma-4-26B-A4B-it-heretic-4bit.json` | Misses hardest counting + period-bounded extraction |
| 5 | Qwen3.6-35B-A3B (MoE, NVFP4) | 55% | `longmemeval_clean_Qwen3.6-35B-A3B-nvfp4.json` | MoE active-3B trades extraction for speed |
| 6 | Qwen3.6-27B-4bit (dense, 4-bit) | **50%** | `longmemeval_clean_Qwen3.6-27B-4bit.json` | Smaller dense baseline |
| ❌ | Qwen3.6-35B-A3B (MoE, UD-MLX-4bit) | 35% | `longmemeval_qwen35ud.json` | **Avoid** — UD quant catastrophically degrades MoE routing |
| ❌ | Qwen3.5-27B + constrained atomise (v9, top-K=5) | **45%** | `longmemeval_qwen27opus_v9constrained.json` | **−30 pts collapse** — see §"Constrained atomise collapse" below |
| — | DeepSeek-R1-Distill-Qwen-32B (MLX-4bit) | — | — | **Untestable** — MLX runtime emits raw BPE markers (Ġ/Ċ) |

**EverCore published baseline (full LongMemEval):** 83%. Best local on this 20-Q slice: **80%** (v2 baseline). Gap closed to −3 pts in clean conditions — much tighter than the −13 pts on the contaminated matrix.

### Slice + noise caveats

- The first 20 questions of `longmemeval_oracle.json` are all `temporal-reasoning`. N=20 with quantized 4-bit inference exhibits **±5pt stochastic drift** at temp=0 (MLX KV-cache + fp4 rounding non-determinism). Differences ≥10pts are signal; <5pts are at the noise edge.
- N=100 (60 temporal + 40 multi-session) is the more stable measurement; the matrix above is per-axis exploration, not the production-grade number.

---

## Ablation 1 — Write-time vs read-time atomisation lifecycle

**The §3.2.1 atomise primitive is the same code at both lifecycle positions. The position is load-bearing.**

| Lifecycle | Effect on conversational LongMemEval data | Mechanism |
|---|---|---|
| **Write-time** (consolidate → atomise → embed → retrieve → compose) | **Destroys signal**: 0/20 correct on first dry-run. SUMMARIZE_PROMPT is tech-biased; conversational details get paraphrased into tech-flavored language or SKIPped. | Lossy compression + 4 downstream stages = unrecoverable. Dropped fact at write is permanent. |
| **Read-time** (ingest → embed → retrieve → atomise → compose) | **+5 pts uniformly** across capability range (Qwen3.6-27B 60→65, Qwen-Opus 70→75). | Lossless augmentation (triples added alongside raw; raw stays as fallback). Question-conditioned. Errors recoverable in same LLM turn. |

**The fix that unlocked the benchmark: DIRECT-IMPRINT (bypass `consolidate()` entirely for LongMemEval haystacks).** Each haystack session imprinted as one Qdrant point preserving raw conversation text verbatim. Measured swing: **0/20 → 13/20** (Gemma 26B compose) and **0/20 → 14/20** (Qwen-Opus compose).

### Architectural conclusion

| Data shape | Lifecycle | Tier | Why |
|---|---|---|---|
| Structured durable facts (user preferences, ACID-eligible records) | **Write-time** atomise into typed schema | guild (operational) | Schema known; queries uniform; lossy compression acceptable |
| Conversational episodic data (sessions, dialogue, free-text events) | **Read-time** atomise from raw store | Qdrant (semantic) | Schema unknown in advance; queries heterogeneous; raw MUST survive |

The §3.x cascade is right for ONE scenario (guild task scrolls). Direct-imprint is right for ANOTHER (LongMemEval-shape conversational data). Importing the log-processing intuition ("compress at write to save on read") inverts when queries-per-memory ≫ 1. See chapter §5.3.3 for the five-reason decomposition (lossy vs lossless, early- vs late-binding, error compounding, amortization, schema imposer vs projection).

---

## Ablation 2 — Volume buffers extraction error (the constrained-atomise collapse)

The seductive next move after read-time atomise lifts both models +5 pts was: "make the extractor more focused — top-K=5 triples per session, question-conditioned, neutral framing." Tested before shipping. **Both models collapsed by ~the same magnitude despite a 40-pt baseline capability gap:**

| Config | Qwen3.6-27B | Qwen-Opus | Delta |
|---|---:|---:|---|
| Baseline (no atomise) | 30% | 70% | — |
| Unconstrained atomise (14-57 triples) + keep raw | 65% | 75% | **uniform +5** |
| Constrained K=5 + keep raw + neutral framing | **25%** | **45%** | **uniform −30 to −35** |

### Mechanism — Bayesian framing

- **MANY weak triples (K=14-57):** each is a low-authority hypothesis; composer's posterior is a mixture; errors dampen → Bayesian model averaging.
- **FEW strong triples (K=1-5):** each is a high-authority hypothesis; composer's posterior collapses early to whichever triple is most prominent; errors are unrecoverable → MAP selection.

The phase transition between regimes is sharp. The volume floor as a production guardrail:

```python
def safe_atomise(extractor, ctx, k_min=8):
    triples = extractor(ctx)
    if count_triples(triples) < k_min:
        return None  # fall back to raw-only
    return triples
```

**`k_min=8` is a hand-picked safe constant in the unmeasured `(1, 14]` gap, NOT a measured inflection point.** The §5.3.4 ablations sampled only two volume regimes; the real phase boundary is somewhere between, awaiting a fine-grained sweep. See chapter §5.3.4 "Foreshadowing — open production direction."

### Cross-stage symmetry

The constrained-atomise collapse at read time mechanistically duplicates the write-time SUMMARIZE_PROMPT failure (BCJ Entry 16): 1-3 compressed facts per scroll → composer anchors on the compressed (often wrong) facts → raw discarded → wrong answer. **Same failure mode at two different lifecycle stages.** Production rule that generalises: *low-volume high-confidence extractions poison downstream consumers, regardless of stage.*

---

## Ablation 3 — Commit-biased prompt is a small-model lift, atomise is uniform

| Configuration | Qwen3.6-27B | Qwen-Opus | Lift mechanism |
|---|---:|---:|---|
| Baseline | 30% | 70% | — |
| + commit-biased prompt | **60%** (+30) | 70% (+0) | **floor lift** — closes prompt-induced abstention bias |
| + commit prompt + read-time atomise | **65%** (+35) | **75%** (+5) | uniform +5 across capability |

**Two distinct lever mechanics, each load-bearing on its own:**

1. **Commit-biased prompt** raises the **floor** for capability-limited models. Replaces "if context lacks answer, abstain" with "default to committing; abstain only when context is unrelated to topic." +30 pts on Qwen3.6-27B closed 75% of the gap to Qwen-Opus with a 20-line prompt change. Cost: 1.5× latency.
2. **Read-time atomise** raises the **ceiling uniformly**. +5 pts on both models, regardless of base capability. Confirmed architectural primitive, not a small-model crutch. Cost: 6× latency (extra LLM call + bloated downstream context).

The uniform +5 across a 40-pt baseline gap is the architectural-primitive evidence: if atomise were a small-model crutch, it would not lift Opus at all.

### Pareto operating points

- **Best accuracy:** Qwen-Opus + atomise (75% / ~48 min wall on 20 Q) — production-grade local, too slow for interactive use.
- **Best speed/accuracy:** Qwen-Opus baseline (70% / ~8 min) — ship this for offline batches.
- **Cheap-to-tune surprise winner:** Qwen3.6-27B + commit prompt (60% / ~6 min) — 20 lines of prompt change recovered +30 pts on the smallest model.

---

## Bad-Case Journal — chapter §6 inventory

Twenty BCJ entries shipped (5 pre-scoped + 15 observed across measurement sessions 2026-05-14 → 2026-05-27). The observed entries are the load-bearing material for interview soundbites; pre-scoped entries are theory scaffolding pending validation.

### Highest-leverage observed entries (anchors for interview soundbites)

| Entry | Symptom | Production rule extracted |
|---|---|---|
| **8** — Reasoning model `summarize_scroll` returns `None`, `finish_reason=length` | gpt-oss-20b emits CoT into `reasoning_content`, exhausts `max_tokens=80`, never reaches final `content` | For reasoning models, `max_tokens` must budget for CoT + answer. Sniff the model class first. |
| **9** — EverCore `/memory/imprint` returns 404 | Wrapper written against hypothetical API surface; actual endpoints are `/api/v1/memories` + `/api/v1/memories/search` | Probe `/openapi.json` FIRST when wrapping a third-party HTTP service. Never hand-write client paths against assumed contracts. |
| **12** — Cross-agent semantic recall returns 0 memories | `agent_id` threaded into EverCore's `user_id` field, which is TENANT identity not per-persona label. Disjoint partitions. | Audit whether a memory store's primary-key field is "agent identity" or "tenant identity" — they are not interchangeable. Two-layer identity model needed. |
| **13** — `imprint` returns `accumulated`, `flush` returns `no_extraction`, search index stays empty | EverCore runs LLM-driven boundary detection; single-message imprints never produce a memcell | API status code is not enough — verify the **post-condition** (data is searchable) before declaring the call successful. Three-part imprint pattern: 2-turn synthetic conversation + unique session_id + immediate flush with same session_id. |
| **14** — Phase 8 dedup test: first scroll imprints 0 atoms because Qdrant has cross-test residue | `lab358_memories` collection shared across tests; new "fresh" scrolls correctly dedup-as-noop against prior runs | Dedup pipelines are STATEFUL across collection history. Tests must either scope to unique namespace or accept dedup-as-success. |
| **16** — §3.x consolidation destroys conversational details on LongMemEval | SUMMARIZE_PROMPT is scenario-bound to guild task scrolls (tech-biased few-shots); paraphrases or SKIPs conversational events | A memory ingest pipeline encodes a **data-shape commitment**. Applying to a different shape silently degrades. Measure cross-over with a known-answer eval before assuming transfer. |
| **17** — Atomise at wrong lifecycle stage destroys signal | Same primitive at write-time: lossy + error-compounding. Same primitive at read-time: lossless + question-conditioned. Lifecycle is a knob. | Most "compress at write" decisions in agent pipelines are leftover habits from log-processing. Agent memory is the opposite shape; importing the intuition costs accuracy. |
| **18** — Constrained atomise collapses both small AND large models by 30-35pts | Single high-confidence triple anchors composer regardless of model size; raw context below cannot override | Enforce K_min volume floor. ANY pipeline stage that produces compressed authoritative extractions risks poisoning the next stage when extractor accuracy < consumer trust threshold. |
| **19** — Sonnet env-shim abandoned: `OMLX_BASE_URL` redirected embeddings + proxy overwrote system role | Two coupled architecture mistakes: shared env across heterogeneous roles + OAuth-cloaking proxy overwrites `payload.system` for billing-fingerprint coherence | (a) Role-scope env vars when one var feeds clients with different shapes. (b) When ANY third-party proxy fronts an OpenAI-compat endpoint, run `system`-vs-`user-only` diagnostic before committing to a prompt design. |
| **20** — Phase 9 chapter code shipped with 3 latent bugs; integration test caught all | (1) `top_k` kwarg vs Protocol's `k`. (2) `dump_all_for_user` named in walkthrough's Protocol but not on real class. (3) `pytest-asyncio` fixture crossed task boundary on anyio TaskGroup. | (a) Chapter code wrapping a class IS the contract-vs-implementation seam. (b) "X is optional" promises in prose must be `hasattr`/try-except enforced. (c) Async fixtures + anyio TaskGroup require `async with` IN the test body, not the fixture. |

Full BCJ in [chapter §6](../../Documents/Obsidian%20Vault/Agent%20Development%20Curriculum/) (entries cited inline in the chapter's "Bad-Case Journal" section).

---

## Production-ready findings (the four interview-grade lessons)

### 1. Lifecycle is data-shape-bound

The two-tier architecture's real shape is data-lifecycle-driven, not technology-driven. Most "two-tier memory" articles split by storage engine (SQL + vector). The real split is **early-bound structured facts vs late-bound retrievable raw**. Same engine could serve both with different lifecycle policies; different engines could serve the same lifecycle. The discipline is the lifecycle choice, not the SQL-vs-vector choice.

### 2. Volume buffers extraction error

Compressed authoritative facts are unsafe at any model size when the extractor isn't near-perfect. The seductive "let the extractor pick the 3 most relevant facts" design pattern is unsafe by construction. Either ship raw only, OR ship many extractions for volume buffering (K_min ≈ 8 as a safe lower bound; real phase boundary unmeasured between 1 and 14).

### 3. Commitment bias makes the benchmark unreliable for production model selection

LongMemEval scores a confident wrong guess identically to an honest "I'm not sure." A 4-bit distillation beating Opus 4.7 by 9 points is **fit to the benchmark, not safety to ship.** Opus 4.7 hedges when the context is genuinely ambiguous — exactly the behaviour you want in a production assistant. The benchmark gap measures eval-specific commitment-bias fit; production preference often inverts.

### 4. Measurement-harness discipline is the difference between signal and noise

Five harness bugs scrambled the §5.3.2 matrix by up to ±40 pts before the clean re-runs landed (cross-test Qdrant residue under reused namespaces, transient proxy errors interpreted as model failures, judge confound where each model graded its own answers, parser swallowing CoT instead of answer, env-var shared across heterogeneous client shapes). The discipline rule: **when a measured result is surprising, suspect the instrument before believing the finding.** Three instrument faults checked in this lab before the commitment-bias result was trusted; commitment-bias survived all three, so it is real.

---

## Interview soundbites (anchored to measured outcomes)

### Soundbite 1 — "How would you architect memory for a multi-agent system?"

"I'd use a two-tier architecture: operational (atomic-claim, scroll handoff) plus semantic (consolidated facts, cross-session recall), connected by a periodic consolidation pipeline. The pattern maps to hippocampus-neocortex separation — fast-write coordination plus slow-write durable semantics, with consolidation as the engineering equivalent of REM sleep. I wired `mathomhaus/guild` (Go MCP) as operational + EverCore (Python/Postgres) as semantic + a Python batch job between them. On LongMemEval oracle subset N=100 with a judge-controlled head-to-head: my two-tier scored **77%** with a 4-bit distillation composer, **68%** with full-precision Opus 4.7, vs EverCore's published 83%. The −6 point gap with a 4-bit local stack is the architectural payoff; the gap to Opus is the commitment-bias finding I'll talk about separately."

### Soundbite 2 — "Tell me about a surprising benchmark result and how you investigated it."

"A 4-bit 27B distillation out-scored full-precision Opus 4.7 by 9 points on my LongMemEval N=100 run. That's the kind of result that should trigger suspicion, not celebration. I checked three instrument faults before believing it. First, the judge confound — my eval defaulted `MODEL_JUDGE` to the compose model, so each model was grading its own answers; I re-judged everything with a single fixed Opus judge and every number moved ≤1 point. Second, the parser — Opus is an extended-thinking model and I worried I was extracting CoT instead of the answer; I categorised Opus's 32 wrong answers and 27 were genuine well-formed answers, not parse errors. Third — the real cause — those 27 wrong answers were all **hedges**. Opus would say 'you mentioned X but the context doesn't confirm Y, so I can't calculate Z.' LongMemEval scores against a concrete gold string; a hedge never matches a concrete gold. The benchmark structurally rewards commitment over calibration. So the gap measures fit to commitment bias, NOT which model I'd ship to production. In production I'd ship Opus."

### Soundbite 3 — "What did you learn building a consolidation pipeline?"

"The biggest lesson was that **lifecycle position is load-bearing, separate from the primitive itself.** My atomisation code — extracting (subject, attribute, value) triples — is correct. When I applied it at WRITE time on LongMemEval haystacks, accuracy was 0/20: it lossy-compressed conversational details into tech-flavored summaries (my SUMMARIZE_PROMPT few-shots were scenario-bound to guild task scrolls). When I applied the same code at READ time after retrieval, accuracy went +5 pts uniformly on every compose model, regardless of base capability. Same code, opposite outcome — because write-time errors compound four stages downstream while read-time errors are recoverable in the same LLM turn, and write-time has no question to condition on while read-time does. The deeper invariant: most 'compress at write' decisions in agent pipelines are leftover habits from log-processing pipelines where queries are rare relative to ingest. Agent memory is the opposite shape — and importing the log-processing intuition costs accuracy. I now treat lifecycle as a knob, with structured durable facts going write-time into the operational tier and conversational episodic data going read-time over the semantic tier."

---

## Deferred work / open questions

1. **The W3.5 15-Q 4-way benchmark** — ✅ MEASURED 2026-05-28 (retrieval-only methodology). Results: 0% / 100% / 86.7% / 100%. See §"4-way benchmark on W3.5 15-Q probe set" above. **Next step:** add LLM-compose extension to the harness so the chapter's predicted ~55% / ~60% / ~85% differential becomes visible (without compose, raw-scroll keyword match is unbeatable on this corpus).
2. **Volume-floor inflection point** between K=1 (catastrophic) and K=14 (safe +5pt lift) is unmeasured. `K_min=8` is a conservative guess in the gap. Fine-grained sweep (volume = 2, 4, 6, 8, 10, 12 …) plotting downstream accuracy is the next data-collection step.
3. **Bucket-1 (user-preference) sub-category breakdown.** The aggregate accuracy hides per-category asymmetry. LongMemEval's 5 question types (single-session, multi-session, temporal-reasoning, knowledge-update, abstention) likely have very different commitment-bias profiles. Per-category accuracy table would refine the commitment-bias finding.
4. **HyperMem L3 tier comparison.** Chapter §"When to Add a Third Tier" sketches when the third tier earns its cost (multi-entity relational queries). W3.5.9's `lab-03-5-9-bench-hypergraph` is where the three-tier vs two-tier head-to-head lives. Not in scope here.
5. **Sonnet contamination correction.** N=100 Sonnet run scored 60% raw; 2/100 questions were proxy-injection misfires (Sonnet snapped to a Claude Code persona). Contamination-corrected ~62%. Cleaner Sonnet measurement with proxy fix is open work.

---

## File inventory

```
src/
  tiered_memory.py            — SQLite backend (W3.5 era + SCD-2)
  tiered_memory_qdrant.py     — Qdrant backend (drop-in via §6)
  consolidation.py            — batch job: scrolls → atoms → imprint
  dedup_synthesis.py          — Phase 8 dedup-and-synthesis
  judge_sonnet.py             — Claude-Sonnet judge harness (BCJ Entry 19 fix)
  memory_tools.py             — Phase 9 memory write/recall/list/forget MCP tools
  portability.py              — Phase 9 versioned export/import + migration chain
  replay.py                   — Phase 9 rejudge harness (no re-inference)
  run_longmemeval_slice.py    — slice-runner orchestrator (Phase 9 Step 1)
  demo_two_agent_shared_knowledge.py  — Phase 4 two-agent cross-session demo
  ...

scripts/
  build_slice.py              — sample N-question slice from longmemeval_oracle
  aggregate_results.py        — fold per-model runs into commitment-bias board
  run_longmemeval_oracle.py   — eval entry point

tests/
  test_phase9.py              — Phase 9 portability + replay + integration (15/15 PASS, BCJ Entry 20)
  ... (10 other test files)

results/
  longmemeval_qwen_distill_n100_rejudged.json        — 77/100 (centerpiece)
  longmemeval_opus47proxy_n100_rerun_rejudged.json   — 68/100
  longmemeval_sonnet46proxy_n100.json                — 60/100
  longmemeval_clean_*.json                            — clean N=20 six-model matrix (8 files)
  ... 44 result files total across N=20 ablations + N=100 commitment-bias runs

main.py                       — CLI entry point
SETUP.md                      — local-first stack bring-up (oMLX + Qdrant + EverCore + guild)
AGENTS.md                     — guild workflow primer for agent harnesses
pyproject.toml + uv.lock      — dependency graph (BCJ Entry 6+7 fixes encoded)
```
