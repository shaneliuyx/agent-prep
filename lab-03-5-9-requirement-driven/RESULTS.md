# Week 3.5.9 — Requirement-Driven Memory Architecture: Results

**Date:** 2026-06-02
**Hardware:** MacBook Pro M-series, unified memory
**Stack:** Qdrant (OrbStack @ :6333) + oMLX local embeddings (bge-m3 @ :8000) + VibeProxy LLM gateway (@ :8317 → Claude Haiku 4.5) + EverCore (@ :1995) + HyperMem L3 shim (@ :1996)
**Companion chapter:** [`Obsidian Vault/Agent Development Curriculum/Week 3.5.9 - Requirement-Driven Memory Architecture.md`](../../Documents/Obsidian%20Vault/Agent%20Development%20Curriculum/) — this RESULTS.md is the measured-numbers digest.

This lab compares seven memory backends on the LongMemEval slice (`data/longmemeval_slice_w358.json`, 20 quality questions: 10 multi-session, 10 knowledge-update): `qdrant` (W3.5.8 2-tier), `evercore`, `mem0`, `atomic_fact`, `hybrid` (router), `three_tier` (L1 guild + L2 Qdrant + L3 HyperMem), `ensemble` (RRF fusion of atomic_fact + mem0).

---

## Running the lab

### 1. Services (must be up before any run)

| Service | Port | Role | Bring-up |
|---|---|---|---|
| **Qdrant** | 6333 | vector store (all backends) | OrbStack/Docker: `docker run -p 6333:6333 qdrant/qdrant` (or OrbStack container) |
| **oMLX** | 8000 | local embeddings (`bge-m3`) + local extraction models | oMLX.app → serve on :8000 |
| **VibeProxy** | 8317 | Claude-subscription gateway → Haiku reader + Sonnet judge | start the VibeProxy app/CLI on :8317 |
| **EverCore** | 1995 | `evercore` backend only | `cd ~/code/EverOS/methods/EverCore && uv run python src/run.py --port 1995` (own `.env`, local gemma) |
| **HyperMem L3 shim** | 1996 | `three_tier` L3 only | start the HyperMem shim on :1996 |

Health check: `curl -s localhost:6333/collections >/dev/null && curl -s localhost:8000/v1/models >/dev/null && curl -s localhost:8317/v1/models >/dev/null && echo "core services up"`. You can skip EverCore/HyperMem with `--skip-evercore` and by not running the `three_tier` backend.

### 2. One-time setup

```bash
cd ~/code/agent-prep/lab-03-5-9-requirement-driven
uv sync                                   # install deps (Python >=3.12)
cp .env.example .env                      # then edit — see §3 (NOT committed; holds keys)
# models: bge-m3 + Qwen2.5-Coder-7B/14B in oMLX; BM25 for mem0 hybrid:
bash scripts/download_bm25_model.sh       # pre-fetch Qdrant/bm25 → ~/.cache/fastembed
uv run python scripts/build_slice.py                       # 2-axis w358 slice (default)
uv run python scripts/build_slice.py --all-axes --per-type 4 --include-abstention \
    --out data/longmemeval_slice_6axis.json                # 6-axis slice (32 Q)
```

### 3. `.env` (role-split; falls back to fully-local if the gateway vars are unset)

```bash
LLM_BASE_URL=http://localhost:8317/v1       # VibeProxy → Haiku: reader + complex jobs
MODEL_READER=claude-haiku-4-5-20251001      # the quality lever — keep capable
MODEL_HAIKU=claude-haiku-4-5-20251001
MODEL_EXTRACT=Qwen2.5-Coder-7B-Instruct-MLX-4bit   # 7B is default for dev AND final (14B ≈ 7B at full scale — a wash; 7B ~2× faster)
EMBED_BASE_URL=http://localhost:8000/v1     # oMLX (VibeProxy has no embed model)
MODEL_EMBED=bge-m3-mlx-fp16
FASTEMBED_CACHE_PATH=/abs/path/to/.cache/fastembed   # ABSOLUTE (no ~) — mem0 BM25
OMLX_BASE_URL=http://localhost:8000/v1      # fallback for both LLM+embed if gateway unset
```
Unset `LLM_BASE_URL`/`EMBED_BASE_URL` → every client falls back to oMLX = a fully-local run (no VibeProxy).

### 4. Run commands

```bash
# wiring smoke (1 question, 1 backend) — validate the chain end-to-end
uv run python -m src.run_longmemeval_slice --backend atomic_fact --smoke 1

# one backend on the 2-axis w358 slice
uv run python -m src.run_longmemeval_slice --backend mem0

# FULL 7-backend matrix on w358 (the headline comparison; ~85 min)
uv run python -m src.run_longmemeval_slice --backend all

# 6-axis universal-solution slice, one backend, isolated results dir
uv run python -m src.run_longmemeval_slice --backend atomic_fact \
    --slice data/longmemeval_slice_6axis.json --results-dir data/results_6axis

# fast read-side iteration: reuse persisted stores (no re-imprint, ~4 min vs ~25)
uv run python -m src.run_longmemeval_slice --backend atomic_fact \
    --slice data/longmemeval_slice_6axis.json --results-dir data/results_6axis --skip-imprint

# opt-in abstention gate (topic-presence); OFF by default
ABSTAIN_GATE=1 uv run python -m src.run_longmemeval_slice --backend atomic_fact \
    --slice data/longmemeval_slice_6axis.json --results-dir data/results_6axis_abstain --skip-imprint

# re-run a subset cleanly (per-run files never clobber; merge picks latest per cell)
uv run python -m src.run_longmemeval_slice --backend atomic_fact --qid 6aeb4375,d7c942c3 --run-tag fix
```

Useful flags: `--smoke N` (first N Q) · `--qid a,b,c` (specific questions) · `--backend <name|all>` · `--skip-evercore` · `--skip-imprint` (reuse stores) · `--slice PATH` · `--results-dir DIR` (isolate slices) · `--run-tag NAME`.

### 5. Aggregate → matrix

```bash
uv run python -m scripts.aggregate                              # merge data/results/*.jsonl → matrix
uv run python -m scripts.aggregate --results-dir data/results_6axis   # the 6-axis matrix
```
`aggregate.py` merges all `run_*.jsonl` in the dir **latest-per-(question,backend) cell** → `merged.jsonl` + prints the accuracy/wall matrix. Keep each slice in its OWN `--results-dir` so questions never mix across slices.

### 6. Gotchas (measured)

- **Re-running re-imprints (appends) unless you `--skip-imprint`** — for a fresh measurement, clear the backend's Qdrant collections first (`af_lme-{qid}-at` for atomic_fact), or use `--skip-imprint` to reuse them. Two driver processes writing the same `--run-tag` collide → corrupt the run file (use one tracked process per run).
- **7B vs 14B extraction:** 7B for dev (2.3× faster); switch `MODEL_EXTRACT` to the 14B for final count/multi-session numbers (7B under-extracts ~1 count question).
- **VibeProxy cools down (503) under volume** — the driver retries; the judge is non-fatal (rejudge later with `scripts/rejudge.py`).

---

## Full 20-Q × 7-backend matrix (2026-06-03) — THE headline result

**Harness:** `uv run python -m src.run_longmemeval_slice` + `scripts/aggregate.py` (20 quality questions: 10 multi-session + 10 knowledge-update; `0a995998` excluded as broken-gold, `synth_books_bought_v1` added). **~85 min** wall for the clean multi-session re-run. **Routing (role-split, see chapter §4.12):** reader = `claude-haiku-4-5` via VibeProxy (the quality lever); per-message extraction (atomic_fact/hybrid/three_tier L2/ensemble) = local Coder-14B; mem0/consolidation/dedup = VibeProxy Haiku + 503 retry; EverCore = local gemma; embeddings = local bge-m3; judge = `claude-sonnet-4-6` (held constant across all backends → fair comparison).

This matrix supersedes the earlier pre-fix run. Three corrections landed between them, all measured: (1) **knowledge-update latest-wins reader** (`[sN]` session-recency tags, "highest [sN] wins") — lifted the atomic-fact family KU 40-50% → 80-100%; (2) **transient-Qdrant retry** on `atomic_fact`'s upsert (`UnexpectedResponse()` was crashing cells, scored as wrong — see §"crash cells"); (3) **per-run result files + merge** (§"Measurement infrastructure") so the multi-session axis is real-measured, not reconstructed.

### Measured matrix (per question-type, % judged correct, n=10 per axis)

| Backend | knowledge-update | multi-session | **overall** |
|---|---:|---:|---:|
| **atomic_fact** | **100%** | 70% | **85%** 🥇 |
| ensemble | **100%** | 60% | 80% |
| mem0 | 70% | **80%** | 75% |
| hybrid | 80% | 70% | 75% |
| three_tier | 90% | 60% | 75% |
| evercore | 40% | 20% | 30% |
| qdrant | 0% | 0% | 0% |

### Reading the matrix

- **The hand-built 1-tier (atomic_fact) WINS overall at 85%** — above the mem0 SDK (75%). User-turn-only extraction (§4.10) + count-aware reader + KU latest-wins reader, all hand-built, beat the production library on this slice. The chapter thesis ("you can hand-build a competitive tier") is now stronger than stated: competitive → winning.
- **mem0 is strongest on multi-session (80%)** — its dense+BM25 hybrid (§4.11) surfaces the scattered entity mentions a cross-session count needs. It loses overall only because its flat store has no recency signal for KU (70%, where the atomic-fact family's `[sN]` tags win).
- **ensemble (RRF of atomic_fact+mem0) does NOT break the component ceiling — it underperforms its best member (80% vs atomic_fact 85%).** It ties at the KU ceiling (100%, both members strong + agreeing) but **drops to 60% on multi-session, below BOTH members** (af 70, mem0 80). This is a real, clean result (zero crashes that run) and the key finding — see the fusion analysis below.
- **three_tier 75%** (KU 90%, multi-session 60%). Its L2 (delegated to atomic-fact) carries it on KU; L3 HyperMem still doesn't fire (no multi-entity-intersection questions) — the §2.6 graduation-trigger null result holds.
- **qdrant = 0%** — architectural: its summarizer SKIPs conversational data (W3.5.8 BCJ Entry 16). Not a bug; the summarize-vs-atomic lesson at scale.
- **evercore = 30%** — its heavy pipeline underperforms on this slice's aggregation; `online` memorize mode may not aggregate cross-session counts the way these questions need.

`★ Insight ─────────────────────────────────────`
- **Ensemble below its best member REFUTES the naive "fusion ≥ best component" assumption — on read-then-reason tasks.** RRF maximizes recall@k of the *union*, but the downstream reader reasons over a *fixed top-k window*. Fusion can (a) demote a needle both members individually retained out of the window, (b) dilute a high-recall member's needles with a low-recall member's facts, (c) union the weaker member's distractors into the stronger member's clean set. Measured all three (below). Fusion helps pure retrieval; it can hurt read-then-count.
- **No single backend dominates all axes** (§2.2 thesis, measured): atomic_fact/ensemble top KU (100%), mem0 tops multi-session (80%). This is the argument for a *router* (pick the right backend per question-type) over a *blind ensemble* (fuse everything) — the router can route KU→atomic_fact, multi-session→mem0 and beat both the ensemble (80%) and any single backend.
- **The reader held constant (Haiku) is what makes the comparison fair.** The spread is now retrieval/extraction quality + read-time recency handling, not reader noise.
`─────────────────────────────────────────────────`

### Why ensemble loses on multi-session (clean, zero-crash run)

RRF fusion (`ensemble_memory.py`, K=60) of atomic_fact + mem0, then the same reader over the fused top-k. Three losses, three distinct mechanisms:

| qid | atomic_fact | mem0 | ensemble | mechanism |
|---|---|---|---|---|
| `dd2973ad` ("what time to bed", single-fact, k=5) | ✓ "2 AM" | ✓ "2 AM" | ✗ "I don't know" | **window truncation** — both kept the needle in their own top-5; RRF blended two lists, truncated to 5, the needle fell out of the fused window |
| `gpt4_59c863d7` (count model kits, gold 5) | ✗ found 3 | ✓ found 5 | ✗ found 3 | **recall dilution** — mem0 alone had all 5 in its 20 hits; fusion interleaved atomic_fact's 3 above mem0's complete set in the read window |
| `3a704032` (count plants, gold 3) | ✓ exactly 3 | ✗ 4 (distractor) | ✗ overcounts | **distractor injection** — fusion unions both fact sets, so mem0's "rose bush"/"basil" distractors polluted atomic_fact's clean set |

The throughline: **RRF is non-monotonic for read-then-reason.** A `correct=False, status=ok` ensemble cell where a member is `correct=True` is the signature — not a bug, a fusion-vs-reader-window mismatch.

### Caveats

- **`0a995998` excluded** from quality (broken gold: counts a sister-lent sweater as a store item; see §"probe-harness investigation"). Replaced by `synth_books_bought_v1` (gold=4, unambiguous).
- **N=20, ±~10pt noise** at this slice size; differences <2 questions are at the noise edge. Treat the *shape* (atomic-family + mem0 cluster ~75-85%, evercore low, qdrant zero) and the *ensemble-below-member finding* (3 clean losses, mechanistically explained) as signal.
- **judge held constant** (sonnet-4-6) across all backends, live and on the crash-cell + multi-session re-runs — no judge-drift between cells.
- **crash cells eliminated.** The earlier run had `UnexpectedResponse()` crashes (scored as wrong) that inverted the ensemble↔atomic_fact ranking; the retry fix (§"crash cells") + clean re-runs removed all of them (multi-session run: 0 crash cells; KU crash cells re-run to `status=ok`).

---

## Headline finding — the answer-quality wall is a CONJUNCTION, not a single knob

The multi-session counting questions ("how many items of clothing do I need to pick up or return?") were returning **"I don't know"** or undercounts across backends. A reader-probe harness (`src/probe_reader.py`) decomposed the failure against the persisted store of question `0a995998` (gold = 3) in **seconds per experiment** instead of 13-min full runs. Three independent levers each turned out to be **necessary and individually insufficient**:

| Lever | Layer | What it fixes | Evidence (probe) |
|---|---|---|---|
| **User-turn-only extraction** | imprint | Removes assistant-advice flood that buries user-action facts | all-turns store: 783 facts, top-40 = 0 needles. user-turn store: 88 facts (9× less), all 3 needles by k=40 |
| **Count-aware reader path** | read | Deeper retrieval (k=40) + enumerate-then-count prompt + token budget | baseline prompt → "I don't know"; count prompt → clean enumeration |
| **Capable reader model** | read | Reliable multi-item count + dedup over a noisy list | gemma-26B-A4B: flaky 1↔2 (non-deterministic at temp=0); Haiku 4.5: stable 3 |

**Only the full stack (user-turn store × count prompt × Haiku) answers correctly.** Drop any one and it fails: a strong reader on the all-turns store honestly returns "0" (nothing to count was retrieved); the count prompt on a weak reader flakes; the baseline prompt on a strong reader bails with "I don't know".

### Verified end-to-end (production path, role-split)

```
uv run python -m src.run_longmemeval_slice --backend atomic_fact --smoke 1
[atomic_fact] qid=0a995998 type=multi-session
  predicted: "Boots (return to Zara), boots (pick up new pair), dry cleaning (pick up)" → 3
  correct=True
  wall: imprint=76.2s  retrieve=0.03s  read=3.08s
```

Before this work the same question returned "I don't know" / undercounted. Extraction runs on the local 7B (per-message, user-turn-only); the reader is Haiku via VibeProxy. The imprint wall also dropped from **233s → 76s** (user-turn extraction does ~½ the LLM calls of all-turns).

---

## 6-axis universal-solution groundwork (2026-06-03)

**Motivation.** The prior W3.5.9 work tested only **2 of LongMemEval's 6 axes** (multi-session + knowledge-update, both current-value shapes). To build toward a **universal** memory solution, the slice was extended to all 6 axes + abstention (`data/longmemeval_slice_6axis.json`, **32 Q = 24 base + 8 abstention, 4/axis**; built by `scripts/build_slice.py --all-axes`). Stack: **7B extraction** (Qwen2.5-Coder-7B-Instruct-MLX-4bit) / **Haiku 4.5 reader** / **Sonnet 4.6 judge**.

> **Caveat throughout: n=4 per axis** — every ±1 question = 25%. Read the SHAPE and the per-component deltas, not exact ranks.

### Cumulative arc (base = 24 answerable questions; each a separate measured run)

| Stage | base acc | what changed |
|---|---:|---|
| baseline (user-turn-only extraction) | **37% (9/24)** | the prior single-tier reader on the 6-axis slice |
| + Component 2 (role-aware extraction) | **45% (11/24)** | extract from BOTH user+assistant turns, tag provenance role; parse TURNS not lines; chunk long turns (`max_chars=700`) |
| + Component 1 (read-side operators) | **62% (15/24)** | deeper retrieval for single-session (k=5→20) + temporal; temporal-ORDERING operator (earliest-`[sN]`-wins) |
| + preference operator | **79% (19/24)** | single-session-preference: generate preference-aligned recommendation instead of extract |

**Per-axis FINAL (base, the +preference run):** knowledge-update **4/4**, multi-session **3/4**, single-session-assistant **2/4**, single-session-preference **4/4**, single-session-user **3/4**, temporal-reasoning **3/4**.

> n=4 per axis. The +preference run measured **79% (19/24)**; the subsequent per-operator seq fix lifted temporal-reasoning 3/4→4/4, raising base to **83% (20/24)**. Both confirmed.

### Component highlights (each MEASURED, each debugged to root)

- **Component 2 debug saga (the real lesson).** v1 REGRESSED everything — a **bug, not a tradeoff**: multi-line assistant turns mis-tagged (measured **283 user / 5 assistant** facts; assistant text leaked into the user store, so the role filter stopped de-flooding). Fixed by parsing **TURNS not lines**. That exposed a 2nd failure: a **2473-char generated children's book** (single-session-assistant qid `89527b6b`) sent as ONE extraction call returned **0 facts**. Fixed by chunking long turns: **0 → 47 facts** including *"The Plesiosaur has a blue scaly body"* → answered *"Blue."* correctly. Tuned chunk size **350→700** (recovers the detail with 4 chunks vs 10; imprint wall **73.9s → 46.1s, −37%**).
- **Component 1 (read-side operators).** temporal-reasoning **0/4 → 3/4** via the ordering operator (*"which did I do first, A or B?"* → tag facts `[sN]`, lowest session = earliest). The 1 miss (`gpt4_2487a7cb`) was the **`[sN]` LIMIT** (both events same session); fixed by per-operator `seq` → **temporal 3/4 → 4/4, base 79% → 83%** (see "Recency signal" subsection).
- **Preference operator.** single-session-preference **1/4 → 4/4** by reframing extract → generate-aligned-with-stated-preferences (the gold is the user's latent preference the answer should match).
- **single-session-assistant 0 → 2/4.** Depth recovered a rank-6 answer; the remaining miss (`e9327a54`) is a **retrieval-ranking miss** (answer entity not in top-40), NOT depth — a different problem.

### Recency signal: `[sN]` → `seq`

**Mechanism.** `[sN]` = session index (`idx` from `enumerate(haystack_sessions)`), stamped at imprint into each fact's payload as `quest_id = "{qid}-sess{idx}"`. The reader helper `_session_recency` parses `sess(\d+)` back at query time. Key properties: (a) **per-session** — every fact from session `idx` shares the same `N`; (b) **not a wall-clock timestamp** — it is an ordinal assigned by `enumerate`; (c) **intra-session blind** — two facts from the same session have identical `N`.

**Finding (measured).** The slice carries real per-session timestamps in `haystack_dates` that were **unused** by the original imprint loop. Of 18 questions with `haystack_dates`: **3 of 18 (~17%)** have index order that does NOT match chronological order. Example `gpt4_2655b836`: s0=17:50, s1=14:47, s2=17:15 — s0 is indexed first but happened last. `[sN]` is doubly limited: (1) intra-session — all turns share one N; (2) cross-session — index assumes chronological, ~17% violate it.

**Fix implemented.** Three coordinated changes:
1. `_chrono_sessions(q)` in `run_longmemeval_slice.py` sorts sessions by parsed `haystack_dates` before the imprint loop (falls back to index order when dates absent). Verified: `gpt4_2655b836` reorders to 14:47 → 17:15 → 17:50.
2. `AtomicFactMemory` stamps a monotonic per-fact `seq` counter (`self._seq`, incremented per extracted fact across all per-session imprint calls). Because sessions are now in chronological order and turns/chunks in dialogue order, `seq` strictly increases with true time at both granularities.
3. Reader prefers `seq` over `[sN]` at query time (`meta.get("seq", _session_recency(meta))`), falling back to `[sN]` for backends without `seq` (mem0/evercore/qdrant).

**Result (measured, atomic_fact, 7B, n=4/axis):**

| recency signal | knowledge-update | temporal-reasoning |
|---|---:|---:|
| `[sN]` session-index (baseline) | 4/4 | 3/4 |
| global `seq` (per-fact) | 3/4 | 4/4 |
| **per-operator** (`seq` for ordering, `[sN]` for KU) | **4/4** | **4/4** |

`seq` fixed the intra-session temporal miss `gpt4_2487a7cb` (temporal-reasoning **3/4 → 4/4**). But global `seq` regressed KU question `6a1eabeb` (**4/4 → 3/4**): a "personal best 5K time" superlative (gold 25:50). Fine per-fact `seq` re-exposed a re-mentioned old value (27:12); coarse `[sN]` collapses the session so "latest session = 25:50" wins correctly. **Per-operator fix:** use `seq` for temporal-ordering; keep `[sN]` for KU latest-wins / superlatives. Chrono-sort applies to both. Result: temporal 4/4 AND KU 4/4 — **net +1 over the `[sN]` baseline**.

Overall 6-axis base impact: temporal-reasoning 3/4 → 4/4 lifts base from **19/24 (79%) → 20/24 (83%)**.

**The lesson:** recency-signal granularity is **per-operator, not global**. Fine `seq` suits ordering (needs intra-session order); coarse `[sN]` suits current-value / superlative latest-wins. Same data, two operators, two recency resolutions — extends the operator-routing thesis to the recency signal.

**Recency-signal fidelity tiers:**

| Tier | Signal | Granularity | Residual limitation |
|---|---|---|---|
| **1 — `[sN]` session-index** | `enumerate` order at imprint | per-session | cross-session only; assumes index=chrono (~17% violate); intra-session blind; correct for KU latest-wins / superlatives |
| **2 — `seq` insert-sequence** | monotonic counter at imprint | per-fact, dialogue order | per-OPERATOR: use for ordering, NOT for KU latest-wins; needs sessions date-sorted; dialogue order ≠ event order |
| **3 — extracted event-time** | LLM temporal-expression extraction | per-fact, event order | highest fidelity; LLM-dependent; normalization pipeline required |

The `gpt4_2487a7cb` intra-session miss is resolved by tier 2 (validated: now correct). The tier-3 "said-late-about-early" ceiling still stands for a different class of temporal question not present in this slice.

### Abstention (8 questions; the hard, risky axis)

- **Binary grounding gate** (GROUNDED/UNGROUNDED) — **REJECTED**: answerable **19/24 → 10/24** (−9 over-refusal), abstention 3/8 → 6/8; **net NEGATIVE**. It conflated "answer not verbatim" with "unanswerable."
- **Topic-presence gate** (research-driven reframe: *"is the question's SUBJECT present at all?"* + few-shot, biased to PRESENT) — answerable **18/24 (−1 only)**, abstention **3/8 → 5/8**, total **22 → 23/32 net +1**. Over-refusal collapsed **−9 → −1**. **Shipped OPT-IN** (`ABSTAIN_GATE=1`), OFF by default (marginal +1 at n=32 + a per-question LLM call). Research basis: AbstentionBench (arXiv 2506.09038), Decision-aware Answer/Ask/Abstain (arXiv 2604.04565) — prompt-only abstention helps but has a ceiling.

### Tooling / method

- **7B vs 14B — full-scale 6-axis result: a WASH. 7B is the default for dev AND final.** The earlier 3-question A/B (n=3 multi-session) showed 14B 2/3 vs 7B 1/3 and prompted the "use 14B for final count/multi-session" guidance. The full 32-Q 6-axis run (n=4/axis) refutes it: 14B ≈ 7B overall (~82% vs 83%), with axis wins trading off (14B gains single-session-assistant 2/4→4/4; loses multi-session 3/4→1/3 and temporal-reasoning 4/4→3/4). 14B is ~2× slower (imprint ~85-90s/Q vs ~30-46s for 7B). **Extraction is commodity, confirmed at full scale — doubling the extraction model nets ~zero and trades questions across axes rather than improving all axes.** Keep `MODEL_EXTRACT` at 7B.
- **`--skip-imprint` flag added.** Reuse persisted stores for read-only re-runs — **4.5 min vs ~25 min**. The reason read-side operators could be iterated fast.
- **Per-run results + `scripts/aggregate.py`** (latest-per-cell merge) and `--results-dir` isolate slices so the 6-axis runs never mix with the w358 matrix.

### 14B-final run: per-axis matrix + root cause

Full 6-axis single-model run (`MODEL_EXTRACT=Qwen2.5-Coder-14B-Instruct-MLX-4bit`), same pipeline as 7B, crash-free after the broadened-retry fix (see below). `0a995998` excluded as broken-gold, n=4/axis:

| axis | 7B (83% base) | 14B | delta |
|---|---:|---:|---|
| knowledge-update | 4/4 | 4/4 | = |
| single-session-assistant | 2/4 | **4/4** | +2 (14B wins — more complete extraction over long assistant narrative) |
| single-session-preference | 4/4 | 4/4 | = |
| single-session-user | 3/4 | 3/4 | = |
| multi-session | 3/4 | **1/3** | −2 (14B loses — see root cause) |
| temporal-reasoning | 4/4 | **3/4** | −1 (14B loses 1) |
| **base** | **20/24 (83%)** | **~19/23 (~82%)** | ≈ wash |

**Root cause of 14B multi-session + temporal losses — over-extraction, not weaker reasoning.** 14B over-extracts: it generates many near-duplicate facts per item (measured: the F-15 model kit restated ~5 ways; 14B store held ~445 facts vs far fewer for 7B). Under a fixed retrieval top-k (40), the redundant duplicates crowd the window — a rare item drops below the cutoff. Concrete: on the "how many model kits" count question (gold 5), the Spitfire fact EXISTS in the 14B store (user-tagged) but ranks below #40, crowded out by ~5 F-15 restatements + multiple B-29/Camaro — the reader enumerated 4, not 5. Same mechanism for the temporal "first issue" miss (an extra issue-fact outranked the target). This is the same flooding mechanism as the original user-turn-only finding (783→88 facts), but the noise source moved from assistant ADVICE to MODEL VERBOSITY (duplicate user-tagged facts) — so the role filter cannot de-flood it. **Sharpened lesson: extraction VOLUME must be controlled; more extraction can HURT fixed-window recall of rare items.**

**Fix analysis (documented; not implemented — 7B default retained):**

- **Write-time fact dedup — REJECTED.** Can eliminate genuinely-distinct facts: "started X" vs "finished X" are different; old vs new value of the same attribute is different — merge would break knowledge-update, temporal, and status questions. Also destructive (violates evidence-before-belief — same mistake as user-turn-only discard). Do not dedup at write.
- **Read-time MMR / diversity rerank (or per-entity read cap) — the right shape.** Non-destructive: store is untouched, policy is per-query and reversible; runtime cost negligible (in-memory rerank over already-retrieved embeddings, no extra LLM/embed call). Costs: one tuning knob (λ) that is hard to validate at n=4; must be scoped per-operator (helps count, would hurt KU-latest-wins / lookup); mild mis-rank risk (can demote a similar-but-distinct needle, reversibly). Not built — the dilution shows mainly under 14B's verbosity and 7B does not over-extract this badly. Documented as the known remedy; build if a larger slice shows persistent count-dilution on 7B. Throughline: **preserve at write, diversify/resolve at read.**

**Broadened-retry fix (code, shipped).** Transient Qdrant `UnexpectedResponse()` was crashing cells at un-retried paths (collection-setup, query) — the earlier retry only covered upsert. A single `_qd_retry` helper now wraps all four client calls (`get_collections` / `create_collection` / `upsert` / `query_points`) with 3× backoff. The 14B run that exposed this crashed 2 cells mid-run (`75832dbd`, `gpt4_70e84552_abs`); both recovered after the fix. The run is now crash-free.

`★ Insight ─────────────────────────────────────`
- **The arc 37% → 79% → 83% came from READ-side operators + role-aware extraction + per-operator recency granularity, not bigger models or more storage.** Every lift was a read-time assembly or write-time provenance fix. The final +4pp (79→83%) came from using `seq` for temporal-ordering and `[sN]` for KU latest-wins — same data, same model, different recency resolution per operator.
- **Every gain was debugged to root** (Component 2 v1 regression → mis-tag → chunking → tuning; `seq` regression on `6a1eabeb` → per-operator fix). Verified, not assumed.
- **Abstention is a prompt-FRAMING problem** (topic-presence vs answer-groundedness), not a strictness-tuning one. Same wiring, same model — only the epistemic framing changed −9 → −1 over-refusal.
- **Recency-signal granularity is per-operator.** Fine `seq` helps ordering (intra-session order matters); coarse `[sN]` is correct for current-value / superlative latest-wins. The operator-routing thesis extends to the recency signal itself.
`─────────────────────────────────────────────────`

---

## Architecture — role-split (forced by the metered proxy)

An initial **"all-Haiku via VibeProxy"** migration was tried (one LLM model for every role, embeddings local). It did not survive contact with volume: VibeProxy is a **metered Claude-subscription gateway**, and the high-volume **per-message atomic-fact extraction** (~thousands of calls across the slice) **COOLED IT DOWN (HTTP 503)**. The resolution is a **role-split**: high-volume commodity roles run on a LOCAL unmetered model, low-volume capability-critical roles stay on the gateway.

| Role | Endpoint | Model |
|---|---|---|
| per-message atomic-fact extraction (atomic_fact, hybrid, three_tier L2, ensemble) | oMLX :8000 (LOCAL) | Qwen2.5-Coder-7B-Instruct-MLX-4bit |
| mem0 fact extraction + update | VibeProxy :8317 | claude-haiku-4-5 |
| consolidation summarize/atomize + dedup | VibeProxy :8317 | claude-haiku-4-5 |
| reader (single-shot answer) | VibeProxy :8317 | claude-haiku-4-5 |
| **embeddings** | **oMLX :8000 (LOCAL)** | **bge-m3-mlx-fp16 (1024-dim)** |
| judge (independent) | VibeProxy :8317 | claude-sonnet-4-6 |
| EverCore backend (own service) | local :1995 | gemma (own `.env`) |

The split rationale: extraction is the **highest-volume role** (one call per user turn × every question × every backend in the atomic-fact family), so it moved to a LOCAL commodity 7B — unmetered, and confirmed a wash vs 14B at full scale (see 14B-final result). The capability-critical **READER stayed on Haiku** (the quality lever the probe-harness isolated). The **judge stays on Sonnet 4.6**, deliberately independent of the model under test — you should not grade a system with the same model it runs on.

**Endpoint split, not a global swap.** Each module now reads a chat endpoint (`LLM_BASE_URL`, fallback `OMLX_BASE_URL`) separately from an embeddings endpoint (`EMBED_BASE_URL`, fallback `OMLX_BASE_URL`). VibeProxy hosts no embedding model, so embeddings keep their own local endpoint. With both env vars unset, every client falls back to oMLX — restoring a fully-local config unchanged. This avoids BCJ W3.5.8 Entry 19 (one `OMLX_BASE_URL` feeding heterogeneous client shapes redirected embeddings to a chat-only proxy).

Files touched: `atomic_fact_memory.py` (chat→local 7B, embed→oMLX, user-turn filter), `mem0_backend_adapter.py` (llm→VibeProxy, embedder→oMLX), `consolidation.py` (3 clients + `import json` fix), `dedup_synthesis.py`, `tiered_memory_qdrant.py` (embed→oMLX), `run_longmemeval_slice.py` (reader→VibeProxy + count-aware path), `.env`.

`★ Cost note ─────────────────────────────────────`
Extraction on the local 7B is ~30-46s/Q imprint; the 14B is ~2× slower for a wash (see the 14B-final result). Moving extraction off the metered gateway is what made the full 7-backend matrix runnable without tripping VibeProxy's 503 cooldown. Embeddings staying local keeps the per-fact embed cost negligible (0.23s).
`─────────────────────────────────────────────────`

---

## The probe-harness investigation (qid 0a995998, gold = 3)

`src/probe_reader.py` replays **retrieve → read → judge** against a persisted store with no re-imprint, because imprint writes to a deterministic address (`user_id = lme-{qid}-{backend[:2]}`, Qdrant collection keyed off it). Read-side knobs become seconds-cheap to sweep.

### Diagnosis chain

| Probe | Question | Result |
|---|---|---|
| `--grep "pick up,return,blazer,boots,sweater"` | Are the needles in the store? | **Yes** — all 3 cleanly extracted |
| `--show-facts --top-k 40` | Do they rank into the window? | **No** — top-40 = 100% generic decluttering advice, 0 needles |
| `--sweep 5,10,20,40,80` | Does raising k fix it? | **No** — all "I don't know" |
| scroll whole collection | Why? | 801 unique facts, ~798 are **assistant advice** (`"use a garment bag for a blazer"`) drowning 3 user-action facts |
| `--reimprint user` vs `all` | Does dropping assistant turns help? | 783 → **88 facts** (9×), 233s → **24s** (10×); boots needle rises into top-20 |
| `--user-id probe-user-... --top-k 40` | Are all needles now retrievable? | **Yes** — boots #13, dry-cleaning #23, sweater #37 |
| `--prompt reader_count.txt` | Does an enumerate-then-count prompt help? | "I don't know" → `boots, dry cleaning → 2` (gemma) / `→ 3` (Haiku) |

### The model A/B (identical store, prompt, k, tokens — only the reader changed)

| Reader | Output | Verdict |
|---|---|---|
| gemma-4-26B-A4B (local oMLX, 4-bit MoE) | `boots → ANSWER: 1` (and `2` on a rerun) | ❌ flaky, non-deterministic at temp=0 |
| **Claude Haiku 4.5 (VibeProxy)** | `Dry cleaning, Boots… → 3` (stable across 2 runs) | ✅ correct |

### Haiku isolation matrix (which levers still matter with a strong reader)

| Store | Prompt | k | Result |
|---|---|---:|---|
| all-turns (noisy) | baseline | 40 | ❌ "I don't know" |
| all-turns | count | 40 | ❌ "0" (top-40 has no needles → honest zero) |
| all-turns | count | 5 | ❌ "0" |
| user-mode | baseline | 40 | ❌ bailed (*but its output named boots* — needles WERE retrieved) |
| **user-mode** | **count** | **40** | ✅ **3** |

`★ Insight ─────────────────────────────────────`
- **Control for reader capability before crediting a retrieval/prompt fix.** The "I don't know" failures were partly the *reader*, not only retrieval. Tuning retrieval against a flaky reader would have chased the wrong layer. The A/B (same everything, swap only the model) is what isolated it.
- **A strong reader on bad retrieval fails HONESTLY ("0"), which is the tell.** Haiku returning "0" on the all-turns store is correct behavior — top-40 genuinely held no concrete items. That distinguishes a retrieval problem (fix the store) from a reasoning problem (fix the reader/prompt).
- **Assistant advice is not the user's memory.** Per-message extraction over assistant turns floods the store with high-cosine-similarity distractors ("organize clothes by type") that outrank the rare user-action facts a count question depends on. Extracting only from user turns is a memory-architecture decision (memory *of the user*), not a hack — and it cut the store 9× while lifting recall.
`─────────────────────────────────────────────────`

### Honest caveat — this question's gold is debatable

Strict store items = navy blazer (dry cleaner) + boots (Zara) = **2**. Gold = **3**: it also counts a green sweater *lent to the user's sister* (not a store). Haiku reaches 3 by counting boots' two actions (return old + pick up new) + dry cleaning. The benchmark answer is matched, but "3" is path-dependent — a strict reader answering "2" is arguably more correct than the gold. **The levers are validated as general improvements; we did not tune the prompt to force "3" on this one noisy label** (that would be fitting the metric, not building a sound counter).

---

## 6-backend smoke + VibeProxy cloak / role fixes (2026-06-02)

Running all 6 backends on qid 0a995998 surfaced two regressions from the all-Haiku migration, both rooted in the **VibeProxy system-role cloak** (W3.5.8 BCJ Entry 19 recurrence): VibeProxy :8317 routes through Claude Code's interactive system prompt, so any LLM call carrying a real `system` role gets a Claude-Code refusal ("I'm Claude Code… handle your dry cleaning yourself") instead of the structured output the caller expects.

| Backend | Before fixes | After fixes | Items found | Note |
|---|---:|---:|---|---|
| `qdrant` | 0 | **0** | none | architectural — summarizer SKIPs conversational data (W3.5.8 Entry 16), not cloak |
| `evercore` | 2 | **2** | blazer + boots | strict-correct (excludes sister-lent sweater) |
| `mem0` | **0 (broken)** | **2** | blazer + boots | **FIXED** — system→user shim; hits 0 → 19 |
| `atomic_fact` | 3 | **2** | dry cleaning + boots | explicitly excludes sweater as non-store item |
| `hybrid` | 2 | **2** | boots + dry cleaning | strict-correct |
| `three_tier` | **1 (broken)** | **2** | dry cleaning + boots | **FIXED** — L2 atomic facts; hits 18 blobs → 40 facts |

**The fixes (use user role, not system role):**
- `consolidation.py` — folded `SUMMARIZE_PROMPT` + `ATOMIZE_PROMPT` from `system` into the `user` turn.
- `mem0_backend_adapter.py` — monkeypatch `OpenAILLM.generate_response` to fold any `system` message into the user turn (Mem0 builds its extraction call with a system role internally; one shim at its sole LLM chokepoint, no SDK fork). mem0 went 0 → 2 facts on a unit probe, 0 → 19 hits in the run.
- `three_tier_memory.py` — L2 now delegates to `AtomicFactMemory` (per-fact, user-turn-filtered, `af_{user_id}` collection) instead of the inherited `TieredMemory` raw-scroll embed. Cause of the `1`: it stored 3 whole-session ~4 KB blobs in the shared `lab358_memories`; the reader truncates each memory to 400 chars, so only the session opening (blazer) survived.

`★ Insight ─────────────────────────────────────`
- **Five functional backends converging on "2" is the system out-reasoning its own gold.** They retrieve the real items, enumerate, and correctly exclude the sister-lent sweater ("not a store item"). Gold = 3 counts it. The pipeline is sound; the label is debatable. We did NOT tune toward 3 (that would be fitting one noisy gold).
- **"All-LLM via VibeProxy" is only achievable with user-role messages.** The cloak fires on a real `system` role, period. The conversational reader (user-only) always worked; the structured extractors (mem0/consolidation/dedup) needed the fold. This is the same role-scoping lesson as W3.5.8 BCJ Entry 19, re-derived from the opposite direction.
- **three_tier's bug was granularity, not the graph.** Its L3 HyperMem (the actual differentiator) was never the problem — the inherited L2 stored whole-session blobs the reader couldn't see past 400 chars. Delegating L2 to the validated atomic-fact store fixed it without touching L3.
`─────────────────────────────────────────────────`

**mem0 hybrid (BM25) re-enabled (2026-06-02).** Earlier `fastembed` was dropped because `Qdrant/bm25` wouldn't download from HF's Xet CDN on this box. The model was later pre-fetched manually to `~/.cache/fastembed`. Re-enabled via three steps: reinstall `fastembed`; set `FASTEMBED_CACHE_PATH=<abs ~/.cache/fastembed>` in `.env` (mem0 instantiates `SparseTextEmbedding("Qdrant/bm25")` with no cache_dir, and its `except` reports any load failure as "fastembed not installed" — the env steers fastembed's default cache to the pre-fetched model, 0.3s, no network); and clear mem0 collections so they recreate with the `bm25` sparse-vector slot (mem0 only adds it at collection-creation). mem0 now runs hybrid dense+BM25. No `HF_HUB_OFFLINE` required.

**This single question is a poor litmus** (debatable gold counts a non-store item). The real signal is the full 20-Q slice — 10 multi-session + 10 knowledge-update, most with unambiguous golds. Deferred (hours at cloud-Haiku latency); the per-backend wiring is now verified end-to-end.

---

## Measurement infrastructure — per-run results, no clobber (2026-06-02)

The driver used to write a single `data/results_w358.jsonl` and `unlink` it at the
start of every run. Any partial run — a `--smoke`, a `--qid` re-run, a single
`--backend` probe — therefore **destroyed the prior full run's raw data**. This
bit three times in one session: the full 21-Q × 7-backend matrix raw was lost to a
3-question probe, then a KU re-run, forcing multi-session numbers to be
reconstructed from memory. Reconstructed numbers are not measured numbers — that
is a measurement-integrity violation, not a cosmetic one.

**Fix:** each run writes its own file, `data/results/run_<tag>.jsonl`
(tag = backends + scope + epoch, or `--run-tag`), and never unlinks.
`scripts/aggregate.py` reconciles all per-run files into the canonical
`data/results/merged.jsonl`, **latest-per-cell wins**, keyed on
`(question_id, backend)` in mtime order:

```bash
# run any subset — full, one axis, one backend, one question — none clobbers
uv run python -m src.run_longmemeval_slice --qid <ms-qids> --run-tag multisession_real
# merge every run file + print the matrix
uv run python -m scripts.aggregate
```

A later single-backend, single-question re-run now edits **exactly one cell**; all
sibling backends and sibling questions are preserved from their own runs. Verified
end-to-end: re-running `atomic_fact` for one KU qid updated that cell to the fresh
result while all 6 sibling backends + all 10 questions survived. Locked as a
regression guard in `tests/test_aggregate_merge.py` (3 cases: latest-per-cell
merge, broken-gold exclusion, `correct=None` pending count).

`★ Insight ─────────────────────────────────────`
- **Single-file `unlink`-and-overwrite is a silent data-loss trap for any eval harness.** It reads as "always shows the latest run" but actually means "any run erases the last." Per-run files + a latest-per-cell merge make partial re-measurement (re-run one axis, keep the rest) a `cp`-free, backup-free, one-command operation — the discipline an eval matrix needs to stay trustworthy across many runs.
- **The merge key is `(question_id, backend)`, not `question_id`.** That granularity is what lets a one-backend re-run refresh its own cell without touching siblings — coarser keying would re-introduce the clobber at the question level.
`─────────────────────────────────────────────────`

---

## Exit criteria

- [x] `src/probe_reader.py` — reader-probe harness (grep / sweep / show-facts / reimprint ablation / user-id override / reader-model override)
- [x] `src/prompts/reader_count.txt` — enumerate-then-count reader prompt
- [x] User-turn-only extraction in `atomic_fact_memory.py` (probe-validated 9× noise reduction)
- [x] Count-aware reader path in `run_longmemeval_slice.py` (auto-detects "how many/much/often")
- [x] All LLM roles on Haiku 4.5 via VibeProxy; embeddings local; judge independent on Sonnet
- [x] End-to-end production smoke on the count question → **correct** (was "I don't know")
- [x] Full 20-Q × 7-backend slice run (clean, crash-free; matrix above)
- [x] Per-question-type accuracy breakdown (multi-session vs knowledge-update)
- [x] Per-run result files + merge (no-clobber) — `scripts/aggregate.py`, `tests/test_aggregate_merge.py`
- [x] Transient-Qdrant retry on atomic_fact upsert (crash cells eliminated)
- [x] Ensemble fusion analysis — RRF non-monotonic on read-then-reason (3 mechanisms)

---

## Deferred work / open questions

1. ~~**Full 20-Q × 6-backend run with the all-Haiku stack.**~~ **DONE (2026-06-03)** — see the 7-backend matrix above. The remaining open question it surfaced: a *router* that routes KU→atomic_fact and multi-session→mem0 should beat both the blind ensemble (80%) and any single backend — worth building/measuring as the §4.15 follow-up.
2. **Boots double-count vs sweater inclusion.** The two paths to "3" on `0a995998` suggest a per-question-type look at how counting questions are graded — some golds count actions, some count physical items.
3. **Does user-turn extraction transfer to the other backends?** Applied to `atomic_fact` (and therefore `hybrid`'s router). `mem0` does its own extraction; `qdrant`/`three_tier` summarize. Whether the same noise-reduction helps them is unmeasured.
4. **Local strong reader.** Haiku-stable / gemma-flaky raises the question of whether a larger local model (or hard-constrained decoding) recovers the counting reliability without the cloud dependency.

---

## File inventory (additions this session)

```
src/
  probe_reader.py             — reader-probe harness (retrieve→read→judge replay, ablation modes)
  prompts/
    reader_count.txt          — enumerate-then-count reader prompt for "how many" questions
  atomic_fact_memory.py       — + user-turn-only extraction; chat/embed endpoint split
  mem0_backend_adapter.py     — llm→VibeProxy, embedder→oMLX
  consolidation.py            — chat→VibeProxy (3 clients); import json fix
  dedup_synthesis.py          — chat→VibeProxy
  tiered_memory_qdrant.py     — embeddings→oMLX endpoint
  run_longmemeval_slice.py    — reader→VibeProxy; count-aware + KU latest-wins path; per-run output (no clobber)
  ensemble_memory.py          — RRF fusion of atomic_fact + mem0 (the 7th backend)
scripts/
  aggregate.py                — merge per-run files latest-per-cell → merged.jsonl + matrix
tests/
  test_aggregate_merge.py     — regression guard for the per-run merge (no-clobber)
.env                          — LLM_BASE_URL (VibeProxy) + EMBED_BASE_URL (oMLX), all models = Haiku 4.5
```

## Supersede soft-delete wired (`_qdrant_supersede`) (2026-06-05)

Previously `supersede` hard-deleted the old fact (forward `supersedes` pointer
survived, but the old row was gone → backward/audit traversal dangled). Now wired
to **payload-patch soft-delete**:

- `dedup_synthesis._qdrant_supersede(tm, old_id, patch)` → Qdrant `points/payload`
  set-payload, merging `superseded_by` / `superseded_at` into the old point
  (vector + content intact).
- supersede branch imprints the new fact FIRST (for the back-pointer), then patches
  the old one (was: delete old → imprint new).
- `query_context(..., include_superseded=False)` (default) adds an
  `is_empty: superseded_by` filter → superseded facts excluded from live recall, so
  **measured accuracy is unchanged** vs the hard-delete era; `include_superseded=True`
  walks the full history for audit / time-travel.

Bidirectional chain now intact: new→`supersedes`→old AND old→`superseded_by`→new.

**Test:** `tests/test_supersede_soft_delete.py` (live Qdrant + oMLX, skip-gated):
old excluded from live recall · new has `supersedes` · old still retrievable with
`include_superseded=True` and carries `superseded_by`. **Suite 3→4 passed, no regressions.**
