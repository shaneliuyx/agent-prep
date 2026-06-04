# W3.5.95 — Self-Observability Memory: measured results

All numbers below are from a real run on this machine (oMLX `:8000`, agent
`Qwen2.5-Coder-14B-Instruct-MLX-4bit`, extractor `Qwen2.5-Coder-7B-Instruct-MLX-4bit`).
Reproduce with the commands in [§ Run process](#run-process). No fabricated data —
where a mechanism underperformed, the shortfall is reported as-is and filed as a BCJ.

---

## TL;DR

| Tier | What was measured | Result |
|------|-------------------|--------|
| OBSERVABILITY | unit suite (append-only, PII scrub, index, signal) | **7/7 pass** |
| OBSERVABILITY | PII scrub backend (Presidio NER vs regex) | **catches PERSON/LOCATION/PHONE/CC/IP regex misses; 2.25 ms/write** |
| LEARNING (consolidation) | 35 obs → typed self-facts (7B) | **~5 facts kept** |
| LEARNING — self-attribution | leak across 20-run ablation ({7B,14B}×{prompt}) | **0/20 leaked (one earlier single run leaked 2/6 — nondeterministic)** |
| Metacognitive recall | paired trial, contrarian facts, recall OFF→ON | **divergence 6/6 = 100%** |
| Metacognitive recall | ON choice == the better tool | **improvement 5/6 = 83%** |
| smolagents integration | both seams on a real agent framework | **works; agent followed recall** |

**One-sentence finding:** the seams work end-to-end, but recall only changes
behavior on **contrarian, environment-specific** self-knowledge (patterns *not*
in the base model's priors), and the consolidation step's self-attribution is
**nondeterministic** — a single run can leak environmental noise (one did, 2/6)
while a 20-run ablation leaks none, so the leak must be measured as a distribution,
not read off one run.

---

## 1. OBSERVABILITY — the append-only behavioral log

`uv run python -m pytest tests/test_observability.py -q` → **7 passed**.

| Test | Property proven |
|------|-----------------|
| `test_append_and_query_by_tool` | rows append; by-tool / by-run / recent queries return correct counts |
| `test_append_only_pk` | duplicate `(run_id, step_idx)` raises `IntegrityError` — overwrites are surfaced, not silent |
| `test_pii_scrubbed_at_write` | `sk-…` keys, `/Users/…` paths, emails redacted **before** persisting |
| `test_presidio_scrubs_named_entities` | Presidio NER redacts PERSON + CREDIT_CARD a regex can't (skips on regex fallback) |
| `test_raw_args_optout_keeps_secrets` | `raw_args=True` opt-out preserved (for trusted internal tools) |
| `test_tool_query_uses_index` | by-tool query hits `ix_obs_tool_ts` (verified via `EXPLAIN QUERY PLAN`) — not a scan |
| `test_user_signal_stamp` | a late thumbs-up/down stamps onto an existing row |

Key design point proven: the log is **read as memory** (indexed for recall-time
queries), not just written as a debug artifact.

### 1.1 PII scrub backend — Microsoft Presidio (NER) vs regex

The write-boundary scrubber (`src/pii_scrub.py`) defaults to **Microsoft Presidio**
(`presidio-analyzer` + `presidio-anonymizer` on a small spaCy model
`en_core_web_sm`), with a **graceful regex fallback** when Presidio/spaCy isn't
installed. Custom `PatternRecognizer`s add the secret shapes Presidio has no
built-in for (OpenAI keys, Bearer tokens, `/Users/` paths, hex); the built-in NER
catches contextual PII a fixed regex never could.

Measured (`backend() == "presidio"`):

| Input | regex output | Presidio output |
|-------|--------------|-----------------|
| `sk-abcdef0123456789abcdef` | `<API_KEY>` | `<API_KEY>` (custom recognizer) |
| `a@b.com` | `<EMAIL>` | `<EMAIL>` |
| `Dr. Sarah Johnson who lives in Seattle` | *(unchanged — miss)* | `Dr. <PERSON> who lives in <LOCATION>` |
| `phone 212-555-0147` | *(unchanged — miss)* | `phone <PHONE_NUMBER>` |
| `card 4111 1111 1111 1111` | *(unchanged — miss)* | `card <CREDIT_CARD>` |
| `from 10.0.0.42` | *(unchanged — miss)* | `from <IP_ADDRESS>` |

**Latency (warm, model loaded once via lazy singleton):** Presidio **2.25 ms/scrub**
vs regex **0.0014 ms/scrub** — ~**1555× slower relatively**, but 2.25 ms absolute
is negligible against real tool latency (a `grep`/`web_search` call is 10s–1000s of
ms). The accuracy gain (named-entity PII) is worth it on a normal agent loop; for
an extreme-volume hot path, scrubbing moves to a batched/async lane (BCJ Entry 7).

---

## 2. LEARNING — hot→warm consolidation (the noisy tier)

Seed: 35 OBSERVABILITY rows — **18 self-pattern** (3 templates ×6), **8
environmental** (2 templates ×4), **9 noise** (3 templates ×3).

### Single run (the one that misled me)

One extractor run (7B + current prompt) emitted **6 facts, kept all 6**
(`dropped_env=0`), of which 2 were environmental outcomes reframed first-person —
"I keep calling web_search with queries that return HTTP 500 errors" (a 500 is
server-side) and "I frequently encounter database connection issues" (infra). I
wrote that up as **"33% (2/6) noise; the self-attribution filter never fires."**

### Validation ablation (n=5 × 4 arms = 20 runs) — `scripts/ablation_filter.py`

Because `is_self_caused` is filled in by the summarizer, the judgment is
nondeterministic — so a *rate* needs many runs. Re-seeding 35 fresh rows and
re-extracting 5× per arm:

| arm | model | prompt | runs that leaked ≥1 env | total env leaked | self-patterns kept (mean) |
|-----|-------|--------|--------------------------|------------------|----------------------------|
| A | 7B  | current  | **0 / 5** | 0 | 5.0 |
| B | 7B  | stronger | **0 / 5** | 0 | 4.0 |
| C | 14B | current  | **0 / 5** | 0 | 3.0 |
| D | 14B | stronger | **0 / 5** | 0 | 3.0 |

### What this actually proves

- **The 33% did not reproduce.** 0 leaks across all 20 runs — *including 5 reruns
  of the exact baseline (arm A) that originally leaked.* The single-run 33% was a
  rare tail event, not a systematic rate. My write-up committed the error of
  reading a rate off n=1.
- **Self-attribution is nondeterministic.** Same config, temperature=0, leaked 2/6
  once and 0/6 twenty times — MLX sampling isn't pinned by temp=0 (same class as
  the reasoning-model nondeterminism elsewhere in the journal).
- **Stronger prompt / bigger model ≠ less leak (it's already ~0)** — they trade
  recall for precision: facts kept fall 5 → 4 → 3, and the stronger prompt's
  self-action-verb rule yields cleaner phrasing ("I keep choosing grep for large
  monorepos"). The axis they move is recall-vs-precision, not noise.
- **Signal extraction works:** all 3 seeded self-patterns surface every run; all 9
  noise rows produce zero facts in every run.
- The corrected lesson (BCJ Entry 1): measure an LLM-filled filter as a
  **distribution over N runs**; one run — leaking *or* clean — characterizes nothing.

---

## 3. Metacognitive recall — does a recalled self-fact change behavior?

Paired trial (`tests/test_self_recall_changes_behavior.py`): same task + same seed,
once recall **OFF**, once **ON**; measure divergence (chosen tool differs) and
improvement (ON choice == the tool the self-fact points to).

```
  divergence: 6/6 = 100%  |  improvement: 5/6 = 83%  (target divergence ≥ 30%)
  [Search this repository for every caller of parse] off=rg          on=grep              (better=grep)              recall=hit
  [Find all usages of the logger across this codeba] off=rg          on=grep              (better=grep)              recall=hit
  [Locate config.yaml somewhere in this project tre] off=fd          on=find              (better=find)              recall=hit
  [Find the settings file in this project.]          off=fd          on=find              (better=find)              recall=hit
  [Look up the current best practice for a caching ] off=web_search  on=read_local_notes  (better=read_local_notes)  recall=hit
  [Search for how to configure retry backoff.]       off=web_search  on=grep              (better=read_local_notes)  recall=hit
```

Every probe diverged (`off ≠ on`); recall hit on all six. Five flipped to the
tool the self-fact points to; the last (retry-backoff) flipped `web_search → grep`
— a different tool, but the wrong one — which is the measured "behavior change ≠
improvement" case.

### The load-bearing caveat (the lab's central finding)

The 6/6 result holds **only because the probes use CONTRARIAN, environment-specific
facts** the base model cannot know from training:

> "In my environment rg segfaults on this repo's symlinked vendor dirs; plain grep
> is the one that completes here."

An **earlier** probe set used general best-practices (`rg > grep`, `fd > find`) —
facts already in the 14B's priors. Measured divergence there: **0/6**. Recalling
something the model already believes changes nothing. **Recall earns its keep only
on idiosyncratic self-knowledge.** (BCJ #3.)

- **improvement 5/6, not 6/6:** the "retry backoff" probe flipped `web_search →
  grep` — a *different* tool, but the *wrong* one. Proof that **behavior change ≠
  improvement**: injecting a prior perturbs the policy; it does not guarantee a
  better action.

The test hard-asserts only `diverged ≥ 1` (the claim is real); the 30% target is
reported, not asserted, because n=6 against a nondeterministic model.

---

## 4. smolagents integration — the same seams on a real framework

`uv run python src/smolagents_agent.py` — `CodeAgent` on oMLX, three instrumented
tool stubs, recall block prepended to the task.

```
>>> recall block injected:
## Self-Patterns You Have Observed
- [tool_preference] When I search this repository, rg segfaults on its symlinked
  vendor dirs; plain grep is the search tool that completes here.

>>> agent final answer: Based on … my prior behavior, I used plain grep …

>>> OBSERVABILITY rows written by the run (4):
  step0 grep status=ok    step1 grep status=ok    step2 grep status=ok    step3 <step_error>
```

- **READ seam** ✓ — recall block injected; the agent **verbalized following it**
  ("Based on … my prior behavior, I used plain grep"), choosing the recalled
  contrarian tool (grep) over its rg prior.
- **WRITE seam** ✓ — real agent-driven tool calls logged to OBSERVABILITY via the
  `@_instrumented` decorator (not via the step callback — `CodeAgent` runs *code*,
  so wrapping the tool is the correct seam).
- **Honest limit:** run-to-run nondeterministic — recall steered all three calls to
  grep this run, but an earlier run used `rg` on one step despite the pattern.
  Recall **biases** the policy, it does not **control** it. Same finding as §3.

Four real integration bugs surfaced and were fixed — BCJ #4/#5/#6 (and #8: a fixed
`run_id` collided with a prior run's rows on the persistent append-only DB).

---

## Run process

```bash
# 0. env: .env holds OMLX_BASE_URL / OMLX_API_KEY / MODEL_AGENT / MODEL_EXTRACTOR
#    oMLX must be up on :8000 (LLM steps skip or fail fast if not).

# 1. unit suite (no LLM — fast, deterministic)
uv run python -m pytest tests/test_observability.py -q

# 2. seed → consolidate → inspect (the LEARNING noise-rate measurement)
uv run python scripts/seed_observability.py
uv run python src/learning_extractor.py

# 3. the headline paired trial (needs oMLX — the 14B agent model)
uv run python -m pytest tests/test_self_recall_changes_behavior.py -q -s

# 4. the same seams on smolagents
uv run python src/smolagents_agent.py
```

---

## Bad-Case Journal (real bugs hit building this lab)

| # | Symptom | Root cause | Fix |
|---|---------|-----------|-----|
| 1 | reported "33% env leak" from one run; it didn't reproduce | `is_self_caused` is filled by the summarizer → nondeterministic judgment; I read a *rate* off n=1 | corrected via 20-run ablation (0/20 leaked, incl. 5 baseline reruns); measure stochastic filters as a distribution. Built `scripts/ablation_filter.py`. |
| 2 | extractor crashed: `KeyError: '"type"'` | `str.format()` on a prompt containing literal JSON braces (`{"type": …}`) read them as format fields | `EXTRACT_PROMPT.replace("{rows}", …)` instead of `.format()` |
| 3 | paired-trial divergence 0/6 (claim looked false) | probes used patterns already in the model's priors (`rg>grep`); recall was redundant | redesigned probes to contrarian, environment-specific facts → 6/6. **The finding, not just a fix.** |
| 4 | smolagents `@tool` raised `DocstringParsingException` | tool docstrings were one-liners; `@tool` needs a Google-style `Args:` block, one arg per line | rewrote all three tool docstrings |
| 5 | `CodeAgent` parse crash: `SyntaxError … </code` | smolagents issue #1851 — the `</code>` stop sequence emits a partial `</code` that leaks into the parsed Python (local/MLX models trigger this) | `use_structured_outputs_internally=True` — agent returns `{"thought","code"}` JSON via `response_format` (oMLX supports json_schema), bypassing the stop-sequence scrape |
| 6 | `ProgrammingError: SQLite objects created in a thread can only be used in that same thread` | `CodeAgent` runs the model's code (and thus the instrumented tools) in a worker thread; the conn was created in the main thread | `connect(..., check_same_thread=False)` — safe here (WAL + GIL + smolagents' sequential single-tool loop serialize writes) |
| 7 | Presidio scrub ~1555× slower than regex (2.25 ms vs 0.0014 ms per write) | NER inference runs per write; a naive build also reloads the spaCy model on every call | lazy **singleton** engine (load model once) + small model `en_core_web_sm`; absolute 2.25 ms is fine on a normal loop. Graceful **regex fallback** when Presidio/spaCy absent so the lab still runs. For extreme volume → batched/async scrub lane. |
| 8 | smolagents demo crashes `UNIQUE constraint failed (run_id, step_idx)` on the *second* run; recall prints the same fact 3× | `main()` hardcoded `run_id="smolagents-demo"` + re-`INSERT`ed the seed fact every run, against a **persistent** append-only DB | unique `run_id` per invocation (`uuid`) so rows never collide; **DELETE-then-INSERT** the demo fact so reruns replace not stack. Append-only memory makes re-running first-class: fresh key per writer, idempotent seeds. |

---

## Interview soundbites (principle-level)

- **"Make the agent's own behavior log a memory tier, not a debug artifact."**
  The same append-only table you'd write for telemetry becomes self-knowledge the
  moment you *index it for read-at-decision-time* and inject the relevant rows back
  into the prompt. The architectural move is cheap; the discipline (append-only PK,
  PII-scrub at the write boundary) is what makes it safe to keep.

- **"In-context recall beats fine-tuning for self-correction — but only on
  knowledge the base model doesn't already have."** Recalling a best-practice the
  model already believes changes nothing (measured: zero divergence). Recall earns
  its cost only on *contrarian, environment-specific* facts ("rg segfaults on
  *this* repo"). Behavior change is also not the same as improvement — a recalled
  prior perturbs the policy; it doesn't guarantee a better action.

- **"A self-attribution filter is only as good as the model that fills it in — and
  that judgment is nondeterministic."** Separating environmental failure ("the API
  returned 500") from self-caused mistake ("I keep mis-using this tool") is the core
  of trustworthy self-memory. The filter is one boolean — but it's filled by the
  summarizer, so it varies run to run: one run leaked 2/6, a 20-run ablation leaked
  0/20. The lesson: build the seam, then *measure the leak as a distribution over N
  runs* — a single run's number (leaking or clean) characterizes nothing.

- **"Wiring memory into a real framework is where the integration tax shows up."**
  The seams were ~10 lines each; making them survive a real `CodeAgent` cost three
  bugs — a docstring schema, a stop-sequence parser leak (#1851), and SQLite
  thread affinity. The seam is the easy part; the framework's executor model
  (off-thread code execution, custom code fences) is the hard part.
