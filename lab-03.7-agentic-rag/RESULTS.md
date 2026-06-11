# Week 3.7 — Agentic RAG: measured results

Every number below traces to a runnable artifact in `results/` or `observations/`. Three RAG
architectures over the same local stack (oMLX Gemma-26B judge/reader, BGE-M3 + BGE-reranker over
Qdrant `bge_m3_hnsw`), compared on the same dev set.

## Phase 2 — single-pass vs canonical vs structural (50-q easy dev set)

One unified 3-arm run → `results/comparison_raw.json` → RAGAS → `results/ragas_scores.json`.

| metric | single-pass | agentic_canonical (skip-allowed) | agentic_structural (fix) |
|---|---|---|---|
| faithfulness | 0.980 | 0.876 | **1.000** |
| answer_relevancy | 0.755 | 0.786 | 0.785 |
| context_precision | 0.982 | 0.694 | **0.982** |
| context_recall | 1.000 | 0.700 | **1.000** |
| latency (mean) | 1.59 s (1.00×) | 3.07 s (1.93×) | 1.56 s (**0.98×**) |
| retrieval skips | 0 / 50 | **15 / 50** | 0 / 50 |

**Finding.** The canonical LangGraph "agentic RAG" lets the LLM decide *whether* to retrieve;
on this local model it **skipped retrieval on 15/50** questions (oMLX ignores `tool_choice`), so
it answered from parametric memory and lost on 3 of 4 metrics at ~1.9× latency. Wiring retrieval
as a **structural graph edge** (`structural_rag.py`, `START → retrieve`) recovers every metric to
≈ single-pass at parity latency and **0 skips**. Verdict: *the canonical pattern is mis-built for a
RAG; build retrieval as a guaranteed edge.* Neither agentic arm fired a single rewrite (0/50).

## Phase 2.6 — difficulty stratification (fair, gold-rank based)

`make_hard_dev_set.py` labels difficulty by the **retrieval rank of the known-gold passage**
(outcome-independent, not "who won"). Of 100 candidates: **easy 80 · medium 9 · hard 3 ·
unreachable 1** (gold-at-rank-1 = 80%). On the 13-row hard set, structural still matched
single-pass on context metrics; canonical degraded further (skipped more under difficulty). The
rewrite loop still **never fired** — even constructed-hard, the reranker rescued the buried cases.
Lesson: *difficulty alone doesn't wake the loop; visible retrieval failure does.*

## Phase 3 — CRAG on out-of-corpus queries

`03_crag_eval.py` on 10 post-cutoff questions the 2018 MS-MARCO corpus cannot answer →
`observations/crag-out-of-corpus.md`.

| arm | answered | behaviour |
|---|---|---|
| single-pass | **0 / 10** | abstains honestly ("passages do not contain this; context insufficient") |
| structural | **0 / 10** | rewrite loop runs away (hits `recursion_limit`) — no answer, ~12 calls + crash |
| **CRAG** | **10 / 10** (Tavily) / 5-9/10 (DuckDuckGo) | scored corpus 0.00 → **routed to web 10/10** → answered from real web evidence |

The **routing is deterministic** (corpus 0.00 → web on 10/10); the answer *count* tracks web-backend
quality (Tavily 10/10, free DuckDuckGo 5-9/10). Sample CRAG (Tavily) answers — all web-grounded:

- *2025 NBA Finals* → "Oklahoma City Thunder defeated the Indiana Pacers in seven games, 103-91."
- *GPT-5 release* → "August 7, 2025, unveiled during a livestream."
- *2025 Nobel Physics* → "John Clarke, Michel Devoret, John Martinis — macroscopic quantum tunnelling."
- *Newest Apple silicon* → "M5 chip, announced October 15, 2025."
- *2025 G20 host* → "South Africa (Johannesburg Expo Centre)."
- *EU AI regulation* → correctly hedges: "the EU AI Act was already in force since 2024-08-01; GPAI rules came into force 2025-08."

**Finding.** This is the first arm to beat the baseline — because the failure (corpus can't answer)
is finally one a *scored evaluator* can see and route around (web), where the rewrite loop only knew
to re-ask the same empty corpus. single-pass isn't wrong here — it correctly abstains — it just
can't reach beyond the corpus.

**Hand-rolled CRAG (`baseline_handrolled.py --out-of-corpus`) — a three-version debugging arc.**
- **v1** keyword `grade_relevance`: corrective fires **0/10**. The keyword-overlap grader is fooled by
  the abstention echoing the question → `pass` → the corrective branch is unreachable.
- **v2** LLM `grade_relevance` (judges the answer text): **5/10**. Fires now, but Gemma flip-flops on
  identical-style abstentions — a hard target (an abstention is linguistically *about* the question)
  + a brittle binary output. The *same* Gemma grades consistently in `crag_variant.py` because that
  evaluator grades the **docs** ("can these answer?") with a **0-1 score binned by a threshold**.
- **v3** canonical abstention: `synthesize()` emits exactly `"I don't know"`, and `grade_relevance`
  short-circuits on it (substring check, no LLM call) → corrective fires **10/10, deterministically**;
  `rewrite_query` runs once per fire (proof: visibly different rephrasings); escalates to
  `next_action = web_search` **10/10**.

Lessons: grade the **retrieval**, not the answer text; prefer a **score + threshold** over a binary;
best of all, make the **generator** emit a canonical abstention so detection needs no judge at all.
Both now **execute** web search inline: a one-function `web_search()` call (Tavily → DuckDuckGo) in
`answer()`'s fallback makes the hand-rolled pipeline **answer all 10 out-of-corpus from real, grounded
([#N]-cited, drift-filtered) web evidence** — fully apples-to-apples with `crag_variant.py`. Remaining
difference: plain functions vs a LangGraph graph. The bounded loop held throughout (1 rewrite per
fire, no runaway).

> Measurement caveat: "answered" is a keyword-abstention heuristic, not a correctness check. An
> early run mislabeled single-pass's honest abstention as "answered" (the phrase list missed "do
> not contain"/"insufficient") — caught by reading raw outputs (`--show`). Trust the routing
> (deterministic) over the per-arm "answered" tally (coarse); for true accuracy, judge each answer
> against a web-sourced ground truth.

## Reproduce

```bash
cd ~/code/agent-prep/lab-03.7-agentic-rag
# env setup: see chapter §2.5 "Environment adjustments" (ragas into lab venv, drop xai_sdk)
uv run python src/02_comparison_harness.py            # 3-arm easy set → results/comparison_raw.json
uv run python src/02b_ragas_eval.py                   # RAGAS → results/ragas_scores.json
uv run python src/make_hard_dev_set.py --out data/hard_dev_set.jsonl   # §2.6 difficulty stratification
uv run python src/03_crag_eval.py --show              # §3.2 CRAG out-of-corpus (TAVILY_API_KEY optional)
```
