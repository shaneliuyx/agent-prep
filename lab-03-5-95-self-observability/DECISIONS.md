# Decision log — W3.5.95 Self-Observability lab

Dated, append-only record of non-obvious decisions and the evidence behind them.
Newest last. (The lab's own thesis applied to the lab: record the behavior, then
re-examine it.)

---

## DEC-001 (2026-06-04) — Keep the baseline `EXTRACT_PROMPT`; do NOT promote `STRONGER_PROMPT`

**Status:** decided — stay with baseline. `STRONGER_PROMPT` retained in
`scripts/ablation_filter.py` as a documented alternative, not the module default.

**Context.** The LEARNING extractor's `is_self_caused` self-attribution filter is
meant to drop environmental failures ("the API returned 500") so they don't get
stored as the agent's own self-patterns.

**History — how the decision was reached:**

1. **First single run (baseline 7B + current prompt).** 35 seeded obs → 6 facts
   kept, `dropped_env=0`. Inspection: 2 of the 6 were environmental rows reframed
   first-person ("…return HTTP 500…", "…database connection issues…").
   → Written up as **"33% environmental leak; the self-attribution filter never
   fires."** Conclusion drawn: *the filter is bounded by the summarizer's judgment.*

2. **Doubt.** A rate from n=1 against an LLM-filled field is not a measurement —
   the field is model-generated, so the judgment is presumably nondeterministic.

3. **Validation ablation** (`scripts/ablation_filter.py`): 2×2 —
   {7B, 14B} × {current prompt, stronger prompt} — re-seed 35 fresh rows and
   re-extract **5× per arm = 20 runs**. Metric: how many runs leak ≥1 environmental
   fact; how many genuine self-patterns survive.

   | arm | model | prompt | runs leaked | total env leaked | self-patterns kept (mean) |
   |-----|-------|--------|-------------|------------------|----------------------------|
   | A | 7B  | current  | **0/5** | 0 | 5.0 |
   | B | 7B  | stronger | **0/5** | 0 | 4.0 |
   | C | 14B | current  | **0/5** | 0 | 3.0 |
   | D | 14B | stronger | **0/5** | 0 | 3.0 |

4. **Finding (refutes the strong claim).** 0 leaks across all 20 runs — including
   5 reruns of the exact baseline that originally leaked. The "33%" was a rare,
   nondeterministic single-sample artifact, not a systematic rate. Stronger prompt
   / bigger model do **not** reduce an already-~0 leak — they trade **recall for
   precision** (facts kept fall 5 → 4 → 3; stronger prompt yields cleaner
   self-action phrasing).

**Decision + rationale.** Stay with the baseline `EXTRACT_PROMPT`:
- The stronger prompt's only measured effect at this scale is *more conservative*
  extraction — it keeps fewer facts (4 vs 5 on the 7B; the 14B keeps only 3). With
  leak already ~0, that extra conservatism mostly risks dropping borderline-real
  self-patterns (recall loss) for no measurable precision gain.
- The baseline is simpler and already passes the no-leak bar across n=5.
- Bigger model (14B) as extractor: rejected for the default — it's the agent's own
  model (echo-chamber risk, §2.2 concept 5) and the most conservative (3 facts),
  contradicting the "separate, smaller summarizer" design.

**Revisit triggers (when to change this decision):**
- A higher-N ablation (e.g. n≥20/arm) shows the baseline's leak frequency is
  materially > 0 → adopt `STRONGER_PROMPT` (its self-action-verb rule is the
  cheapest precision lever).
- Real (non-seed) OBSERVABILITY data shows environmental rows leaking in production
  → promote the stronger prompt and/or add a second verifier pass.

**Artifacts:** `scripts/ablation_filter.py` (harness + `STRONGER_PROMPT`),
`RESULTS.md` §2, chapter Phase 3 "Empirical follow-up (2026-06-04)", BCJ Entry 1.
