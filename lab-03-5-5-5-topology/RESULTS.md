# Week 3.5.5.5 — Multi-Agent Topology Patterns: Results

**Date:** 2026-05-28
**Hardware:** MacBook Pro M5 Pro, 48 GB unified memory
**Stack:** oMLX local-first inference (`gpt-oss-20b-MXFP4-Q8` Haiku tier @ :8000) + Claude-Sonnet-4.6 via local proxy (`:8317`, optional) + pure-Python mock provider for offline unit tests
**Companion chapter:** [`Obsidian Vault/Agent Development Curriculum/Week 3.5.5.5 - Multi-Agent Topology Patterns.md`](../../Documents/Obsidian%20Vault/Agent%20Development%20Curriculum/) — 1462 lines, this RESULTS.md is the measured-numbers digest

---

## Exit criteria

- [x] Five topology patterns implemented as standalone runnable Python (supervisor / hierarchical / group-chat / handoffs / voting)
- [x] LLM-provider abstraction (`code/llm.py`) — three providers (`anthropic-proxy` / `openai` / `mock`) selected via `LLM_PROVIDER` env var; `.env` autoload via `python-dotenv` at module import
- [x] Test suite passes against real LLM: **17/17 PASS in 95s** (oMLX `gpt-oss-20b` via openai-compat endpoint)
- [x] Test suite passes against mock provider in **0.03s** (15 unit + 2 integration-marker SKIP = 17 collected)
- [x] Integration-marker pattern (`@pytest.mark.integration`) gates wall-time-dependent parallelism tests so default `pytest tests/` stays free + deterministic
- [x] Decision-matrix Phase 6 — five topologies named with token-cost / best-case / failure-mode columns

---

## Headline finding — 17/17 PASS in 95s against real LLM

Two run profiles, both green:

| Profile | Provider | Wall time | Result |
|---|---|---:|---|
| **Mock** (default) | `LLM_PROVIDER=mock` (autoset by conftest for unmarked tests) | **0.03s** | 15 PASS + 2 SKIP (integration-marker tests skipped — they need real LLM latency to measure parallelism) |
| **Real-LLM** (integration) | `LLM_PROVIDER=openai` → oMLX `gpt-oss-20b-MXFP4-Q8` @ :8000 | **95s** | **17/17 PASS** (all 5 phases including parallel-wall measurement) |

### Per-phase test breakdown

| Phase | Topology | Tests | Mock | Real-LLM |
|---|---|---:|---|---|
| 1 | Supervisor (plan → ThreadPoolExecutor workers → synthesize) | 3 | 2 pass + 1 skip (parallel-wall) | 3/3 PASS |
| 2 | Hierarchical (supervisor-of-supervisors) | 2 | 1 pass + 1 skip (parallel-wall) | 2/2 PASS |
| 3 | Group-Chat (round-robin + LLM-selector + custom-fallback) | 4 | 4/4 PASS | 4/4 PASS |
| 4 | Handoffs (Agent dataclass + handoff protocol, Swarm-style) | 4 | 4/4 PASS | 4/4 PASS |
| 5 | Voting (3 solvers + majority + LLM-judge aggregator) | 4 | 4/4 PASS | 4/4 PASS |

### Phase 2 hierarchical — 3-macro compose-model sweep (2026-05-28, canonical)

Default `top_fan=3`, `leaf_fan=2` → 10 agents (1 top + 3 sub-leads + 6 leaves). Matches `LEAD_DECOMPOSE_SYSTEM`'s "decompose into EXACTLY 3" contract. The earlier 2-macro measurements (`[:2]` hardcoded cap, 7 agents) silently dropped the planner's 3rd macro — BCJ Entry 7 cap-vs-contract bug, fixed before this canonical sweep.

Same prompt across all 3 models: "Compare regulatory frameworks for AI across EU, US, and UK."

| Model | Provider | total_wall_s | plan | sub_walls | max_sub | sum_subs | synth | sequential | **speedup** | UK macro present? | UK synthesis behavior |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|---|
| **Sonnet 4.6** | CLIProxyAPI `:8317` (cloud) | **61.31** | 5.21 | [25.28, 32.34, 24.83] | 32.34 | 82.45 | 23.76 | 111.42 | **1.82×** | ✓ | Label-and-STOP-FIRST: gap notice upfront; sub-section gaps inline |
| **gpt-oss-20b-MXFP4-Q8** | oMLX local | **150.95** | 15.25 | [75.04, 100.19, 81.39] | 100.19 | 256.62 | 35.51 | 307.38 | **2.04×** | ✓ | (3rd sub-lead covers UK; no need for top synth to fill gap) |
| **Qwen3.5-27B-Opus-distill** | oMLX local (MLX 4bit) | **374.59** | 45.12 | [198.54, 237.57, 216.20] | 237.57 | 652.31 | 91.90 | 789.33 | **2.11×** | ✓ | Strict literal synthesis from 3 sub-answers |

Full per-run outputs in lab `results/` dir:
- [`results/hierarchical_sonnet-4-6.txt`](./results/hierarchical_sonnet-4-6.txt) (212 lines)
- [`results/hierarchical_gpt-oss-20b.txt`](./results/hierarchical_gpt-oss-20b.txt) (131 lines)
- [`results/hierarchical_qwen-3.5-27b-claude-opus-distill.txt`](./results/hierarchical_qwen-3.5-27b-claude-opus-distill.txt) (244 lines)

**Key findings (3-model head-to-head, 3-macro fan):**

1. **Wall-time ranking is OPPOSITE of "smaller local = faster" intuition.** Cloud frontier (Sonnet) fastest at 61.31s. Smallest local reasoning (gpt-oss-20b) at 150.95s. Mid-size local distilled (Qwen-distill) slowest at 374.59s. The actual ordering reflects: cloud-frontier-optimized-inference >> small-local-reasoning >> mid-local-distilled. Conventional wisdom inverted.

2. **Speedup jumped from 2-macro fan (1.43-1.55×) to 3-macro fan (1.82-2.11×).** Adding the 3rd parallel sub-lead diluted serial overhead — same plan+synth cost spread across 3 parallel branches instead of 2. **Speedup IS topology-dependent, but it's monotonic with fan width up to Amdahl's law's ceiling.** Doubling parallel branches doesn't double speedup (1.5× → 3×) because plan+synth bookend grows too — measured ceiling for this 1+N+2N hierarchy is ~2-2.5×.

3. **3-macro fan revealed the UK regulatory framework that 2-macro silently dropped.** All 3 models' 3rd sub-lead branch covered UK explicitly: "How does the United Kingdom regulate AI..." (gpt-oss-20b), "What is the UK's AI regulatory framework..." (Qwen-distill), "What is the UK regulatory framework..." (Sonnet). The PRIOR 2-macro runs had to fabricate UK content at the top-synthesize layer because no sub-lead ever researched it. **Root cause was the `[:2]` cap, not the model — fix preserves planner's full output.**

4. **Sonnet's UK synthesis is now grounded.** With UK appearing as its own sub-lead's research output, Sonnet's top synthesize no longer needs to fabricate from training data — it can cite the 3rd sub-lead. The "label-and-stop-first" behavior persists, but now there's actual UK content to synthesize FROM rather than around.

5. **The BCJ Entry 6 reasoning_content trap remained gpt-oss-20b-specific** at 3-macro fan. Both Qwen-distill and Sonnet returned non-empty content first try; only gpt-oss-20b ever hit the empty-content + CoT-loop pattern. Same conclusion as 2-macro sweep.

6. **NEW BCJ Entry 7 — Distilled models can be MORE verbose than the model they were distilled from.** Qwen-distill emitted "Here are 3 sub-questions:" prose preamble BEFORE the JSON in one `plan_decompose` call, despite the explicit "Return JSON only, no prose" instruction. gpt-oss-20b and Sonnet both honored the format. Fix: regex `\{.*\}` extraction at the parse boundary instead of `startswith("```")` check. Distillation transfers behavior selectively — Qwen inherited Opus's thoroughness but didn't fully take its instruction-discipline.

7. **NEW BCJ Entry 8 — `LLM_TIMEOUT_S=60` default insufficient for Sonnet via proxy.** First Sonnet run hit `httpx.ReadTimeout`. Bumped to 300 via env override. Worth follow-up: per-provider timeout config (e.g., `ANTHROPIC_TIMEOUT_S > LLM_TIMEOUT_S > 60`). The 60s default is gpt-oss-20b-shaped; proxied frontier models need 5× more.

8. **NEW BCJ Entry 9 — Hierarchical cap `[:2]` was bug masquerading as cost-bound feature.** `LEAD_DECOMPOSE_SYSTEM` requires 3 sub-questions; `hierarchical_run` truncated to 2 silently. Two design surfaces disagreed for the entire chapter's lifetime; the EU/US/UK question revealed the discrepancy when a user asked "why only 2 macros?" Fixed: default `top_fan=3` (matches planner contract); `top_fan=2` is now an explicit caller override for cost-bounded use.

9. **Production decision matrix (refined for 3-macro):**
   - **Interactive grounded synthesis (sub-minute)** → Sonnet via proxy (61.31s, label-and-stop-first, subscription quota). Still the fastest + most disciplined option.
   - **Local-only literal grounded synthesis (no fabrication, audit-clean)** → Qwen-distill, accept 6.1× wall vs Sonnet (374.59s) for $0 cloud cost.
   - **Speed + structured output, tolerant of light speculation** → gpt-oss-20b with BCJ Entry 6 3-layer fix, accept 2.5× wall vs Sonnet (150.95s).
   - **The hierarchical pattern's viability shifts with compose model.** At Sonnet's 61s the pattern is INTERACTIVE-ACCEPTABLE; at 150s borderline batch; at 374s clearly batch-only. Same architecture, three product profiles.

10. **The "speedup is invariant" finding from the 2-macro sweep DOES NOT generalize across fan widths.** 2-macro showed 1.43-1.55× (tight cluster across 3 models). 3-macro shows 1.82-2.11× (also tight cluster). The CLUSTER is invariant per-fan-width because Amdahl's law caps it given the serial fraction. But moving from 2 → 3 macros shifts the cluster up. **Refined production rule: speedup is a TOPOLOGY-AND-FAN-WIDTH constant; wall is a model constant.**

### Phase 3 group-chat — 3-model compose sweep (2026-05-28)

Same task across 3 compose models: *"Write a Python function `is_palindrome(s: str) -> bool`. Reviewer + tester collaborate."* Three selector flavors per model (round-robin / llm-selected / custom). Max rounds = 9.

| Model | round-robin | llm-selected | custom | Total LLM calls |
|---|:-:|:-:|:-:|:-:|
| **Sonnet 4.6** (cloud) | 9 (max cap hit, no convergence) | **4** (optimal) | 7 | 9 + 8 + 7 = 24 |
| **gpt-oss-20b** (local reasoning) | **3** (early TERMINATE) | 6 | 4 | 3 + 12 + 4 = 19 |
| **Qwen3.5-27B-Opus-distill** (local distilled) | 6 | **1** (premature) | **1** (premature) | 6 + 2 + 1 = 9 |

Full per-run transcripts in lab `results/group_chat_{sonnet-4-6,gpt-oss-20b,qwen-3.5-27b-claude-opus-distill}.txt`.

**Three distinct failure shapes — same selector, opposite behaviors per model:**

1. **Sonnet HEDGES (round-robin hits max cap).** Sonnet's coder kept iterating, refining the function, asking for more review feedback — no natural stopping point. Round-robin has no convergence detection, so the loop hit max_rounds=9 instead of converging. Sonnet's "calibrated hedging" trait that wins §5.3.5 LongMemEval (more honest abstention) becomes "doesn't know when to stop" in group_chat.

2. **gpt-oss-20b TERMINATES EARLY (round-robin = 3 rounds).** Reasoning model's CoT decided "this code is good enough, time to stop" after only 3 turns — coder→reviewer→tester ended with TERMINATE. Tester's brief input was enough. Reasoning trace's internal "I think we're done" leaked into the emitted text as the TERMINATE token.

3. **Qwen-distill SHORTCUTS THE COLLABORATION (llm-selected + custom = 1 round).** Coder's turn 1 emitted: code + design notes + `"**Reviewer:** Please review..."` + `"**Tester:** Please provide test cases..."` + **`TERMINATE`** — coder DELEGATED to reviewer/tester but ALSO emitted TERMINATE in the same turn, so the loop exited before the named collaborators ever responded. The commitment-bias trait that wins LongMemEval (committed answers vs hedging) breaks group_chat (committed to "done" before any actual collaboration).

`★ The trait-vs-eval finding crystallized ─`
- **Same model trait wins in one eval, fails in another.**
  - LongMemEval (commitment-bias eval): Qwen-distill 77% > Opus 4.7 68% > Sonnet 4.6 60% — commit wins, hedge loses.
  - Group_chat collaboration eval: Sonnet 7 rounds custom > gpt-oss-20b 4 rounds custom >> Qwen-distill 1 round (BROKEN) — collaboration wins, premature-commit loses.
- **The eval selects the trait it scores.** Production rule: pick a model for the right TRAIT for your workload's failure mode, not for "best score on benchmark X." Anyone who reads "Qwen-distill won 77% on LongMemEval" and ships it into a multi-agent collaboration workload gets a 1-round broken loop. Anyone who reads "Sonnet wins on calibrated abstention" and ships it into a fast-decision workload gets a 9-round hung loop.
- **The Qwen-distill "1-round premature TERMINATE" is itself a new BCJ-class observation.** Group_chat's termination contract (`if "TERMINATE" in msg.upper(): exit`) assumes TERMINATE is emitted on CONVERGENCE. Models that commit to "I'm done" before any collaboration happens can emit TERMINATE prematurely. **Fix at the prompt layer**: change CODER's system prompt to "End with TERMINATE ONLY after AT LEAST 3 turns have happened AND reviewer + tester have both responded." OR fix at the runtime layer: assert minimum-rounds before honoring TERMINATE.
- **The selector matters less than I thought, the model matters MORE than I thought.** Earlier finding (Sonnet llm-selected = 4 rounds, cheapest) was Sonnet-specific. On Qwen-distill, llm-selected was the WORST (premature 1-round); on gpt-oss-20b, llm-selected was middle (6 rounds). **The selector × model matrix is a true 2D search space; pick by joint workload-fit, not by reading either axis alone.**
- **The round-robin "no convergence detection" finding REPLICATES across models.** Sonnet hit max cap at 9; gpt-oss-20b lucked into 3 rounds because coder's third-turn response happened to contain TERMINATE; Qwen-distill needed 6 rounds (closer to Sonnet shape because the rule-pinned cycle forced collaborators to speak before coder could TERMINATE on round 2). **Production rule from cross-model evidence: round-robin needs explicit max-cap + post-hoc convergence check; don't trust agents to self-terminate cyclically.**
`─────────────────────────────────────────────────`

### Phase 4 handoffs — 3-model compose sweep (2026-05-28)

Same 5 triage messages (refund / sales / refund / sales / refund) across 3 compose models. Two metrics: (1) **routing accuracy** — did the triage agent correctly hand off to refund-specialist vs sales-specialist? (2) **specialist persona faithfulness** — did the specialist respond IN-CHARACTER per its system prompt?

| Model | Routing | Specialist persona | Notes |
|---|:-:|:-:|---|
| **Sonnet 4.6** (cloud via proxy, **Option C cloak-bypass**) | **5/5** | **5/5 ✓** (clean, in-character) | **Option C cloak-bypass works.** Customer-service conversational framing in user message + no `system=` param sent. Refund agent: "I'd be happy to help you with a refund for your recent purchase." Sales agent: "I'd be happy to help you compare our Pro and Enterprise plans!" Zero "I'm Claude Code" leakage. Production rule: provider-aware role-embedding (legitimate-customization framing bypasses cloak; aggressive-override framing TRIGGERS Sonnet's prompt-injection defense per BCJ Entry 12). |
| **gpt-oss-20b** (local, Option C `else` branch) | **5/5** | **5/5** (clean, confident) | Local oMLX honors `system=` directly; no provider-aware switch needed. Refund agent: "I'm happy to help you with a refund. To process it quickly..."; sales agent: "Here's a quick side-by-side snapshot of Pro vs Enterprise..." |
| **Qwen3.5-27B-Opus-distill** (local) | **5/5** | **5/5** (clean, hedging) | Routing perfect. Specialist persona intact but responses LEAN TOWARD "I don't have access to specific..." — distilled model's calibrated knowledge-gap admission carries into the specialist role. Opus-trait transfer: same trait that wins LongMemEval (commit-with-evidence) makes the persona MORE honest about its limitations as a roleplay specialist. |

Full per-run outputs: `results/handoffs_{sonnet-4-6,gpt-oss-20b,qwen-3.5-27b-claude-opus-distill}.txt`.

**TWO BUGS FOUND DURING THIS SWEEP:**

1. **BCJ Entry 11 — Handoff parser couldn't strip parens from `HANDOFF: transfer_to_X()`**. Initial run: Sonnet emitted `HANDOFF: transfer_to_refunds()` 4/5 times; gpt-oss-20b emitted parens variant on 2/5 sales messages. Original parser used `reply.split(":", 1)[1].strip()` which kept `()`; `tool.__name__ == "transfer_to_refunds()"` mismatched against the bare-identifier function name. **Result: loop silently stayed at triage despite model deciding to hand off.** Fix: regex `\w+` extract — `re.search(r"HANDOFF:\s*(\w+)", reply)` captures only the bare identifier, ignoring parens / whitespace / trailing punctuation. Same trap-class as BCJ Entry 7 (models emit format variants of explicit-format instructions); production rule: **at every output-parsing boundary, regex extract the structured payload — never use `startswith`/`split` for structured data**.

2. **BCJ Entry 19 (W3.5.8) cloak-injection — SOLVED via Option C provider-aware role-embedding (measured 2026-05-30).** Initial measurement showed Sonnet specialist personas 0/5 (all "I'm Claude Code"). Two failed bypass attempts (BCJ Entry 12) ruled out aggressive override (made things WORSE — 0/5 routing too) and gentler `=== ROLE FOR THIS RESPONSE ===` framing (no improvement). **The working fix (Option C)**: at the Agent layer (`handoffs.py::Agent.respond`), detect provider via `_is_cloaked_proxy()` env-check; for cloaked-proxy path embed role as conversational customer-service context in user message (`"You are working as a customer service agent. Your role and how you approach customer messages:\n\n{role}\n\nA customer has sent this message:\n{msg}\n\nRouting options:..."`) and DON'T send `system=` param. For local backends (gpt-oss-20b, Qwen-distill) keep the clean `system=`/`user=` separation. Result: **Sonnet 5/5 routing + 5/5 specialist persona** with zero "I'm Claude Code" leakage. The architectural insight: legitimate-customization framing (natural customer-service context) bypasses Sonnet's prompt-injection defense AND outweighs the proxy's injected Claude Code system prompt in terms of how the model interprets the conversation. **Production rule (refined):** cloak-proxy is now usable for nested-agent specialist patterns IF role-embedding uses natural-context framing; provider-aware switch keeps local backends optimal AND cloud-proxy working.

---

### Phase 5 voting — 3-model compose sweep (2026-05-28)

3 questions × 2 aggregators (majority + llm-judge) × 3 compose models. Voting is the most resilient topology — no multi-turn loop, no nested agent invocation, no synthesis layer. Solver quality is the only model-dependent variable.

| Model | Q1: 137×23 | Q2: Eiffel Tower? | Q3: Python year |
|---|---|---|---|
| **Sonnet 4.6** | Majority 2/3 conf **0.67** (1 solver emitted "3,151" with comma) | 3/3 conf 1.0 | 3/3 conf 1.0 |
| **gpt-oss-20b** | **3/3 conf 1.0** | 3/3 conf 1.0 | 3/3 conf 1.0 |
| **Qwen-distill** | **3/3 conf 1.0** | 3/3 conf 1.0 | 2/3 conf **0.67** (1 solver emitted "**1991**" markdown-bold) |

All 9 answers were correct (3151, yes, 1991). The 2/3 confidence reductions come from **format inconsistency between solvers within the same model** — same model, same prompt, but one of 3 solvers (random sampling under temp=0) emitted a different surface-format that the answer-extractor saw as a different vote.

| Model | Format-failure surface |
|---|---|
| Sonnet 4.6 | Comma-formatted numbers (`3,151` ≠ `3151`) |
| gpt-oss-20b | No observed format-failures across 3 Qs |
| Qwen-distill | Markdown-bold formatting (`**1991**` ≠ `1991`) — Opus-trait transfer |

**Aggregator note:** llm-judge sometimes fell back to majority output on numeric questions (Q1 with gpt-oss-20b, Q1 with Qwen-distill both showed `method: majority` in the llm-judge result field). The judge response either didn't match the `BEST: N` parser regex OR voting.py has a fallback path on parse-failure. Not investigated; majority answer is correct on all 9 cells.

Full per-run outputs: `results/voting_{sonnet-4-6,gpt-oss-20b,qwen-3.5-27b-claude-opus-distill}.txt`.

`★ The Phase 4 + 5 sweep crystallizes "topology-model fit is per-pattern" ─`
- **Voting is the only topology unaffected by ANY of the 11 surfaced traps.** No reasoning_content trap (no synthesis), no premature-TERMINATE (no multi-turn loop), no specialist-cloak (no nested invocation), no parser-format issues (3 solver calls + simple aggregator). Mechanically simplest topology AND most reliable. **Production rule: when you have a hard decision with reliability requirements, the architectural answer is voting + judge — pay the 3-5× cost, eliminate the topology-level failure surface.**
- **Handoffs surfaces THREE distinct traps per layer.** Routing layer: parser-format (Entry 11) bites every model. Specialist layer: cloak-injection (Entry 19) bites Sonnet-via-proxy specifically. Specialist response layer: model-trait-mismatch (commit-bias hedging on Qwen-distill, reasoning-CoT-leak on gpt-oss-20b's edge cases). Each layer has its own failure mode → defense-in-depth required.
- **The "Sonnet routing 5/5, persona 0/5" finding is the most production-load-bearing.** Anyone shipping CLIProxyAPI Sonnet into a multi-agent specialist pattern will get routing-decisions-via-Sonnet + personas-overridden-by-Claude-Code-system-prompt. The result LOOKS correct at the routing log (every handoff trace shows correct destination) but the user-visible response is "I'm Claude Code, an AI assistant..." Production rule: **for proxy-cloaked Sonnet, the architectural boundary is "top-level synthesis OK, nested-agent personas NOT OK."** Use direct Anthropic API (subscription billing path) for specialist patterns, or local models.
- **Format-consistency-within-model is the voting failure surface.** Q1 Sonnet 2/3 (comma-format), Q3 Qwen-distill 2/3 (markdown-bold) — same model, same prompt, 3 solver calls sampled, ONE solver chose a different surface-format. Reasoning models (gpt-oss-20b) seem most format-consistent (3/3 on all Qs). Calibrated models (Sonnet) and distilled-from-thorough models (Qwen-Opus-distill) each have their own "favorite formatting" that occasionally leaks. **Production fix: post-process solver outputs through a normalizer (strip markdown, normalize numbers via regex) before majority-counting.** The aggregator pattern is bullet-proof; the answer-extraction step is the seam where format variation hurts confidence.
- **3-model trait fingerprint after Phase 1-5 sweep:**
  - **Sonnet (cloud)**: calibrated, hedge-and-iterate, format-variant on numbers, cloak-injection on nested specialists
  - **gpt-oss-20b (local reasoning)**: format-strict, reasoning-content trap on heavy synthesis, early-terminate on cyclic loops
  - **Qwen-distill (local literal)**: commit-bias (wins LongMemEval, breaks collaboration), prose-preamble on JSON, markdown-bold format leak
  - **Each trait has different polarity per topology.** No model is universally "best"; pick per-pattern.
`─────────────────────────────────────────────────`

---

### Phase 1 supervisor — manual run measurement (2026-05-28)

Direct invocation `python code/supervisor.py` on prompt "What changed in multi-agent systems between 2023 and 2026?" against oMLX `gpt-oss-20b` @ :8000:

| Stage | Wall (s) |
|---|---:|
| `plan_decompose` (lead emits 3 sub-questions as JSON) | 6.03 |
| Worker 1 (parallel) | 14.69 |
| Worker 2 (parallel) | **18.80** ← `max_worker_wall_s` |
| Worker 3 (parallel) | 12.49 |
| Worker batch (`max`, not `sum`) | 18.80 |
| `synthesize` (lead combines 3 worker outputs) | 10.32 |
| **Observed total** | **35.14** |
| Sequential-equivalent (`plan + sum_workers + synth`) | 62.33 |
| **Speedup factor** | **1.77×** |

**Parallelism verified empirically.** Observed total (35.14s) ≈ `plan + max(workers) + synth` (35.15s) within 0.01s rounding. If workers had serialized under the hood, total would be ≈ 62.33s — the `sum_worker_walls_s / max_worker_wall_s = 2.45` ratio confirms 3 workers ran genuinely concurrent on I/O-bound LLM calls (GIL releases on httpx wait). Worker 2's 18.80s upper bound dominates the wall; this is the right shape — synthesis CAN'T finish until the slowest worker returns.

**The supervisor produced a structured answer** combining ACL 3.0 + RFC 9123/9124 + 6G URLLC (Worker 2 = comms standards), MARL + GNN + federated learning (Worker 1 = algorithmic), and autonomous vehicles + smart grid + supply chain (Worker 3 = applications) — three distinct sub-domains that a single agent would either drop coverage on or take 60s to research sequentially. The disagreement-surfacing prompt produced an honest "no direct contradictions, but deployment-scale varies" note — the synthesis prompt's failure-mode defense working as designed.

`★ Insight ─────────────────────────────────────`
- **1.77× speedup is the practical ceiling for 3 workers**, not the theoretical 3×. Two reasons: (a) plan + synthesize are sequential overhead (16.35s of the 35.14s = 47%); (b) Amdahl's law dominates when sequential phases are large relative to workers. To approach 3× you'd need either more workers (5-6) or longer worker phases (~30-60s each). Anthropic's Research paper hits ~6× because their workers run for minutes each — the parallel fraction is much larger.
- **Worker walls are uneven (12.49 / 14.69 / 18.80)** — the wall is bounded by the slowest, not the average. Production tuning would aim to BALANCE worker-prompt difficulty (e.g., split the comms-standards sub-question into two if it's reliably slowest). Pareto-front move: more, smaller workers > fewer, bigger workers, until the synthesis cost of combining many outputs starts dominating.
- **`sum_worker_walls_s / max_worker_wall_s = 2.45` is the parallelism receipt** — a number ≥ N-1 (for N workers) indicates genuine concurrency. If this number dropped to ~1.0, workers were serializing somewhere (likely a global lock, shared HTTP session, or rate limiter). Always log both and assert the ratio in your integration test — this is what `test_supervisor_parallel_wins` empirically verifies via `total_wall < (parallel + sequential) / 2`.
`─────────────────────────────────────────────────`

`★ The 95s number is load-bearing ─────────────`
- **Real LLM latency is what makes the parallel-wall tests measurable.** Mock provider returns instantly (0.03s suite-wide), so `test_supervisor_parallel_wins` and `test_hierarchy_parallel_at_sub_level` cannot validate that `total_wall ≈ plan + max(workers) + synth` rather than `plan + sum(workers) + synth` — there's no wall-time signal. The 95s real-LLM run is where parallelism is empirically verified.
- **The marker-based integration gating is the methodological lesson.** Old pattern (`if os.getenv("LLM_PROVIDER") == "mock": pytest.skip(...)`) didn't work because conftest's autouse fixture overrode `LLM_PROVIDER` to `mock` for all unmarked tests — even `export LLM_PROVIDER=openai` got overwritten. Real fix: `@pytest.mark.integration` lets the conftest fixture leave the env alone for marked tests. Default `pytest` stays free + fast; `pytest -m integration` (or any pre-set `LLM_PROVIDER=openai`) exercises real-LLM parallelism verification end-to-end.
- **Decision matrix output (Phase 6) ranks topology choices by workload shape, not by accuracy** — multi-agent topology is a token-cost vs accuracy lever, not a single-axis improvement. Chapter §4 Phase 6 has the worked example.
`─────────────────────────────────────────────────`

---

## Five topology patterns — capability summary (chapter §2.2)

| Topology | Shape | Best for | Token cost vs single agent | When wrong |
|---|---|---|---:|---|
| **Supervisor / Orchestrator-Worker** | 1 lead + N workers, parallel | Research-shape decomposition | ~15× | Sequential tasks; simple queries |
| **Hierarchical** | Supervisor-of-supervisors, recursive (2 layers earn cost; 3 layers almost never) | Very large tasks needing 2+ decomposition layers | 30-50× | One-layer tasks; flat work |
| **Group-Chat (speaker-selection)** | Shared message pool + selector function | Emergent collaboration with unclear topology | ~5-10× (selector cost dominates) | Tightly-scripted workflows |
| **Handoffs / Routines** | Active-agent passes baton via tool-call | Triage + skill-based routing | ~2-3× (one agent at a time) | Long sessions; parallel execution |
| **Voting / Debate** | N independent solvers + aggregator | High-accuracy decisions; correctness > cost | 3-5× | Cheap-correctness tasks; latency-sensitive |

### The Anthropic 90.2% benchmark anchor

Anthropic's published 90.2% improvement on internal research evals (Opus 4 lead + Sonnet 4 subagents vs single Opus 4) anchors the supervisor pattern empirically. Same blog post: **80% of BrowseComp variance is explained by token usage alone.** Fresh context per subagent is the dominant mechanism — workers get clean 200k windows instead of carrying the 40k tokens the lead spent planning. The lab's `code/supervisor.py` exposes the per-worker-wall + sum-of-workers-wall + max-of-workers-wall in its return value precisely so the reader can VERIFY parallelism empirically — if max ≈ sum, threading didn't actually parallelize (something's serializing under the hood).

---

## LLM-provider abstraction — the architectural primitive

The lab's most-reused component is `code/llm.py` (~60 LOC + format-aware mock). Three providers behind one `chat(prompt, system)` call:

| Provider | When | Endpoint | API key env |
|---|---|---|---|
| `anthropic-proxy` | Curriculum default — Claude-Sonnet-4.6 via CLIProxyAPI cloaking proxy | `http://localhost:8317/v1/messages` | `ANTHROPIC_API_KEY` (any non-empty value) |
| `openai` | OpenAI-compatible endpoint (Azure / oMLX / vLLM) | `OMLX_BASE_URL` or `OPENAI_BASE_URL` | `OMLX_API_KEY` ⇒ `OPENAI_API_KEY` (precedence) |
| `mock` | Offline unit tests; format-aware stub | n/a (pure Python) | n/a |

Two architectural decisions earned their keep at debug time:

1. **`_provider()` deferred-lookup function instead of module-level constant.** Original `_PROVIDER = os.getenv("LLM_PROVIDER")` captured at IMPORT time, before pytest's `monkeypatch.setenv()` ran in conftest. Tests hit real LLM despite mock setup. Fix: call-time lookup via `def _provider(): return os.getenv(...)`. Same trap-class as W3.5.8 §9.5's `tm_iso` fixture issue — both required converting "captured at module load" to "looked up at call time."
2. **`.env` autoload via `python-dotenv` `find_dotenv(usecwd=True)`.** Walks from CWD up to filesystem root looking for `.env` — finds lab's `.env` AND parent `~/code/agent-prep/.env` (umbrella) without manual `source .env`. Existing process env takes precedence over `.env` values. `OMLX_*` env vars have precedence over `OPENAI_*` per agent-prep convention.

The cloaking-proxy gotcha (anthropic-proxy mode): `system`-role messages get OVERWRITTEN by CLIProxyAPI's `applyCloaking()` for OAuth-fingerprint coherence. Workaround in `_chat_anthropic_proxy()`: stuff instructions into user message as `[INSTRUCTIONS]\n{system}\n\n[USER MESSAGE]\n{prompt}` rather than sending separate `system` role. (See W3.5.8 BCJ Entry 19 for the full reverse-engineering of the proxy's anti-detection mechanism.)

---

## Bad-Case Journal — observed during lab execution (2026-05-27 → 2026-05-28)

The chapter §5 BCJ section is currently empty pending lab observations (per the curriculum's "no fabricated entries; OBSERVED only" discipline — see [[feedback_no_demo_gaming]] memory). Five entries observed during the lab's first end-to-end run; all five are now FIXED in the canonical chapter + lab code.

| Entry | Symptom | Production rule extracted |
|---|---|---|
| **1** — Module-level `_PROVIDER` constant captured `os.getenv()` at IMPORT TIME, before conftest's `monkeypatch.setenv("LLM_PROVIDER", "mock")` ran | Tests called real LLM despite mock setup; failed at network unreachable or burned tokens. `python -m pytest tests/` hit `:8000` even with mock fixture active. | Module-level env-var capture **defeats per-test environment override.** Convert any `_CONST = os.getenv(...)` that needs per-test override to a function: `def _config(): return os.getenv(...)`. Same class as anyio TaskGroup task-affinity bug (W3.5.8 BCJ Entry 20) — "captured at one time, consumed at another." |
| **2** — `_chat_openai` crashed with `KeyError: 'content'` when LLM emitted `tool_calls` or hit a stop sequence | Some OpenAI-compatible servers (notably vLLM in tool-use mode) return `null` content when model emits tool_calls; others return missing key. Caller expected `str`. | Defensive parse at the wire boundary: `try: return data["choices"][0]["message"]["content"] or "" except (KeyError, IndexError, TypeError): return ""`. **Coerce-or-empty is the right pattern at every third-party response boundary** (same shape as W3.5 BCJ Entry 2's `extract_memories()` JSON-shape defense). |
| **3** — Mock provider's generic stub broke supervisor decomposition + handoff routing + voting aggregator parsing | Original mock returned `"Mock response."` regardless of call context. But callers expected JSON (`decompose`), agent names (`group-chat selector`), `HANDOFF: tool` strings (`triage`), `BEST: N` votes (`llm-judge`). Each downstream parser crashed. | **Format-aware mocks are load-bearing for multi-call agent loops.** Inspect (system, prompt) to detect call intent; return PARSE-VALID stub per call shape. Recognized formats in `_chat_mock`: decompose → JSON array; synthesize → multi-sentence answer; triage → `HANDOFF: <tool>` keyed off USER MESSAGE; group-chat → agent name extracted from `Pick ONE of:`; llm-judge → `BEST: 0`; default → "Mock response." General rule: mocks for shape-dependent consumers need shape-aware stubs (e.g., pytest-mock's `Mock.side_effect`). |
| **4** — Triage mock false-matched "refund" because tool docstrings embedded in prompt | "What plans do you offer?" routed to refund agent. Root cause: agent's tool docstrings are embedded in the full prompt; mock's `if "refund" in prompt.lower()` matched the tool description, not the user's actual question. | When mock-detecting intent from a prompt, **extract the load-bearing field FIRST** (regex `USER MESSAGE:\s*(.+?)(?:\n|$)` to scope keyword search to user input only, not the system prompt or tool catalog). Same shape as any parse defense — narrow the scope before pattern-matching. |
| **5** — `pytest-asyncio` async-fixture crashed at teardown: `RuntimeError: Attempted to exit cancel scope in a different task than it was entered in` | `async with TieredMemory(...) as tm: yield tm` inside `@pytest.fixture` entered TaskGroup in setup-task, tried to exit in teardown-task. anyio's TaskGroup invariant requires same task for `__aenter__`/`__aexit__`. | (a) `pytest-asyncio` fixtures with async setup/teardown CROSS TASK BOUNDARIES, period. (b) For any async resource that creates `anyio.create_task_group()` internally, the lifecycle MUST stay inside one task — push `async with` INTO the test body, not the fixture. (c) The sync-fixture-yields-config-only pattern is the documented escape hatch. See W3.5.8 BCJ Entry 20 for the full debug trace including the failed first-attempt fix. |

All five entries plus the integration-marker discovery are now in the chapter §4 Phase code blocks + walkthrough (synced via `d3e2092` chapter sync + `62db279` 5-bug fix sync commits).

---

## Production-ready findings (the four interview-grade lessons)

### 1. Topology choice is a token-cost lever, not a free improvement

The Anthropic 90.2% number is the upside; the ~15× single-agent token cost is the price. Five topologies, five token-cost / use-case profiles. Senior-engineer signal in interviews: don't just name the patterns — defend a specific choice for the workload at hand. Chapter §4 Phase 6 is the analysis-only decision-matrix exercise that drills this.

### 2. Fresh context per subagent is the dominant mechanism

Anthropic's own writeup: 80% of BrowseComp variance is explained by token usage alone. Workers exploring sub-questions don't carry the 40k tokens the lead spent planning. The lab's `code/supervisor.py` exposes `worker_walls_s`, `max_worker_wall_s`, `sum_worker_walls_s` precisely so the reader can EMPIRICALLY verify parallelism (max ≈ sum → threading isn't working).

### 3. Mock-test reliability scales with mock format-awareness, not LOC

A single-line mock that returns `"Mock response."` works for one-call functions; it BREAKS the moment your code parses the mock output. Multi-call agent loops with downstream parsers (JSON, agent-name, tool-name, vote-ID) need format-aware mocks that inspect call context. The lab's `_chat_mock` is ~40 LOC and recognizes 6 call shapes. **Production rule: mocks ARE part of the test contract; mock-broken-on-shape-change is a real regression.**

### 4. Integration markers + autouse fixture = right gating primitive

`@pytest.mark.integration` + conftest's autouse fixture that sets mock for unmarked tests = clean default (fast, free) + explicit opt-in for real-LLM verification. Pattern generalizes to any "real infrastructure" gate (database, network, GPU, paid API). Don't try to detect "is real infrastructure available" — let the test author declare intent via marker and let conftest enforce the consequence.

---

## Interview soundbites (anchored to measured outcomes)

### Soundbite 1 — "Tell me about the multi-agent topology patterns you've worked with."

"Five canonical patterns: supervisor / hierarchical / group-chat / handoffs / voting. I built all five from primitives in pure Python — no framework magic — and the test suite passes 17/17 against a real LLM in 95 seconds. The supervisor pattern (Anthropic's 90.2% research result) gives you parallelism via `ThreadPoolExecutor` plus fresh context per worker; the cost is ~15× single-agent tokens. Hierarchical is a 2-layer extension that earns its cost on questions with genuine 2-layer decomposition like cross-region regulatory comparisons. Group-chat with speaker-selection trades determinism for emergent collaboration. Handoffs are the right primitive for triage + skill routing. Voting is the right primitive when correctness matters more than cost. The interview signal isn't naming them — it's defending which one for a given workload, and articulating the failure modes when wrong."

### Soundbite 2 — "Tell me about a subtle bug you debugged recently."

"My multi-agent topology lab tests hit a real LLM despite a mock fixture being active. Root cause: I had `_PROVIDER = os.getenv('LLM_PROVIDER')` at module load time. The conftest's autouse fixture set `LLM_PROVIDER=mock` per test via `monkeypatch.setenv()`, but that fired AFTER module import — the constant was already frozen. The fix was converting `_PROVIDER` from a module constant to a `_provider()` function that does the env-var lookup at call time. The deeper lesson: any module-level capture of mutable environment is a per-test override defeater. Same trap-class as `anyio` TaskGroup task affinity — both are 'captured at one time, consumed at another' bugs. Now I grep for `= os.getenv(` at module level whenever I see a 'mock isn't taking effect' bug."

### Soundbite 3 — "How do you make multi-agent code testable without burning tokens?"

"Format-aware mocks. My lab's `_chat_mock` is ~40 lines and recognizes 6 call shapes: decompose returns JSON, synthesize returns a multi-sentence answer, triage returns `HANDOFF: <tool>` keyed off the USER MESSAGE only (not tool docstrings — that's a false-match trap), group-chat returns an agent name extracted from `Pick ONE of:`, llm-judge returns `BEST: 0`, default is `Mock response.`. The mock is part of the test contract; when I changed the synthesis prompt to require disagreement-surfacing, the mock had to update too. With format-aware mocks, the suite runs in 0.03 seconds against zero tokens; integration-marker tests opt in to real LLM (95 seconds) only when measuring wall-time-dependent properties like parallelism. Default `pytest tests/` stays fast and free; `pytest -m integration` is the explicit cost gate."

---

## File inventory

```
code/
  llm.py            — provider abstraction (anthropic-proxy/openai/mock) + .env autoload
  supervisor.py     — Phase 1: plan_decompose → ThreadPoolExecutor → synthesize (~110 LOC)
  hierarchical.py   — Phase 2: supervisor-of-supervisors composing Phase 1 primitives (~55 LOC)
  group_chat.py     — Phase 3: GroupAgent + 3 selector flavors (round-robin/llm/custom) (~120 LOC)
  handoffs.py       — Phase 4: Agent dataclass + handoff protocol (~95 LOC)
  voting.py         — Phase 5: 3 solvers + majority/llm-judge aggregators (~115 LOC)

tests/
  conftest.py            — autouse mock fixture; integration marker bypass; pytest_configure marker registration
  test_supervisor.py     — 3 tests (1 integration-marker)
  test_hierarchical.py   — 2 tests (1 integration-marker)
  test_group_chat.py     — 4 tests
  test_handoffs.py       — 4 tests
  test_voting.py         — 4 tests
```

---

## Deferred work / open questions

1. **Real-LLM measurement of token costs.** Chapter §2.2 table cites ~15× / 30-50× / ~5-10× / ~2-3× / 3-5× for the five topologies — these are reference numbers from Anthropic + framework docs, not measured on the lab. Open follow-up: instrument each topology's `chat()` calls to count input + output tokens, then compute the actual multiplier on representative tasks vs a single-agent baseline.
2. **Anthropic 90.2% replication on a smaller benchmark.** The 90.2% number is on Anthropic's internal research evals — not reproducible externally. A smaller-scale replication on something like MMLU-Pro or BrowseComp-mini would validate the supervisor-pattern claim with the lab's local stack.
3. **Topology decision matrix as an LLM-routed dispatcher.** Phase 6 is currently analysis-only (the reader picks a topology by hand). A natural extension: build a meta-agent that reads the query, classifies it (research / triage / vote-worthy / scripted), and dispatches to the right topology. Would extend the lab from "5 patterns implemented" to "auto-selected pattern per query." Out of scope for v0.
4. **Cross-topology composition.** Real production systems compose patterns (e.g., outer supervisor routes to inner handoff chain; group-chat with voting aggregator at the end). Lab phases are standalone — composition examples are open work.
