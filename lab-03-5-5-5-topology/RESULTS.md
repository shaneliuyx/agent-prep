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

### Phase 2 hierarchical — manual run measurement (2026-05-28, post reasoning-model fix)

Direct invocation `python code/hierarchical.py` on prompt "Compare regulatory frameworks for AI across EU, US, and UK" against oMLX `gpt-oss-20b` @ :8000, AFTER the 3-layer reasoning-model fix (BCJ Entry 6 — `reasoning_content` fallback + gap-acknowledge prompt + `max_tokens=4096` for TOP synthesize):

| Stage | Wall (s) |
|---|---:|
| `plan_decompose` (top-lead emits 2 macros) | 9.05 |
| Sub-lead 1 (parallel; 2 leaves + sub-synth) | 58.90 |
| Sub-lead 2 (parallel; 2 leaves + sub-synth) | **68.15** ← `max_sub_wall_s` |
| Sub-leads batch (`max`, not `sum`) | 68.15 |
| TOP `synthesize` (combines 2 sub-answers) | 38.25 |
| **Observed total** | **115.46** |
| Sequential-equivalent (`plan + sum(subs) + synth`) | 174.35 |
| **Speedup factor (sub-level parallelism)** | **1.51×** |

Topology: depth=2, agents_total=7 (1 top + 2 sub-leads + 4 leaves).

**Parallelism verified at sub-level.** Observed 115.46s ≈ `plan + max(subs) + synth` (115.45s) within 0.01s. Speedup is less than supervisor's 1.77× (Phase 1) because hierarchical's sequential overhead (plan + synth = 47.30s) is 41% of total — Amdahl's law floor for 2 parallel sub-leads with ~40% serial fraction.

**TOP synthesize NON-EMPTY** (3-layer fix verified working). Produced a full 7-row comparison table covering EU/US/UK with explicit gap acknowledgments. The reasoning-model trap (BCJ Entry 6) is closed.

`★ The fix's partial compliance is worth noting ─`
- **Gap-acknowledge prompt reduces but does NOT eliminate speculation.** The instruction "do NOT speculate or fill in missing material" produced a UK column populated with the model's own training-data knowledge (*UK AI Strategy 2021*, *Data Protection Act 2018*, ICO, £17M fine cap) — speculation the prompt explicitly forbade. The model ALSO labeled the UK row "(not covered by workers)" AND listed UK as a gap in the "Gaps in the workers' coverage" section. So it's **label-and-fill**, not **label-and-stop**.
- **The repetition loop is gone, which was the catastrophic failure mode.** Trading "no answer at all" (the empty TOP synthesize trap) for "labeled speculation" is a clear win. But reasoning models don't strictly obey scope-bounding instructions — they fill in what they know while flagging the gap.
- **Production implication:** if you need STRICT non-speculation (e.g., for fact-grounded RAG), the prompt-level gap-acknowledge instruction is insufficient. You'd need either (a) a smaller non-reasoning model that doesn't have the latent knowledge to speculate from, or (b) a post-processing step that strips out any content not directly traceable to retrieval context. Both are heavier than prompt engineering alone.
- **The hierarchical speedup (1.51×) is the Amdahl-ceiling for 2-sub-lead parallelism.** 41% sequential overhead caps the speedup. Going to N=3 macros wouldn't help much because the synthesize step grows with N (more sub-answers to combine). The right knob is making each sub-lead's work LONGER (more leaves per sub-lead) so parallelism dominates the sequential plan+synth bookend.
`─────────────────────────────────────────────────`

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
