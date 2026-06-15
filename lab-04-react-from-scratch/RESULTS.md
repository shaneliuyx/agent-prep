# Lab 04 — ReAct From Scratch: Results

Stack: MacBook M5 Pro (48 GB), local **oMLX** engine on `:8000` (one OpenAI- +
Anthropic-compatible endpoint, model-routed by the `model:` field). All numbers
measured on this hardware unless noted. Raw probe data:
`data/fleet_probe_20260615_omlx.json`.

> Status: scaffold + tools + fleet probe complete and measured. The Phase 5
> 15-failure ReAct bad-case suite and an end-to-end `agent_run()` task are **not
> yet run** — see [Pending](#pending).

## Fleet probe — 2026-06-15 (oMLX migration, reason cap 512)

Single run, 3 trials/probe. `recommend()` cheap-role floor = `reason ≥ 0.5 AND
instr ≥ 0.5`. Source: `scripts/probe_fleet.py`, dumped to
`data/fleet_probe_20260615_omlx.json`.

| Tier | Model | Ping (ms) | Tool | JSON | Reason | Instr |
|---|---|---|---|---|---|---|
| sonnet | `gemma-4-26B-A4B-it-heretic-4bit` | 317 | 1.00 | 1.00 | 1.00 | 1.00 |
| haiku | `MLX-Qwen3.5-35B-A3B-…-Reasoning-Distilled-4bit` | 152 | 1.00 | 0.00 | 0.83 | 0.00 |
| fast | `Qwen3.5-4B-MLX-4bit` (4 GB) | 236 | 1.00 | 1.00 | 0.83 | 1.00 |
| opus_qwen | `Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit` | 726 | 1.00 | 0.00 | 0.83 | 0.00 |
| opus_lazy | `gemma-4-31B-uncensored-heretic` (`Gemma4-31`) | — | — | — | — | — |

- `opus_lazy` cannot load: oMLX memory_guard `507` (projected 47.60 GB > 37.44 GB
  ceiling) once another model is hot — only one heavy model is resident at a time.
- `fast` tool calls are **structured** (Qwen3.5 is server-parsed); the previous
  `Qwen2.5-Coder-7B` fast tier was text-parsed and needed the client fallback.

### Role map (driven by the table → `src/models.py::ROLE_MAP`)

| Role | Model | Why |
|---|---|---|
| loop, tool_arg, reason, compose, finisher | Gemma-26B | only all-1.00 model |
| classify | Qwen3.5-4B (`fast`) | 235 ms, all 4 dims, structured tools |
| hard_loop | Qwen3.5-27B-Distilled | only other loadable tool-capable model |

## Tool smoke test (4/4 green)

`set -a; source .env; set +a; uv run python -c "import src.tools; …"` from the lab root:

| Tool | Result |
|---|---|
| registration | `['web_search', 'python_repl', 'read_file', 'write_file']` |
| `web_search("python list comprehension site:docs.python.org")` | `[1] 5. Data Structures — Python 3.14.6 documentation …` (live SearXNG) |
| `python_repl("print(2 ** 10)")` | `1024` |
| `write_file` + `read_file` round-trip | `hello from the agent` |

- `web_search` delegates to `shared/web_toolkit` (introduced W3.7); backend SearXNG
  via `docker compose -f shared/searxng/docker-compose.yml up -d`.
- `python_repl` hardening verified separately: a parent `SECRET_TOKEN` is **absent**
  in the child (env stripped via `env=_REPL_ENV`); `RLIMIT_CPU`/`RLIMIT_AS` set
  best-effort (`RLIMIT_AS` is a no-op on macOS, swallowed). Not a sandbox — see the
  SECURITY BOUNDARY note in `src/tools.py`; real isolation deferred to W11.5.

## Bad cases (what broke + the fix)

- **vMLX → oMLX migration.** The lab's original multi-port vMLX fleet + its headline
  model (`MLX-Qwen3.5-9B-GLM5.1-Distill`) no longer exist. Re-pointed everything at
  the single oMLX `:8000` endpoint (model-routed). *Fix:* `OMLX_URL` + bare served
  ids; `src/models.py` + `scripts/probe_fleet.py` rewritten.
- **Tool calling is a model × server-parser pairing.** `Qwen2.5-Coder-{7B,14B}` form
  correct calls but oMLX has no parser for that family → leaked as `<tools>`/
  `<function>` text on **both** API surfaces; `tool=0.00`. Gemma / Qwen3 / gpt-oss
  parse. *Fix:* prefer a parsed family; client fallback
  `scripts/probe_fleet.py::extract_text_tool_calls` recovers text → `tool_calls`.
  (Also overturned the old "heretic destroyed tool calling" claim — that was a vMLX
  artifact; heretic tool-calls fine on oMLX.)
- **Reasoning-distilled format collapse.** `tool=1.00` but `json=instr=0.00` for the
  35B-A3B / 27B reasoning models — `<think>` blocks bust tight format caps. *Fix:*
  route format-sensitive roles to the non-reasoning Gemma-26B; or disable thinking.
- **Probe token-cap manufactured a false `reason=0`.** The reason probe capped at 64
  tokens → clipped verbose-but-correct derivations (`finish_reason="length"`). *Fix:*
  raise reason cap to 512 (reason recovered 0.00→0.83); `recommend()` cheap floor now
  requires `reason` AND `instr` so a reason-only-capable model isn't picked.
- **`Qwen3.5-9B-OptiQ-4bit` is broken on oMLX** — `500` on every call (incompatible
  quant build). *Fix:* not adopted; use a standard MLX 4-/8-bit Qwen3.5-9B instead.
- **SearXNG container `not a directory` mount error.** A bind-mount pointed at a
  non-existent `/tmp/searxng-cfg/settings.yml`, so Docker created it as a directory
  and couldn't mount a dir onto the container's file. *Fix:* remove the bogus dir +
  use the canonical `shared/searxng/docker-compose.yml` (mounts `./settings.yml`).
- **Smoke test `ModuleNotFoundError: src.react`.** Ran the system `python`, not the
  lab venv. *Fix:* `uv run python …` from the lab root (`src/` is a namespace
  package; no `__init__.py`).

## End-to-end agent run — 2026-06-15

First full `agent_run()` via `src/run.py` (default loop model Gemma-26B; obs
sidecar logging one row per iteration to SQLite, `src/obs.py`).

Task: *"What is the square root of 144? Use the python_repl tool to verify."*
Final answer: **"The square root of 144 is 12."**

| Metric | Value |
|---|---|
| iterations | 2 (iter 0 `tool_call` python_repl → iter 1 `final_answer`) |
| tool calls / errors | 1 / 0 |
| python_repl latency | 31 ms |
| prompt tokens (total) | 1296 |
| completion tokens (total) | 42 |
| avg / max tool latency | 15.5 ms / 31 ms |

Confirms the loop terminates correctly on a `final_answer` (no `tool_calls`), the
tool dispatch path works (0 errors), and the obs sidecar writes one event row per
iteration. Source: `src/run.py` stdout + the SQLite event log.

### Multi-tool trajectory — 2026-06-15

Task: *"web_search the year Python was first released, python_repl to add 10, write_file the result."*
Final answer: **"…first released in 1991. Adding 10 gives 2001, saved to `py_plus10.txt`."** (correct)

| iter | event | tool | latency |
|---|---|---|---|
| 0 | tool_call | web_search | 3065 ms (SearXNG) |
| 1 | tool_call | python_repl | 55 ms |
| 2 | tool_call | write_file | 0 ms |
| 3 | final_answer | — | — |

Summary: 3 tool calls, **0 errors**, 4096 prompt + 119 completion tokens, max latency 3065 ms (web_search). Confirms the loop chains all three tool *types* in one trajectory and the model reads each tool result before choosing the next call — the defining ReAct property. Source: `agent_run(..., obs=True)` + SQLite.

## Phase 5 — bad-case suite — 2026-06-15

`uv run pytest tests/test_bad_cases.py` → **17 passed** (15 scenarios + Scenario 2's
second case + a deeper oldest-drop eviction test). Covers: max-iter guard, hallucinated tool name, oversized-result
truncation, malformed/missing args, premature stop, circular-reasoning detection,
tool timeout, error-as-observation, context-window eviction, inconsistent tool
schema, prose-instead-of-tool-call, nested tool calls, bounded retries, stale
scratchpad.

Two test bugs found + fixed on first green run (not loop bugs):
- Scenario 08 (`test_python_repl_times_out`): `python_repl` was never imported
  (a dead `autouse=False` fixture). Fix: import it at module top.
- Scenario 10 (`test_context_guard_evicts_oldest_entries`): fixture stuffed only
  30×500 chars (~3750 tokens) yet asserted it exceeded the 28k limit. Fix: 30×4000
  chars (~30k tokens) and assert `> CONTEXT_TOKEN_LIMIT`.

Eviction is now genuinely tested: the inline CTX_GUARD was extracted into
`react.py::_evict_if_over_limit(scratchpad)`, and `test_evict_drops_oldest_tool_result_first`
asserts the **oldest** tool result is dropped first, order is preserved, repeated
calls converge under the limit, and it is a no-op when under — not just "the loop
didn't crash."

## Pending

- Re-measure if the oMLX engine version changes (tool parsing + memory ceiling are
  engine-version-dependent).
- `react.py` carries pre-existing `reportPossiblyUnbound` / typing pyright lint
  (`llm_resp`, `tool_latency_ms`, `log_event`) — runtime-safe; cleanup deferred.
