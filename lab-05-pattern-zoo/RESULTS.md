# Lab 05 — Pattern Zoo: Results

Stack: MacBook M5 Pro (48 GB), local **oMLX** engine on `:8000` (one OpenAI-
compatible endpoint, model-routed by the `model:` field). Model for every pattern
and the judge: `gemma-4-26B-A4B-it-heretic-4bit` (the only fleet model scoring 1.00
on tool+json+reason+instr — see lab-04 RESULTS.md fleet probe 2026-06-15). Zero
cloud spend. All numbers below are **measured** by `src/05_compare.py`, not asserted.

Interpreter: `/Users/yuxinliu/code/agent-prep/.venv/bin/python` (Python 3.13, `openai` 2.41.0).
oMLX base_url `http://127.0.0.1:8000/v1`, model `gemma-4-26B-A4B-it-heretic-4bit`.

## Canonical task

> Using the tools, find the population of 'Springfield' and the population of
> 'Shelbyville' from the knowledge base, add them together, then multiply that sum
> by the 'growth_factor' value from the knowledge base. Report the final number.

Ground truth: `(30000 + 12000) * 3 = ` **126000** (fixed `kb_lookup` table in
`src/tools.py`). Tools: `kb_lookup(key)`, `add(a,b)`, `multiply(a,b)` — deterministic
and offline, so the run is reproducible and isolates the *control flow* of each
pattern (the thing being compared).

Correctness is scored two ways: **exact** (does 126000 appear in the answer, a
deterministic ground-truth check) and **judge** (LLM-as-judge PASS/FAIL on the same
local model).

## Comparison matrix — measured 2026-06-16 (oMLX)

Run command (reproduces the table):

```bash
cd lab-05-pattern-zoo
set -a; source .env; set +a
PYTHONPATH=. /Users/yuxinliu/code/agent-prep/.venv/bin/python \
  -c "import runpy; runpy.run_module('src.05_compare', run_name='__main__')"
```

| Pattern | Correct (exact) | Correct (judge) | LLM calls | Prompt tok | Completion tok | Total tok | Latency (ms) |
|---|---|---|---|---|---|---|---|
| ReAct | PASS | PASS | 6 | 2941 | 107 | 3048 | 4028 |
| Plan-and-Solve | PASS | PASS | 7 | 3596 | 212 | 3808 | 4931 |
| **CodeAct** | PASS | PASS | **1** | 321 | 88 | **409** | **1258** |
| **ReWOO** | PASS | PASS | **2** | 522 | 278 | **800** | 3496 |

Token counts are stable across repeated runs (temperature 0.0); latency varies a
few hundred ms per run (local MLX scheduling). The two new patterns were re-run a
second time with byte-identical token columns — see "Stability" below.

### What the numbers show (the teaching point, measured)

- **CodeAct collapses N tool round-trips into 1.** The action space is *code*, so the
  model emitted a single ```python``` block that called `kb_lookup` three times,
  `add`, `multiply`, and printed `FINAL: 126000.0` — all in **one** LLM call (409
  total tokens). ReAct needed **6** calls (one per tool step + the final answer) and
  **3048** tokens. CodeAct used **~7.5× fewer tokens** and **6× fewer LLM calls**
  than ReAct on the identical task. (Source: `src/05_compare.py` matrix row +
  `impl_codeact.run().trace` = `exec -> 'FINAL: 126000.0'`.)

- **ReWOO decouples LLM calls from tool steps.** Planner (1 call) emitted all 5 steps
  with `#E` placeholders; the Worker executed all 5 tool calls with **zero** LLM
  calls; the Solver (1 call) wrote the answer. So **2** LLM calls regardless of the
  5-step trajectory, vs Plan-and-Solve's **7** (it re-feeds every observation). ReWOO
  used **800** tokens vs Plan-and-Solve's **3808** — **~4.8× fewer** — because
  observations never re-enter the reasoning loop. (Source: matrix + `impl_rewoo`
  trace: 5 `worker #E… ->` lines, `calls=2`.)

- **Plan-and-Solve costs more than ReAct here**, not less: the up-front PLAN call adds
  a round-trip and a longer prompt, and on a short linear task the explicit plan buys
  no step savings. The matrix makes that visible rather than assumed.

## Stability (CodeAct / ReWOO re-run)

Second invocation of `src/05_compare.py`, token columns unchanged:

| Pattern | LLM calls | Total tok | Latency (ms) |
|---|---|---|---|
| CodeAct | 1 | 409 | 1240 |
| ReWOO | 2 | 800 | 3497 |

## Implementation notes

- **Shared harness, additive.** The lab directory was empty; the zoo's common pieces
  (`src/schema.py` `AgentResult` + `CANONICAL_TASK`, `src/tools.py`, `src/llm_client.py`)
  were written to match `lab-04-react-from-scratch` conventions exactly (same oMLX
  client construction, same OpenAI function-calling schema shape, real
  `response.usage` token accounting). Every impl exposes `run(task) -> AgentResult`,
  so `05_compare.py` drives them identically.
- **CodeAct sandbox.** Model code runs via `exec` in a restricted namespace: the three
  tool callables + a tiny builtins allow-list (`print`, `int`, `float`, `sum`, `range`,
  …); no `import`, `open`, or `__import__`. Failure modes are handled: prose-instead-of-
  code is nudged and retried; a raised exception is fed back as the observation
  (traceback, limit 3); `MAX_ITER=6` bounds the loop. This is NOT a real sandbox
  (attribute gadgets can escape) — acceptable only for a non-adversarial teaching
  agent; real isolation is the lab-04 `python_repl` story / W11.5.
- **ReWOO vs Plan-and-Solve are genuinely different.** Plan-and-Solve re-feeds each
  observation and adapts step by step (observation-driven SOLVE loop). ReWOO fixes the
  whole plan before any tool runs and never sends an observation back to a model — the
  evidence table lives only in the Worker.

## Bug hit during build/run

- **`src/05_compare.py` cannot be imported as a normal module** — the filename starts
  with a digit (`05_compare`), which is not a valid Python identifier, so
  `import src.05_compare` is a SyntaxError. Root cause: the lab's existing naming
  convention (`NN_name.py`) collides with Python's import grammar. Fix: run it via
  `runpy.run_module('src.05_compare', run_name='__main__')` (and the file documents
  this at the top). No code change to the impls was needed.

Otherwise: no failures observed. All four patterns passed exact + judge on the first
live run.
