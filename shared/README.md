# agent-prep/shared — cross-chapter infrastructure

Reusable **plumbing** extracted from the lab chapters, so later chapters stay lean. This holds
infrastructure only (LLM clients, retry, model presets, GBrain connect/read) — **not** the
teaching primitives. The metric/routing/reader logic each chapter introduces stays in that
chapter's lab so the reader sees the mechanism.

## The rule: introduce inline, reuse via import

The chapter that **first teaches** a pattern shows it in full, inline. Later chapters that
**merely reuse** it import from here. So a finished chapter is never re-churned to satisfy a
refactor, and every concept's first appearance stays fully visible.

Extraction follows **rule-of-three**: a util lands here only once ≥2 chapters genuinely need it.
Everything below already had 2+ real call sites when extracted.

## Modules

| module | lang | what | provenance (introduced → reused by) |
|---|---|---|---|
| `llm.py` | py | OpenAI-compatible client, model-preset registry (`resolve`), `resilient` retry, `chat`, `judge`, `load_pass_criteria` | W3.5.96 `reader_ab.py`/`answer_route_ab.py`/`verify_arch.py` + W3.5.95 client pattern → any chapter doing gen/judge |
| `gbrain_cli.py` | py | `gbrain_get` / `gbrain_query_slugs` wrappers + snippet-guarded `build_context` | W3.5.96 `ground_truth_ab.py`/`answer_route_ab.py` → any **GBrain** chapter |
| `gbrain_engine.ts` | ts | `bootstrapEngine()` — the standard GBrain connect sequence (one place) | W3.5.96 `policy_eval.ts`/`route_eval.ts`/`query_policy.ts` → any **GBrain** chapter |

`llm.py` is provider-agnostic (every chapter can use it). `gbrain_*` is GBrain-specific — a
non-GBrain chapter (e.g. W3.5.95) imports only `llm`.

## API reference — every utility

### `llm.py` (Python) — provider-agnostic LLM plumbing
| symbol | signature | what it does |
|---|---|---|
| `make_client` | `(base_url=None, api_key=None) -> OpenAI` | OpenAI-compatible client; unset args fall back through `LLM_BASE_URL`→`OPENROUTER_BASE_URL`→`OMLX_BASE_URL` (key similar, non-empty `EMPTY` sentinel so it builds with no `.env`). |
| `PROFILES` | `dict[str, (base,key,model)]` | named model presets: `haiku`, `opus` (VibeProxy→Claude), `14b`, `qwen` (oMLX). Add a model = one line. |
| `resolve` | `(role, default_profile) -> (client, model, label)` | resolve a ROLE (e.g. `"GEN"`/`"JUDGE"`) with no code change: `<ROLE>=haiku` preset, or raw `<ROLE>_MODEL` (+ optional `<ROLE>_BASE_URL`/`_API_KEY`). |
| `chat` | `(client, prompt_or_messages, model, temperature=0.0) -> str` | one completion; accepts a raw prompt string or a messages list. |
| `judge` | `(client, answer, criteria, model) -> bool` | LLM judge — PASS/FAIL of `answer` against `criteria` (retries through drops). |
| `JUDGE_TMPL` | `str` | the judge prompt template (`{criteria}`, `{answer}`). |
| `resilient` | `(fn, *args, retries=4, backoff=2.0)` | retry an LLM call through transient connection drops; raises `LLMUnavailable` if it never recovers. |
| `LLMUnavailable` | `Exception` | endpoint refused after retries — caller should SKIP the item, not crash. |
| `load_pass_criteria` | `(ground_truth_path) -> dict[str,str]` | question text → `pass_criteria`, from a W2.7-style `eval_ground_truth.json`. |

### `gbrain_cli.py` (Python) — GBrain CLI wrappers + read assembly *(GBrain chapters only)*
| symbol | signature | what it does |
|---|---|---|
| `gbrain_query_slugs` | `(q, limit) -> list[str]` | hybrid retrieval — ranked slugs only (snippets are too thin to ground). |
| `gbrain_get` | `(slug) -> str` | full page body via `gbrain get <slug>` (NOT the truncated `query --json` snippet). |
| `build_context` | `(slugs, max_body_chars=0, min_body_chars=80) -> str` | assemble reader context from full bodies; raises `SnippetRegression` if any body is suspiciously short; `max_body_chars>0` caps each body for a small-context generator. |
| `SnippetRegression` | `Exception` | a reader injected a truncated snippet instead of a full `gbrain get` body. |
| `server_env` | `() -> dict` | env for shelling `gbrain` (puts `~/.bun/bin` on PATH; defaults DB + embed endpoint). |

### `gbrain_engine.ts` (TS/Bun) — GBrain engine bootstrap *(GBrain chapters only)*
| symbol | signature | what it does |
|---|---|---|
| `bootstrapEngine` | `() -> Promise<{engine, runEval, config}>` | connect to GBrain exactly as the CLI does (load config → create engine → connect → wire gateway). `GBRAIN_SRC` env overrides the gbrain source path. |
| `Bootstrapped` | `interface` | return shape: `{ engine, runEval, config }`. |

## Use it

**Python** (flat import via `sys.path`; no packaging ceremony):
```python
import sys
sys.path.insert(0, "/Users/yuxinliu/code/agent-prep/shared")
from llm import resolve, chat, judge, resilient, LLMUnavailable, load_pass_criteria
from gbrain_cli import gbrain_get, build_context          # GBrain chapters only

gen_client, gen_model, _ = resolve("GEN", "haiku")        # GEN=14b uv run … to switch, no code change
ans = resilient(chat, gen_client, prompt, gen_model)
```

**TypeScript / Bun** (absolute-path import, same style as the labs' `gbrain/src` imports):
```typescript
import { bootstrapEngine } from "/Users/yuxinliu/code/agent-prep/shared/gbrain_engine.ts";
const { engine, runEval } = await bootstrapEngine();
```

## Env (same `.env` the labs already use)
- **endpoints** — `LLM_BASE_URL` / `OPENROUTER_BASE_URL` / `OMLX_BASE_URL` (chain-resolved), `*_API_KEY`.
- **model presets** — `GEN=` / `JUDGE=` ∈ {`haiku`, `opus`, `14b`, `qwen`}; raw override `GEN_MODEL=…`.
- **GBrain** — `GBRAIN_DATABASE_URL`, `GBRAIN_BIN`, `GBRAIN_SRC` (TS bootstrap path override).

## Maintenance
- Keep utils small + stable; the provenance table is what makes a breaking change auditable
  (you can see every chapter that imports a given util).
- Add a model = one line in `llm.PROFILES`. Add a util = only when a 2nd chapter needs it.
