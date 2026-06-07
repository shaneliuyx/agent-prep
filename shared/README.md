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
