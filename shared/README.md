# agent-prep/shared — cross-chapter infrastructure

Reusable code extracted from the lab chapters, so later chapters stay lean. Two shapes live here:
**flat infra modules** (LLM client/retry/presets, GBrain connect/read — documented below) and
larger **packages** (`rag_hybrid/`, `tree_index/`, `phoenix_tracing/`, `agent_loop_tools/`,
`web_toolkit/`, each its own folder with a README + tests). What does NOT live here: the **teaching primitives** —
the metric/routing/reader/policy logic each chapter *introduces* stays in that chapter's lab so
the reader sees the mechanism.

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
| `web_search.py` | py | cached web-search backend (SearXNG → Tavily → DuckDuckGo) + on-disk reproducibility cache + `rerank_results` (cross-encoder rerank of result strings). `searxng/` ships a ready `docker-compose.yml` for the free local backend. **For agent action-spaces** (structured results + fetch/batch-fetch/browse) see the `web_toolkit/` package | W3.7 `baseline_handrolled.py` + `crag_variant.py` (CRAG web fallback) → any chapter with a web fallback |

`llm.py` and `web_search.py` are provider-agnostic (every chapter can use them). `gbrain_*` is
GBrain-specific — a non-GBrain chapter (e.g. W3.5.95) imports only `llm`. `web_search.py` keeps the
reranker MODEL out (`rerank_results` takes it as a param) so the module stays light — the reranker
lives in the `rag_hybrid/` package.

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

### `web_search.py` (Python) — cached web-search backend *(promoted from W3.7)*
| symbol | signature | what it does |
|---|---|---|
| `web_search` | `(query, k=4) -> list[str]` | cached backend call; precedence `SEARXNG_URL` → `TAVILY_API_KEY` → DuckDuckGo. Cache hit replays; miss fetches live + persists. |
| `rerank_results` | `(query, docs, top_k, reranker) -> list[str]` | cross-encoder rerank of result STRINGS — the accuracy half of the web fallback (rerank each sub-query's docs against ITS sub-query so each entity's figure surfaces). `reranker` = a `rag_hybrid` CrossEncoderReranker, passed in to keep this module torch-free. |
| `cache_lookup` / `cache_store` | `(key) -> list[str]\|None` / `(key, docs) -> None` | generic disk-cache access for callers that cache at a HIGHER level than one query (e.g. a fanned-out planned query whose whole pool replays as one unit). Honor `WEB_CACHE`. |
| `web_cache_key` | `(query, k) -> str` | per-query cache key, backend+config aware (changing engine/language/backend invalidates). |
| `web_cache_enabled` | `() -> bool` | the `WEB_CACHE` env toggle. |

Determinism + accuracy rationale and the `searxng/` docker setup are documented in the W3.7
chapter §3.3.1 and `searxng/README.md`. Env: `SEARXNG_URL`, `SEARXNG_LANGUAGE` (default `en`),
`SEARXNG_ENGINES` (optional allowlist), `TAVILY_API_KEY`, `WEB_CACHE` (1/0), `WEB_CACHE_PATH`.

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

## Packages (subdirectories)

Larger libraries — each its own package with a dedicated README + tests. Read the package's own
README for its API; the table is the index.

| package | what | extracted from | docs |
|---|---|---|---|
| `rag_hybrid/` | modular hybrid-RAG building blocks: char/sentence chunkers, BGE-M3 hybrid + dense encoder (lazy), Qdrant schema, RRF `fusion`, cross-encoder `rerank` (fp16-opt-in), `retrieve` (auto hybrid/dense + RRF + optional rerank), `ingest`, system-aware `autoconfig` | W2 `lab-02-rerank-compress` + W2.5 `lab-02-5-graphrag` | `rag_hybrid/README.md` |
| `tree_index/` | PageIndex-pattern tree-index RAG primitives: `builder`, `summary_index`, `page_vector_index`, `entity_index`, `ensemble`, `agentic` search, `prompts` | W2.7 `lab-02-7-pageindex` (lifted tree judge 0.44 → 0.885) | `tree_index/README.md` |
| `phoenix_tracing/` | one-call Phoenix observability — wraps `register()` + OpenAI/LangChain instrumentors + span helpers for any RAG/agent lab | W3 `lab-03-rag-eval/src/05_trace.py` | `phoenix_tracing/README.md` |
| `agent_loop_tools/` | iterative agent-loop primitives: `interrupt_state` (pause/resume), `token_accounting` | ported from gnhf (MIT) → agent-loop labs | `agent_loop_tools/README.md` |
| `web_toolkit/` | agent-facing web tools with structured results: `web_search` (SearXNG→Tavily→DDG, ranked), `web_fetch` + `web_batch_fetch` (scrapling CLI), `web_browse` (agent-browser CLI). CLI-driven backends, typed dataclass results, no torch | synthesized from `Wade11s/pi-web-toolkit` + promoted from `web_search.py` → W4+ ReAct / tool-harness labs | `web_toolkit/README.md` |

**Loose helpers (repo-root of `shared/`):**
- `guild_client.py` — Python wrapper over guild's MCP stdio interface (schema-verified against `list_tools()`).
- `parity_baseline.py` + `parity/` — freeze ground-truth state (Qdrant point counts, sample vector signatures) before a refactor, for mechanical before/after diffing. Baseline at `parity/pre_refactor.json`.

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
