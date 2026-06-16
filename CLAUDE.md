# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

`agent-prep` is a lab-driven curriculum monorepo that converts a cloud-infra background into AI Agent / LLM Engineer skills. Each `lab-NN-*/` directory is a **self-contained week** with its own `pyproject.toml` + `uv.lock`. Every lab follows the same shape and that shape is the point:

> scaffold → instrumented implementation → **measured** comparison → committed `RESULTS.md`

The governing ethos is **measured engineering**: every claim is grounded in a number produced by a runnable artifact, not vibes. When you report a result, trace it to its source (`RESULTS.md` row, eval JSON, or the script that computed it). Do not churn a *finished* chapter to adopt a later refactor — labs are append-only history once shipped.

`README.md` holds the curriculum spine (week → lab → status) and the Akshay 6-area hiring-rubric coverage map. Read it for "what week does X." Per-lab measured findings live in each lab's `RESULTS.md`.

## Commands

Labs are **per-directory `uv` projects** — `cd` into the lab first. There is no repo-wide build.

```bash
# Per-lab Python (the common case)
cd lab-03-5-96-gbrain
uv sync                              # install that lab's deps
uv run python src/ingest_agent.py    # run a lab script
uv run pytest                        # run that lab's tests
uv run pytest tests/test_foo.py::test_bar   # single test
uv run coverage run -m pytest && uv run coverage report

# Root-level tests (the shared/tree_index suite lives at repo root: src/test_*.py + conftest.py)
uv run pytest                        # from repo root

# Week-0 smoke test — verifies the whole local stack is up before any lab
python smoke-test.py                 # oMLX chat + BGE-M3 embed + Qdrant + reranker + Phoenix

# Type-check (config in pyrightconfig.json; .venv + shared/ on path)
pyright
```

**Integration vs unit tests.** Tests that need a real LLM endpoint are marked `@pytest.mark.integration` and **skip by default**; they only run when a provider is configured (e.g. `LLM_PROVIDER=openai` / `anthropic-proxy`, see a lab's `tests/conftest.py`). Plain `uv run pytest` runs the offline unit tests. Lint/format per the global Python rules: `ruff`, `black`, `isort`.

**TypeScript labs** (GBrain-backed weeks) run under **Bun**: `bun run <script>.ts`. There is no compile step for lab TS — Bun executes `.ts` directly.

## Architecture

### Lab + shared-library split (the load-bearing convention)

`shared/` holds cross-chapter **infrastructure** so later labs stay lean. The rule (enforced socially via `AGENTS.md`, not tooling):

- **Introduce inline, reuse via import.** The chapter that *first teaches* a pattern (a metric, a router, a reader, a policy) shows it in full, inline in that lab. Later chapters that merely *reuse* plumbing import it from `shared/`. So a concept's first appearance stays fully visible and finished chapters are never re-churned.
- **Read `shared/README.md` before writing new lab code** and import what exists instead of re-implementing. It carries a provenance table (which util was introduced by which chapter, reused by which).
- **Promote on rule-of-three** — infra lands in `shared/` only once a 2nd chapter genuinely needs it.

Key `shared/` modules:
- `llm.py` — provider-agnostic LLM plumbing: `make_client`, `PROFILES` model presets, `resolve(role, default)` (swap models via env with no code change), `chat`, `judge`, `resilient` retry / `LLMUnavailable`, `load_pass_criteria`.
- `gbrain_cli.py` / `gbrain_engine.ts` — GBrain connect/read wrappers (GBrain chapters only).
- Packages (each its own README + tests): `rag_hybrid/` (hybrid-RAG building blocks + `autoconfig` host probe), `tree_index/` (PageIndex tree-index RAG), `phoenix_tracing/` (one-call Phoenix), `agent_loop_tools/` (interrupt/token-accounting), `parity/` (refactor-safety ground-truth freeze).

**Python imports `shared/` via `sys.path`** (no packaging ceremony):
```python
import sys; sys.path.insert(0, "/Users/yuxinliu/code/agent-prep/shared")
from llm import resolve, chat, judge, resilient
```
**TS/Bun** imports it by absolute path: `import { bootstrapEngine } from "/Users/yuxinliu/code/agent-prep/shared/gbrain_engine.ts"`.

### Model tiers (local-first inference)

Models are addressed by **tier alias**, not model name, so a lab is portable across hosts. Tiers resolve through `.env` (`MODEL_OPUS` / `MODEL_SONNET` / `MODEL_HAIKU` / `MODEL_VMLX`) — see `run_local.py` `TIERS` and `shared/llm.py` `PROFILES`. Locally these all point at **oMLX on `:8000`** (OpenAI + Anthropic API surface); `vmlx` is a second backend on `:8003`. Cloud keys (`OPENAI_API_KEY` / `ANTHROPIC_API_KEY`) stay blank until the few weeks that need frontier models (W7/W8/W7.3/W9.5); the whole program has a ~$13 cloud cap.

### Local services the labs assume

- **oMLX** `:8000` (chat + embeddings), **vMLX** `:8003` — local inference
- **Qdrant** `:6333` (OrbStack/Docker) — vector DB
- **Phoenix** `:6006` — tracing/observability
- **Memory infra** (W3.5.5 / .8 / .9): `guild` (Go MCP), EverCore (`:1995`), HyperMem (`:1996`)

Run `python smoke-test.py` to confirm the core four are up before starting a lab.

## Gotchas

- **`gbrain/` is a cloned external tool, NOT lab content.** It is gitignored and has its own git history + its own `CLAUDE.md`. The labs reference it only for MCP schemas / as a memory backend. Do not treat changes there as part of this repo, and follow `gbrain/CLAUDE.md` if you genuinely need to work inside it.
- **`data/` and `models/` are gitignored** (large artifacts). A lab's test set may be force-added as a small `golden_*.json`; the bulk corpus is regenerated locally.
- **`AGENTS.md` defines a `guild` MCP workflow** (quest = tasks, lore = persistent memory across sessions/agents). If that MCP server is wired, `AGENTS.md` asks you to `guild_session_start(project="agent-prep")` first and use `quest_*` instead of the built-in task tools. This is an agent-coordination convention, not part of lab code.
- **Secrets live in per-lab `.env`** (gitignored; each lab ships `.env.example`). The repo `.env.example` documents the canonical keys.
