# W3.5.96 — results: a memory-augmented agent over GBrain MCP

All numbers are from a real run on this machine: GBrain 0.42.25.0 (Postgres engine
on Docker/OrbStack pgvector), embeddings + agent brain on local **oMLX :8000**
(`ollama:nomicai-modernbert-embed-base-bf16`, 768d), agent = **smolagents
`CodeAgent`** on `Qwen2.5-Coder-14B`. No cloud, no metered API.

## TL;DR

A **future agent** (smolagents, *not* Claude Code) uses GBrain as its memory layer
over **MCP**, end-to-end:

| stage | tool | result |
|-------|------|--------|
| connect | Python MCP client → `gbrain serve` (stdio) | **~70 tools** exposed; 5 needed present |
| read | `read_sources()` (local tool) | 2 raw files (email thread + transcript), any shape |
| extract | `extract_pages()` (oMLX, local tool) | raw → structured GBrain pages (JSON) |
| store | `put_page` (MCP) | **10 pages** written by the agent |
| wire | `gbrain extract links` | **11 typed edges** (self-wiring graph) |
| answer | `query` (MCP) | `deals/acme-seed` @ **score 0.93**, correctly cites `[[people/sam-okafor]]` anchoring |

**Headline finding:** the framework + MCP wiring is the easy part. The graph only
materializes if the **extraction prompt hard-mandates `[[wikilinks]]`** — without
that, the 14B writes prose and `extract links` produces **0 edges**; with it, **11**.
Graph quality = extraction quality, not plumbing.

## Pipeline (`src/ingest_agent.py`)

Idiomatic smolagents — **thin agent, fat tools**. The agent's own code is ~4 lines:
```python
raw = read_sources()
pages = extract_pages(raw)
for p in pages: put_page(slug=p["slug"], content=p["content"])
final_answer(query(query="Who is anchoring the acme-seed round and on what terms?"))
```
The hard work lives in tools: `read_sources` (file I/O), `extract_pages` (the oMLX
raw→structured call), and GBrain's MCP tools (`put_page`, `query`).

## Measured output

- **Probe** (`src/probe_mcp.py`): a plain Python MCP client spawned `gbrain serve`
  and listed ~70 tools; `put_page, add_link, add_timeline_entry, query, search` all present.
- **Run 1 (no wikilink mandate):** agent wrote 5 pages, but `extract links` →
  **`Links: created 0`** — pages were prose ("Alice Chen, founder of Acme AI"), no
  `[[wikilinks]]`. Self-wiring graph empty.
- **Run 2 (wikilink-mandate prompt + few-shot):** 10 pages, `extract links` →
  **`Links: created 11`**. `deals/acme-seed` now: *"Seed round for `[[companies/acme-ai]]`…
  `[[people/sam-okafor]]` is anchoring the remainder."*
- **query** "who is anchoring acme-seed?" → top hit `deals/acme-seed` score **0.926**, answer correct.
- **graph-query** `deals/acme-seed` → multi-hop typed traversal: `--invested_in->`,
  `--works_at->`, `--mentions->` across people/companies (depth 1–5).

## Run process

```bash
# prereqs: GBrain initialized (Postgres engine on Docker, oMLX embeddings) — see
# Week 3.5.96 Phase 1. Raw samples under ~/brain/sources/.
cp .env.example .env   # fill GBRAIN_DATABASE_URL, OLLAMA_*/LLM_* (oMLX), key
uv sync

uv run python src/probe_mcp.py      # 1. prove MCP client ↔ gbrain serve
uv run python src/ingest_agent.py   # 2. agent: read → extract → put_page → query

# wire + inspect the graph the agent built
export PATH="$HOME/.bun/bin:$PATH"
gbrain extract links --source db && gbrain stats
gbrain graph-query deals/acme-seed
```

## Bad-Case Journal (real, observed)

| # | symptom | root cause | fix |
|---|---------|-----------|-----|
| 1 | `extract links` → `Links: 0`; pages are prose | one-shot 14B extraction ignored the wikilink rule; wrote names as plain text + used `<!-- timeline -->` not `---` | extraction prompt: few-shot example WITH `[[wikilinks]]` + hard rule "a page with zero wikilinks is invalid" → Links 0→11 |
| 2 | fully-autonomous `CodeAgent` failed (naive regex extractor, hardcoded dates) + `InterpreterError: import pathlib not allowed` | asked a 14B to read files AND write an extractor AND compose markdown in one code loop; CodeAgent sandbox blocks `pathlib`/`json` | **thin agent, fat tools** — move file I/O + extraction into `@tool`s; agent only orchestrates |
| 3 | 14B confused / huge prompt | GBrain exposes ~70 MCP tools; loading all into the agent drowns it | filter `ToolCollection.from_mcp` to the ~5 needed (`put_page`, `query`, …) |
| 4 | `ModuleNotFoundError: mcpadapt` on `ToolCollection.from_mcp` | smolagents MCP support is an extra | depend on `smolagents[mcp]` |
| 5 | spawned `gbrain serve` can't reach oMLX/DB | an MCP server is a separate process — no shell-env inheritance | pass `GBRAIN_DATABASE_URL`/`OLLAMA_*` via `StdioServerParameters(env=…)` |
| 6 | oMLX emits `<code>` smolagents mis-parses (issue #1851) | oMLX has no native tool_calls | `CodeAgent(use_structured_outputs_internally=True)` |

## Interview soundbite (principle-level)

> The framework and MCP plumbing took an afternoon; the graph quality came down to
> one prompt rule. A capable-but-small local model will happily store well-written
> prose pages and silently produce a zero-edge "graph" — because it dropped the
> wikilinks the extractor needed. The lesson for wiring an LLM into a structured
> memory system: the *contract* the extraction prompt enforces (every entity is a
> typed link) is what makes the graph real, not the storage layer. Measure edges,
> not pages.
