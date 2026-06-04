# lab-03-5-96-gbrain — a memory-augmented agent over GBrain MCP

Companion lab for **Week 3.5.96 — Self-Wiring Memory (GBrain)**. The transferable
lesson: **how a future agent you build uses a memory system (GBrain) as a tool over
MCP** — not by hand-editing files, and not with a bespoke converter.

## What it does

A standalone **smolagents `CodeAgent`** (brain = local oMLX) connects to GBrain's
**MCP server** (`gbrain serve`) and, given only raw heterogeneous sources
(`~/brain/sources/*` — emails, transcripts, any shape):

1. reads the raw text (`read_sources` tool)
2. converts it to structured GBrain pages — two-layer + `[[wikilinks]]` — via a
   local LLM (`extract_pages` tool, oMLX)
3. stores each page with the `put_page` MCP tool (GBrain chunks, embeds, auto-links)
4. answers a question with the `query` MCP tool

The graph is then deterministically wired from the wikilinks (`gbrain extract links`,
zero LLM calls) and traversable (`gbrain graph-query`).

## Files

- `src/probe_mcp.py` — smallest proof: a Python MCP client lists `gbrain serve`'s tools
- `src/ingest_agent.py` — the agent (thin agent, fat tools; smolagents + MCP + oMLX)
- `.env.example` — config (DB + oMLX + LLM); copy to `.env` (gitignored)

## Prereqs

- GBrain initialized per **Week 3.5.96 Phase 1** (Postgres engine on Docker/OrbStack,
  embeddings on oMLX). `gbrain` on PATH (`~/.bun/bin`).
- oMLX serving on `:8000` (chat model + an embedding model).
- Raw samples under `~/brain/sources/` (the chapter ships two: an email thread + a transcript).

## Run

```bash
cp .env.example .env        # fill GBRAIN_DATABASE_URL, OLLAMA_*/LLM_* + oMLX key
uv sync
uv run python src/probe_mcp.py      # list GBrain's MCP tools
uv run python src/ingest_agent.py   # read → extract → put_page → query

export PATH="$HOME/.bun/bin:$PATH"
gbrain extract links --source db && gbrain stats   # wire + count the graph
gbrain graph-query deals/acme-seed                 # traverse typed edges
```

## Stack + key choices

- **smolagents `CodeAgent`** — chosen because the brain (oMLX) has **no native
  tool-calling**; CodeAgent has the LLM write code that calls tools, so it doesn't
  need function-calling (PydanticAI / OpenAI Agents SDK would require a tool-calling
  model — e.g. via VibeProxy/Haiku). `use_structured_outputs_internally=True` for oMLX.
- **`ToolCollection.from_mcp`** (`smolagents[mcp]`) loads GBrain's MCP tools; filtered
  to the ~5 needed (it exposes ~70).
- **thin agent, fat tools** — the small model can't read files + extract + compose
  markdown in one loop; the hard work is in `read_sources` / `extract_pages` tools.

See `RESULTS.md` for measured numbers + the Bad-Case Journal (the wikilink-mandate
finding is the headline).
