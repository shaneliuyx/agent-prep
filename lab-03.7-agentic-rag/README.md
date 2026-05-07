# lab-03.7-agentic-rag

Hand-rolled Self-RAG + CRAG pipeline (Phase 5) + LangGraph canonical (Phase 1-4) +
query decomposition mini-lab (Phase 6) + MCP server wrapper (Phase 7).

Companion to: [Week 3.7 — Agentic RAG](../../Documents/Obsidian%20Vault/Agent%20Development%20Curriculum/Week%203.7%20-%20Agentic%20RAG.md).

## Provenance

The hand-rolled pipeline (`src/baseline_handrolled.py` + `src/decompose.py`) is
ported from the user's older personal repo `github.com/shaneliuyx/rag` (commit
`dae7d6f`, 2025-08-17). The 2025 repo had a working Self-RAG + CRAG pipeline on
Chroma + Ollama-Gemma2:2b; this port adapts it to the 2026 curriculum stack
(Qdrant + oMLX + `shared/rag_hybrid` for encoder + reranker).

The MCP server wrapper (`src/mcp_server.py` + `mcp-config.json`) is also lifted
from the old repo and adapted — exposes the lab's `answer()` as a tool
consumable from Claude Desktop / Cursor.

## Pipeline overview

```
query
  ↓ ComplexityDecider (heuristic)
  ↓ optional: LLMDecomposer → topo-sort sub-queries (Phase 6)
  ↓ MultiRetrieve (dense + RRF fusion)
  ↓ Rerank (BGE-reranker-v2-m3)
  ↓ Synthesize (bullets with [#i] citations + drift filter)
  ↓ SelfRAG checks (faithfulness + citation + coverage)
  ↓ Grade hallucination + grade relevance
  ↓ if not pass → CorrectiveRAG (rewrite query + retry)
  ↓ if still not pass → suggest web-search fallback (host-level handoff)
output: {answer, hits, selfrag, grade_hallucination, grade_relevance, next_action?}
```

## Setup

```bash
cd ~/code/agent-prep/lab-03.7-agentic-rag
cp .env.example .env  # edit API key
uv venv && source .venv/bin/activate
uv pip install -e .

# Pre-req: Qdrant running, oMLX serving MODEL_SONNET, a Qdrant collection
# ingested. Reuses lab-03's bge_m3_hnsw or any other shared/rag_hybrid collection.
```

## Run

```bash
# Hand-rolled pipeline (Phase 5)
python src/baseline_handrolled.py "What did Buffett write about non-controlled businesses in 2023?"

# Standalone query decomposition (Phase 6)
python src/decompose.py "Compare BNSF Railway revenue vs Berkshire Energy revenue in 2023"

# MCP server (Phase 7) — register in Claude Desktop config
python src/mcp_server.py
```

## Files

- `src/baseline_handrolled.py` — full hand-rolled pipeline (300 LOC, single file). Ports `graph/builder.py` + nodes from old repo.
- `src/decompose.py` — standalone JSON query planner with `depends_on` + topological execution. Ports `graph/nodes/decompose_llm.py`.
- `src/mcp_server.py` — FastMCP wrapper exposing `rag_query` + `rag_status` tools. Ports `mcp_server/server.py`.
- `mcp-config.json` — Claude Desktop / Cursor MCP server registration snippet.

## Status

Built 2026-05-07 from scratch via batch port of `shaneliuyx/rag@dae7d6f`. Not
yet end-to-end-tested against an eval set; first run will populate
`results/RESULTS.md`. The W3.7 chapter Phase 1-4 (LangGraph canonical) is not
yet ported here either — those phases use a separate `langgraph_official.py`
that hasn't been authored yet (see [W3.7 chapter Phase 1](../../Documents/Obsidian%20Vault/Agent%20Development%20Curriculum/Week%203.7%20-%20Agentic%20RAG.md)).
