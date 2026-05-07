"""FastMCP wrapper exposing the lab's hand-rolled agentic RAG pipeline as a
tool consumable from Claude Desktop / Cursor.

Ported from shaneliuyx/rag mcp_server/server.py. Original exposed Chroma
+ Ollama; this port exposes the Qdrant + oMLX pipeline from
baseline_handrolled.py.

Usage:
  # Direct invocation (manual smoke test):
  python src/mcp_server.py

  # Claude Desktop integration: see mcp-config.json at lab root for the
  # registration snippet to drop into ~/Library/Application Support/Claude/
  # claude_desktop_config.json under "mcpServers".

Exposes 3 tools:
  - rag_query(query, k=6, allow_corrective=True) → answer + hits + grades
  - rag_status() → collection size + config
  - rag_decompose(query) → JSON sub-query plan (Phase 6 standalone)

Once registered, Claude Desktop / Cursor surfaces these as tools the user
can invoke with @rag_query or via tool-routing.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# Ensure src/ on sys.path when launched by an MCP host
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from baseline_handrolled import (  # noqa: E402
    QDRANT_COLLECTION, _qdrant, answer,
)
from decompose import decompose_query, topo_sort  # noqa: E402

try:
    from fastmcp import FastMCP
except ImportError:
    print("ERROR: fastmcp not installed. Run: uv pip install -e .", file=sys.stderr)
    sys.exit(1)

mcp = FastMCP("agentic-rag",
              dependencies=["qdrant-client", "sentence-transformers",
                            "FlagEmbedding", "openai", "python-dotenv"])


@mcp.tool()
def rag_query(query: str, k: int = 6, allow_corrective: bool = True) -> dict[str, Any]:
    """Answer a question over the configured Qdrant collection using the
    hand-rolled Self-RAG + CRAG pipeline.

    Args:
        query: The user's question.
        k: top-K to keep after rerank (default 6).
        allow_corrective: If False, skip the rewrite-and-retry loop on
                          relevance-grade failures.
    """
    out = answer(query, top_k=k)
    if not allow_corrective:
        out.pop("corrective", None)
        out.pop("next_action", None)
    return {
        "query": query,
        "answer": out["answer"],
        "hits": out["hits"],
        "selfrag": out["selfrag"],
        "grade_hallucination": out["grade_hallucination"],
        "grade_relevance": out["grade_relevance"],
        "decision": out["decision"],
        "next_action": out.get("next_action"),
    }


@mcp.tool()
def rag_status() -> dict[str, Any]:
    """Return collection size + active config for diagnostics."""
    info = _qdrant.get_collection(QDRANT_COLLECTION)
    return {
        "collection": QDRANT_COLLECTION,
        "points_count": info.points_count,
        "indexed_vectors_count": info.indexed_vectors_count,
        "model_sonnet": os.getenv("MODEL_SONNET", "(unset)"),
        "decomposition_enabled": os.getenv("ENABLE_DECOMPOSITION", "0") == "1",
    }


@mcp.tool()
def rag_decompose(query: str) -> dict[str, Any]:
    """Phase 6 standalone: return the JSON sub-query plan for a complex query
    without running the rest of the pipeline. Useful for inspecting how a
    query would decompose before paying for full retrieval + synthesis.
    """
    plan = decompose_query(query)
    ordered = topo_sort(plan)
    return {"query": query, "plan": plan, "topo_ordered": ordered}


if __name__ == "__main__":
    # Stdio transport for Claude Desktop / Cursor (default for FastMCP)
    mcp.run()
