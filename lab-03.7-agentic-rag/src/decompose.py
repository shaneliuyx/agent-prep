"""Standalone JSON sub-query planner (Phase 6 mini-lab).

Ported from shaneliuyx/rag graph/nodes/decompose_llm.py — the original
implemented this against Ollama-Gemma2:2b; this port adapts to oMLX
OpenAI-compatible client + JSON-mode response_format.

Decomposes a complex query into 2-6 atomic sub-queries with explicit
`depends_on` dependencies + `type` ("lookup" | "synthesis"). The W2.7
chapter §4.3.3 "Reusable design patterns" calls out query-planning-with-
topo-sort as a pattern not yet covered elsewhere in the curriculum.

Differs from W3.7's LangGraph `rewrite_question` node (single-shot
reformulation): this is real *planning* — output is a DAG that can be
executed in topological order with citation remapping across sub-queries.

Usage:
  python src/decompose.py "Compare BNSF Railway revenue vs Berkshire Energy revenue in 2023"

Returns:
  [
    {"id": "q1", "text": "BNSF Railway 2023 revenue", "depends_on": [], "type": "lookup"},
    {"id": "q2", "text": "Berkshire Energy 2023 revenue", "depends_on": [], "type": "lookup"},
    {"id": "q3", "text": "compare BNSF vs Berkshire Energy 2023 revenue", "depends_on": ["q1", "q2"], "type": "synthesis"}
  ]
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
omlx = OpenAI(base_url=os.getenv("OMLX_BASE_URL"), api_key=os.getenv("OMLX_API_KEY"))
MODEL = os.getenv("MODEL_SONNET") or ""


DECOMPOSE_PROMPT = """You are a query-planning assistant. Decompose the user's
question into 2-6 atomic sub-queries that can each be answered from documents.

Return a JSON object with a single key "sub_queries" whose value is an array.
Each item has:
  - id: "q1", "q2", ... (sequential, unique)
  - text: the atomic sub-query (string)
  - depends_on: array of ids this sub-query needs answered first (often empty)
  - type: "lookup" (factoid retrieval) or "synthesis" (combines other sub-queries)

Rules:
- Mark "synthesis" type for any sub-query that compares / contrasts /
  combines results from other sub-queries; populate its depends_on with the
  ids of those sub-queries.
- Mark "lookup" type for atomic factoid sub-queries with no dependencies.
- For simple factoid queries that don't decompose, return a single item.
- Output strict JSON, no prose preamble.

User query:
{query}

JSON output:"""


def decompose_query(query: str) -> list[dict[str, Any]]:
    """Returns a list of {id, text, depends_on, type} sub-queries.

    Falls back to [{"id": "q1", "text": query, "depends_on": [], "type": "lookup"}]
    on any LLM / JSON-parse error so callers can always proceed.
    """
    try:
        resp = omlx.chat.completions.create(
            model=MODEL, temperature=0.2, max_tokens=512,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": DECOMPOSE_PROMPT.format(query=query)}],
        )
        raw = (resp.choices[0].message.content or "").strip()
        parsed = json.loads(raw)
        items = parsed.get("sub_queries", [])
        cleaned = []
        for i, it in enumerate(items, start=1):
            cleaned.append({
                "id": str(it.get("id", f"q{i}")),
                "text": str(it.get("text", "")).strip() or query,
                "depends_on": list(it.get("depends_on", []) or []),
                "type": str(it.get("type", "lookup")).lower(),
            })
        if not cleaned:
            return [{"id": "q1", "text": query, "depends_on": [], "type": "lookup"}]
        return cleaned
    except Exception as e:  # noqa: BLE001
        print(f"[decompose] fallback to single query — {type(e).__name__}: {e}",
              file=sys.stderr)
        return [{"id": "q1", "text": query, "depends_on": [], "type": "lookup"}]


def topo_sort(plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Topological order: lookups first, synthesis after their dependencies."""
    indeg: dict[str, int] = {n["id"]: 0 for n in plan}
    for n in plan:
        indeg[n["id"]] = len(n.get("depends_on", []))
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    ready = [n for n in plan if indeg[n["id"]] == 0]
    while ready:
        cur = ready.pop(0)
        out.append(cur)
        seen.add(cur["id"])
        for n in plan:
            if n["id"] in seen or n in ready:
                continue
            if all(d in seen for d in n.get("depends_on", [])):
                ready.append(n)
    # Residual cycles: append remaining nodes in original order
    for n in plan:
        if n["id"] not in seen:
            out.append(n)
    return out


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "Compare BNSF Railway and Berkshire Energy 2023 revenue"
    plan = decompose_query(query)
    ordered = topo_sort(plan)
    print(json.dumps({"plan": plan, "topo_ordered": ordered}, indent=2))
