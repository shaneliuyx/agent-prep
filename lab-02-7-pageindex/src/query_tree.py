"""Tree-search retrieval — thin Berkshire-specific wrapper around
shared/tree_index/AgenticTreeRetriever.

The agentic-loop logic, tool schema, and system prompt all live in
`shared/tree_index/`. This file just supplies:
  - the lab's `tree.json` path
  - the lab's PDF page provider
  - the lab's model client + model name
  - thin module-level `answer(query)` function for compare_three.py compatibility

Refactored 2026-05-07 from inline implementation to shared-lib import after
the W2.7 optimization run hit judge=0.79; see W2.7 chapter §4.3.3 for the
extracted-pattern rationale.

Public signature unchanged: answer(query) -> {"answer": str, ...}.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

# Bootstrap shared/tree_index onto sys.path
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "shared"))

from tree_index import (  # noqa: E402
    AGENTIC_SYSTEM_TEMPLATE,
    AGENTIC_SYSTEM_TEMPLATE_V2,
    AgenticTreeRetriever,
    EntityIndex,
    TreeIndex,
)

load_dotenv()
omlx = OpenAI(base_url=os.getenv("OMLX_BASE_URL"), api_key=os.getenv("OMLX_API_KEY"))
# Tree backend uses MODEL_TREE (isolated from vector/graph's MODEL_SONNET) to
# avoid Qwen3.6 KV-cache pollution observed when all 3 backends shared one model.
# Falls back to MODEL_SONNET if MODEL_TREE is unset.
MODEL = os.getenv("MODEL_TREE") or os.getenv("MODEL_SONNET")

_PDF_CACHE: dict[str, list[str]] = {}


def _pdf_pages(pdf_path: str) -> list[str]:
    if pdf_path not in _PDF_CACHE:
        reader = PdfReader(pdf_path)
        _PDF_CACHE[pdf_path] = [p.extract_text() or "" for p in reader.pages]
    return _PDF_CACHE[pdf_path]


def _make_page_provider(pdf_path: str):
    pages = _pdf_pages(pdf_path)

    def provider(start: int, end: int) -> str:
        start_idx = max(0, int(start) - 1)
        end_idx = min(len(pages), int(end))
        if end_idx < start_idx + 1:
            return f"[ERROR] Invalid range: end ({end}) < start ({start})"
        return "\n\n".join(
            f"[page {i+1}]\n{pages[i]}" for i in range(start_idx, end_idx)
        )

    return provider


_RETRIEVER_CACHE: dict[str, AgenticTreeRetriever] = {}


def _get_retriever(tree_path: str, pdf_path: str, *, v2: bool = True) -> AgenticTreeRetriever:
    """Build (and cache) a retriever. v2 wires entity-graph + auto-merge tools."""
    key = f"{tree_path}|{pdf_path}|v2={v2}"
    if key in _RETRIEVER_CACHE:
        return _RETRIEVER_CACHE[key]

    tree = json.loads(Path(tree_path).read_text())
    page_provider = _make_page_provider(pdf_path)
    kwargs = dict(
        tree=tree,
        page_provider=page_provider,
        model_client=omlx,
        model_name=MODEL or "",
        debug_log_path="/tmp/tree_debug.log",
    )
    if v2:
        # Build a body-text page provider WITHOUT [page N] headers for
        # entity extraction — cleaner regex matching on hyphenated proper
        # nouns like "Coca-Cola" that headers would interrupt.
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        pages_raw = [p.extract_text() or "" for p in reader.pages]

        def raw_provider(s: int, e: int) -> str:
            sp = max(0, int(s) - 1)
            ep = min(len(pages_raw), int(e))
            return "\n\n".join(pages_raw[i] for i in range(sp, ep))

        ti = TreeIndex(tree)
        ei = EntityIndex(ti, page_provider=raw_provider)
        kwargs["tree_index"] = ti
        kwargs["entity_index"] = ei
        kwargs["system_prompt"] = AGENTIC_SYSTEM_TEMPLATE_V2
    else:
        kwargs["system_prompt"] = AGENTIC_SYSTEM_TEMPLATE

    retriever = AgenticTreeRetriever(**kwargs)
    _RETRIEVER_CACHE[key] = retriever
    return retriever


def answer(query: str, tree_path: str = "data/tree.json",
           pdf_path: str = "data/brk-2023-ar.pdf", *, v2: bool = True) -> dict:
    return _get_retriever(tree_path, pdf_path, v2=v2).answer(query)


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What was Berkshire's net earnings in 2023?"
    out = answer(q)
    print(json.dumps(out, indent=2, default=str))
