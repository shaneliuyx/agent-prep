"""Tree-search retrieval via agentic tool-calling loop (PageIndex pattern).

Replaces the prior greedy single-shot navigate+answer with a multi-turn agent
loop that:
  1. Sees the full tree-of-contents (compact: id/title/page-range/summary)
  2. Calls get_page_content(start, end) to fetch raw PDF text for candidates
  3. Iterates until ready to answer with citation [pages X-Y] OR refuses

Three structural improvements over the prior greedy navigate():
  - Body text is visible to the decision-maker (was: only summaries)
  - Multi-section synthesis is possible (was: locked to one leaf)
  - TOC-trap fix: explicit AGENTIC_SYSTEM rule "TOC pages list section names
    but DO NOT contain answers — descend past page 3 to actual content"

Public signature unchanged: answer(query) -> {"answer": str, ...}.
Compatible with compare_three.py call site.

PageIndex reference: examples/agentic_vectorless_rag_demo.py.
Smoke-tested model: Qwen3.6-35B-A3B-UD-MLX-4bit (4/4 PASS via
scripts/test_qwen36.py — JSON, tools, multi-turn, 16K-context).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

load_dotenv()
omlx = OpenAI(base_url=os.getenv("OMLX_BASE_URL"), api_key=os.getenv("OMLX_API_KEY"))
# Tree backend uses MODEL_TREE (isolated from vector/graph's MODEL_SONNET) to
# avoid Qwen3.6 KV-cache pollution observed when all 3 backends shared one model.
# Falls back to MODEL_SONNET if MODEL_TREE is unset.
MODEL = os.getenv("MODEL_TREE") or os.getenv("MODEL_SONNET")

MAX_ITERATIONS = 6
MAX_PAGE_RANGE_CHARS = 8000


AGENTIC_SYSTEM = """You answer questions about a long structured document by
navigating its Table of Contents tree and fetching raw page text on demand.

You see a tree of sections, each with: node_id, title, page range, summary.
You have one tool: get_page_content(start_page, end_page) — fetches raw text.

Workflow:
1. Read the tree to identify candidate page ranges most likely to contain the
   answer. Many sections may look relevant; be specific.
2. Call get_page_content(start_page, end_page) for the most promising range.
   Page ranges should be focused (3-10 pages typical, 20 pages absolute max
   per call).
3. If the fetched text contains the answer, write the final answer with an
   inline citation in the form [pages X-Y].
4. If the fetched text contains the answer fully, write it. If it contains
   PARTIAL information that contributes to the answer, ACCUMULATE it — don't
   throw it away. Many cross-section synthesis questions (e.g. "what does
   Buffett write about non-controlled businesses?") have their answer spread
   across multiple sub-sections after the recursive split — each fetch
   contributes one piece. Track what each fetch tells you.
4a. After 3+ fetches that each contribute partial information, SYNTHESIZE the
    final answer by combining the fragments you've collected. Do not refuse
    just because no single fetch contained the complete answer — combining
    fragments across fetches is the intended workflow for synthesis questions.
    Cite the page ranges you fetched.
4b. Only refuse with "insufficient context" if (a) the tree has no plausibly
    relevant section, or (b) you've fetched all plausibly relevant sections
    and none contain even partial information about the question topic.
5. If no section in the tree could plausibly contain the answer (the question
   is out of scope for this document), respond with TWO parts:
   (a) one sentence explaining what the document IS and why it does not
       contain the answer (e.g., "The provided document is the Berkshire
       Hathaway 2023 Annual Report, which does not contain information about
       [the question topic].");
   (b) close with the exact phrase: insufficient context.
   Bare "insufficient context" without the explanation is a partial answer —
   always include the one-sentence explanation first.

CRITICAL RULES (these prevent the most common failure modes):
- The Table of Contents (typically pages 1-3) lists section names but
  DOES NOT contain the answer text. Never cite pages 1-3 as the answer
  source unless the question is literally "what sections does this document
  have?" Descend past the TOC to the actual content sections.
- For factoid queries about specific numbers (revenues, earnings, dates),
  look at Form 10-K Item 8 / Consolidated Statements / Notes to Financial
  Statements — these contain canonical figures. The Chairman's Letter
  often paraphrases or summarizes; the Statements give the authoritative
  numbers.
- Cite the EXACT page range you fetched, not the parent section's range.
- Do not synthesize answers from training data. If the fetched text does
  not contain the answer, fetch a different range or refuse — do not
  fabricate.
"""


def _tree_view(tree: dict) -> str:
    """Compact JSON view: id, title, pages, summary. Skip raw text/children fields."""
    def walk(node: dict, depth: int = 0) -> list[dict]:
        out = [{
            "node_id": node.get("node_id"),
            "title": node.get("title"),
            "pages": f"{node.get('start_page', '?')}-{node.get('end_page', '?')}",
            "summary": (node.get("summary") or "")[:300],
            "depth": depth,
        }]
        for c in node.get("nodes", []):
            out.extend(walk(c, depth + 1))
        return out
    return json.dumps(walk(tree), indent=1)


_PDF_CACHE: dict[str, list[str]] = {}


def _pdf_pages(pdf_path: str) -> list[str]:
    if pdf_path not in _PDF_CACHE:
        reader = PdfReader(pdf_path)
        _PDF_CACHE[pdf_path] = [p.extract_text() or "" for p in reader.pages]
    return _PDF_CACHE[pdf_path]


def _get_page_content(pdf_path: str, start: int, end: int) -> str:
    pages = _pdf_pages(pdf_path)
    start = max(1, int(start))
    end = min(len(pages), int(end))
    if end < start:
        return f"[ERROR] Invalid range: end ({end}) < start ({start})"
    text = "\n\n".join(f"[page {i+1}]\n{pages[i]}" for i in range(start - 1, end))
    if len(text) > MAX_PAGE_RANGE_CHARS:
        text = text[:MAX_PAGE_RANGE_CHARS] + "\n[... truncated]"
    return text


def answer(query: str, tree_path: str = "data/tree.json",
           pdf_path: str = "data/brk-2023-ar.pdf") -> dict:
    tree = json.loads(Path(tree_path).read_text())
    tree_str = _tree_view(tree)

    tools = [{
        "type": "function",
        "function": {
            "name": "get_page_content",
            "description": "Fetch raw text from a page range of the source PDF.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_page": {"type": "integer", "description": "Start page (1-indexed)"},
                    "end_page": {"type": "integer", "description": "End page (inclusive, 1-indexed)"},
                },
                "required": ["start_page", "end_page"],
            },
        },
    }]

    msgs = [
        {"role": "system", "content": AGENTIC_SYSTEM},
        {"role": "user", "content": f"Document tree:\n{tree_str}\n\nQuestion: {query}"},
    ]

    tool_call_log: list[dict] = []
    final_answer = "insufficient context"
    iteration = 0

    for iteration in range(MAX_ITERATIONS):
        resp = omlx.chat.completions.create(
            model=MODEL, messages=msgs, tools=tools,
            temperature=0.0, max_tokens=800,
        )
        msg = resp.choices[0].message
        tcalls = getattr(msg, "tool_calls", None) or []

        if not tcalls:
            final_answer = (msg.content or "").strip()
            break

        msgs.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in tcalls
            ],
        })

        for tc in tcalls:
            try:
                args = json.loads(tc.function.arguments)
                if tc.function.name == "get_page_content":
                    sp = int(args.get("start_page", 1))
                    ep = int(args.get("end_page", sp))
                    content = _get_page_content(pdf_path, sp, ep)
                    tool_call_log.append({
                        "iter": iteration, "tool": "get_page_content",
                        "args": {"start": sp, "end": ep},
                        "content_chars": len(content),
                    })
                else:
                    content = f"[ERROR] Unknown tool: {tc.function.name}"
            except Exception as e:  # noqa: BLE001
                content = f"[ERROR] {type(e).__name__}: {e}"
            msgs.append({"role": "tool", "tool_call_id": tc.id, "content": content})

    # Debug log every call so compare_three failures can be diagnosed
    try:
        with open("/tmp/tree_debug.log", "a") as _f:
            _f.write(f"[{iteration+1}it/{len(tool_call_log)}tc] q={query[:60]!r} "
                     f"ans={final_answer[:80]!r}\n")
    except Exception:  # noqa: BLE001
        pass
    return {
        "answer": final_answer,
        "tool_calls": tool_call_log,
        "iterations": iteration + 1,
    }


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What was Berkshire's net earnings in 2023?"
    out = answer(q)
    print(json.dumps(out, indent=2, default=str))
