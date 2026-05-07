"""Agentic tree-index retriever — PageIndex pattern.

Replaces greedy single-shot tree-walk with multi-turn tool-calling loop:
the LLM sees the document tree (ids/titles/page-ranges/summaries), decides
which page range is most likely to contain the answer, calls
get_page_content(start, end), iterates if needed, and either composes the
final answer with a citation or refuses with explanation.

Closes the architectural blind spot of greedy navigation: navigator can now
inspect body text mid-decision instead of being limited to titles + summaries.
"""
from __future__ import annotations

import json
from typing import Protocol


class PageProvider(Protocol):
    """Returns raw text for a 1-indexed inclusive page range."""

    def __call__(self, start: int, end: int) -> str: ...


_DEFAULT_TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_page_content",
        "description": "Fetch raw text from a page range of the source document.",
        "parameters": {
            "type": "object",
            "properties": {
                "start_page": {"type": "integer", "description": "Start page (1-indexed)"},
                "end_page":   {"type": "integer", "description": "End page (inclusive, 1-indexed)"},
            },
            "required": ["start_page", "end_page"],
        },
    },
}]


def _tree_view(tree: dict) -> str:
    """Compact JSON view: id, title, pages, summary. Skip raw text/children fields."""
    def walk(node: dict, depth: int = 0) -> list[dict]:
        out = [{
            "node_id": node.get("node_id"),
            "title":   node.get("title"),
            "pages":   f"{node.get('start_page', '?')}-{node.get('end_page', '?')}",
            "summary": (node.get("summary") or "")[:300],
            "depth":   depth,
        }]
        for c in node.get("nodes", []):
            out.extend(walk(c, depth + 1))
        return out
    return json.dumps(walk(tree), indent=1)


class AgenticTreeRetriever:
    """Multi-turn agent loop over a tree + page-content tool.

    Args:
        tree:           dict with `node_id` / `title` / `start_page` / `end_page` /
                        `summary` / `nodes` fields. Compatible with W2.7
                        `data/tree.json` shape.
        page_provider:  callable returning raw text for a 1-indexed inclusive
                        page range. Typically wraps a `pypdf.PdfReader`.
        model_client:   OpenAI-compatible client (e.g., `openai.OpenAI`).
        model_name:     target model on the server.
        system_prompt:  agentic system prompt. Default is W2.7's hardened version
                        (TOC-trap guard + explained refusal + synthesis-from-
                        fragments). Pass a different prompt only when the
                        corpus has a structurally different shape.
        max_iterations: bounded loop ceiling (default 6).
        max_range_chars: per-fetch char cap on returned text (default 8000).
        debug_log_path: if set, append `[Nit/Mtc] q=... ans=...` per call for
                        cross-process debugging.

    Public method:
        answer(query) -> {answer, tool_calls, iterations}.
    """

    def __init__(
        self, *,
        tree: dict,
        page_provider: PageProvider,
        model_client,
        model_name: str,
        system_prompt: str,
        max_iterations: int = 6,
        max_range_chars: int = 8000,
        debug_log_path: str | None = None,
    ) -> None:
        self.tree = tree
        self.page_provider = page_provider
        self.client = model_client
        self.model = model_name
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.max_range_chars = max_range_chars
        self.debug_log_path = debug_log_path

    def _fetch(self, start: int, end: int) -> str:
        text = self.page_provider(start, end)
        if len(text) > self.max_range_chars:
            text = text[: self.max_range_chars] + "\n[... truncated]"
        return text

    def answer(self, query: str) -> dict:
        tree_str = _tree_view(self.tree)
        msgs: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": f"Document tree:\n{tree_str}\n\nQuestion: {query}"},
        ]
        tool_call_log: list[dict] = []
        final_answer = "insufficient context"
        iteration = 0

        for iteration in range(self.max_iterations):
            resp = self.client.chat.completions.create(
                model=self.model, messages=msgs, tools=_DEFAULT_TOOLS,
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
                     "function": {"name": tc.function.name,
                                  "arguments": tc.function.arguments}}
                    for tc in tcalls
                ],
            })

            for tc in tcalls:
                try:
                    args = json.loads(tc.function.arguments)
                    if tc.function.name == "get_page_content":
                        sp = int(args.get("start_page", 1))
                        ep = int(args.get("end_page", sp))
                        content = self._fetch(sp, ep)
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

        if self.debug_log_path:
            try:
                with open(self.debug_log_path, "a") as f:
                    f.write(f"[{iteration+1}it/{len(tool_call_log)}tc] "
                            f"q={query[:60]!r} ans={final_answer[:80]!r}\n")
            except Exception:  # noqa: BLE001
                pass

        return {
            "answer": final_answer,
            "tool_calls": tool_call_log,
            "iterations": iteration + 1,
        }
