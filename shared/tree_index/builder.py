"""Tree-builder helpers — recursive node split (PageIndex pattern).

The optimization win on W2.7 was applying recursive split AFTER initial
heuristic-tree build: any leaf spanning > MAX_PAGES OR > MAX_CHARS gets LLM-split
into 2-5 sub-sections. Prevents Chairman's-Letter-style monolithic leaves that
hide content from the navigator.
"""
from __future__ import annotations

import json
from typing import Iterable

DEFAULT_MAX_PAGES = 5
DEFAULT_MAX_CHARS = 20_000


def _node_text(node: dict, pages: Iterable[dict]) -> tuple[int, int, str]:
    """Return (start, end, concatenated text) for a node's page range."""
    start = node.get("start_page", 1)
    end = node.get("end_page", start)
    text = "\n".join(p["text"] for p in pages if start <= p["page_num"] <= end)
    return start, end, text


def split_large_nodes(
    node: dict,
    pages: list[dict],
    *,
    model_client,
    model_name: str,
    split_system_prompt: str,
    max_pages: int = DEFAULT_MAX_PAGES,
    max_chars: int = DEFAULT_MAX_CHARS,
    doc_root: bool = True,
    text_cap_chars: int = 18_000,
    max_tokens: int = 600,
) -> dict:
    """Walk the tree; for any leaf spanning > max_pages AND > max_chars, ask
    the LLM (via `split_system_prompt`) to split into 2-5 sub-sections. Mutates
    the tree in-place and returns it.

    Args:
        node:                root or sub-tree node dict (with nodes/start_page/
                             end_page/title/node_id fields).
        pages:               list of page dicts each with `page_num` + `text`.
        model_client:        OpenAI-compatible client.
        model_name:          target model.
        split_system_prompt: system prompt that asks for JSON sub_sections.
                             Recommended: tree_index.SPLIT_SYSTEM.
        max_pages, max_chars: split threshold. Defaults match W2.7 (5 pages,
                              20K chars ≈ 5K tokens).
        doc_root:            skip splitting the document root (always too big).
        text_cap_chars:      cap on input text passed to the LLM for splitting
                             (default 18K — leaves headroom in 32K context).
        max_tokens:          max output tokens for the LLM split call.

    Idempotent: re-running on an already-split tree skips non-leaves.
    """
    children = node.get("nodes", [])
    if not children and not doc_root:
        start, end, text = _node_text(node, pages)
        span = end - start + 1
        if span <= max_pages or len(text) <= max_chars or not text.strip():
            return node

        text_for_llm = text[:text_cap_chars]
        try:
            resp = model_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": split_system_prompt},
                    {"role": "user",   "content":
                        f"Section: {node.get('title', 'Untitled')} "
                        f"(pages {start}-{end})\n\nText:\n{text_for_llm}"},
                ],
                temperature=0.0, max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            decision = json.loads(resp.choices[0].message.content or "{}")
            subs = decision.get("sub_sections", [])
        except Exception:  # noqa: BLE001
            return node

        if not subs:
            return node

        parent_id = node.get("node_id", "0000")
        new_children = []
        for i, s in enumerate(subs, start=1):
            sp = max(start, int(s.get("start_page", start)))
            ep = min(end, int(s.get("end_page", sp)))
            if ep < sp:
                continue
            new_children.append({
                "node_id": f"{parent_id}.{i:02d}",
                "title": str(s.get("title", f"Sub-section {i}")).strip(),
                "start_page": sp,
                "end_page": ep,
                "nodes": [],
            })
        if new_children:
            node["nodes"] = new_children

    for child in node.get("nodes", []):
        split_large_nodes(
            child, pages,
            model_client=model_client,
            model_name=model_name,
            split_system_prompt=split_system_prompt,
            max_pages=max_pages,
            max_chars=max_chars,
            doc_root=False,
            text_cap_chars=text_cap_chars,
            max_tokens=max_tokens,
        )

    return node
