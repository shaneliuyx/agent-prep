"""Berkshire PDF → article-shape corpus for downstream ingest.

Output format mirrors lab-02-5's data/corpus.json shape so the W2.5 build_graph
pipeline (entity extraction + Neo4j) accepts it without modification:

  [
    {"id": "brk_<node_id>", "title": "<section title>", "text": "<full text of section pages>"},
    ...
  ]

Section boundaries come from data/tree.json (built by build_tree.py). Each
top-level + nested node with a page range becomes one "article" in the corpus.
This gives the entity extractor + vector chunker reasonable section-sized
inputs (~3-15 pages each) instead of the whole 152-page PDF as one chunk.

Usage:
    python src/_brk_corpus.py
    # writes data/brk_corpus.json
"""
from __future__ import annotations

import json
from pathlib import Path

from pypdf import PdfReader


def collect_section_articles(node: dict, pages_text: list[str], parent_title: str = "") -> list[dict]:
    """Recurse the tree; emit one article per node that has a page range.

    `pages_text[i]` is the full text of page i+1 (1-indexed in tree, 0-indexed
    here). Title is concatenated with parent for context.
    """
    articles: list[dict] = []
    if "start_page" in node and "end_page" in node:
        title = f"{parent_title} > {node['title']}" if parent_title else node["title"]
        start, end = node["start_page"], node["end_page"]
        text = "\n".join(
            pages_text[i] for i in range(start - 1, min(end, len(pages_text)))
            if i < len(pages_text)
        )
        if text.strip():
            articles.append({
                "id": f"brk_{node['node_id']}",
                "title": title,
                "text": text,
            })
    # Recurse into children with the current node's title as parent context
    new_parent = f"{parent_title} > {node['title']}" if parent_title else node["title"]
    for child in node.get("nodes", []):
        articles.extend(collect_section_articles(child, pages_text, new_parent))
    return articles


def main() -> None:
    pdf_path = Path("data/brk-2023-ar.pdf")
    tree_path = Path("data/tree.json")
    out_path = Path("data/brk_corpus.json")

    if not pdf_path.exists():
        raise FileNotFoundError(f"Missing {pdf_path}. Run §1.2 curl first.")
    if not tree_path.exists():
        raise FileNotFoundError(f"Missing {tree_path}. Run build_tree.py first.")

    print(f"[1/3] Loading {tree_path} ...")
    tree = json.loads(tree_path.read_text())

    print(f"[2/3] Extracting page text from {pdf_path} ...")
    reader = PdfReader(str(pdf_path))
    pages_text = [p.extract_text() or "" for p in reader.pages]
    print(f"      {len(pages_text)} pages.")

    print("[3/3] Building per-section corpus ...")
    articles = collect_section_articles(tree, pages_text)
    out_path.write_text(json.dumps(articles, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path} — {len(articles)} sections, "
          f"avg {sum(len(a['text']) for a in articles) // max(len(articles), 1)} chars/section.")


if __name__ == "__main__":
    main()
