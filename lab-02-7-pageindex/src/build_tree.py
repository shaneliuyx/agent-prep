"""Build a hierarchical Table-of-Contents tree from a long PDF.

Three passes:
1. PDF parse + heading detection — heuristic over-recall on all-caps + numbered
   prefixes; produces a noisy candidate list.
2. LLM tree builder — one Gemma call consolidates the candidates into a clean
   {title, node_id, nodes: [...]} JSON tree, filtering page numbers, running
   headers, footer text. JSON-mode response_format enforces parse-safe output.
3. LLM per-node summaries — recurse over the tree, summarize each node's page
   range in 80-120 words. The navigation LLM at query time sees only summaries,
   never raw content, so summary specificity is load-bearing.

Output: data/tree.json. One JSON file is the entire index. No vector DB,
no graph database — versionable, diff-able, inspectable with jq.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

load_dotenv()
omlx = OpenAI(
    base_url=os.getenv("OMLX_BASE_URL"),
    api_key=os.getenv("OMLX_API_KEY"),
)
MODEL = os.getenv("MODEL_SONNET")

# ---------------------------------------------------------------- PDF parsing

def extract_pages(pdf_path: str) -> list[dict]:
    """Return list of {page_num, text} for every page (1-indexed)."""
    reader = PdfReader(pdf_path)
    return [
        {"page_num": i + 1, "text": p.extract_text() or ""}
        for i, p in enumerate(reader.pages)
    ]


def detect_heading_candidates(pages: list[dict]) -> list[dict]:
    """Heuristic-first heading detection — deliberately over-recall.

    Returns {page_num, line_text, candidate_level} for lines that look like
    headings:
      - All-caps lines (level 1) — SEC 10-K section banners ("RISK FACTORS")
      - Numbered prefixes 1., 1.1., 1.1.1. (level = depth of numbering)
      - Title Case short lines (level 2) — Berkshire-style sub-headings
        ("Acquisition Criteria", "Operating Earnings", "Owner's Manual")

    All three heuristics produce false positives (page numbers like "PAGE 5",
    list items like "1. Buy more milk", proper nouns like "Berkshire Hathaway"
    in body text). The LLM in build_tree() filters them out — optimizing this
    for precision burns engineering effort the LLM can absorb cheaply.
    """
    candidates: list[dict] = []
    for page in pages:
        for raw in page["text"].splitlines():
            line = raw.strip()
            if not line or len(line) > 80:
                continue
            # All-caps short lines = likely section header
            if line.isupper() and len(line) > 4:
                candidates.append({
                    "page_num": page["page_num"],
                    "line_text": line,
                    "candidate_level": 1,
                })
            # Numbered prefix "1.", "1.1.", "1.1.1." — depth = nesting level
            elif line[0].isdigit() and "." in line[:8]:
                depth = line.split()[0].count(".")
                candidates.append({
                    "page_num": page["page_num"],
                    "line_text": line,
                    "candidate_level": min(depth + 1, 4),
                })
            # Title Case short lines (3-8 words, mostly capitalized) — common
            # in financial annual reports for sub-section headings. Examples:
            # "Acquisition Criteria", "Owner's Manual", "Common Stock Data".
            # Filters: 3-8 words; >= 60% words start with uppercase; no
            # sentence-ending punctuation; not just a person's name.
            else:
                words = line.split()
                if 3 <= len(words) <= 8 and not line.endswith(("."  , "!", "?", ":")):
                    capitalized = sum(1 for w in words if w and w[0].isupper())
                    if capitalized / len(words) >= 0.6:
                        candidates.append({
                            "page_num": page["page_num"],
                            "line_text": line,
                            "candidate_level": 2,
                        })
    return candidates


# ---------------------------------------------------------- LLM tree builder

TREE_BUILDER_SYSTEM = """You receive a list of heading-candidate lines from a long
PDF document with their page numbers and detected hierarchy level. Your job is to
produce a clean hierarchical JSON tree with this schema:

{
  "title": "<document title>",
  "node_id": "0001",
  "nodes": [
    {"title": "<section title>", "node_id": "0002",
     "start_page": <int>, "end_page": <int>, "nodes": [...]}
  ]
}

Rules:
- Filter out spurious matches: page numbers (e.g. "PAGE 5"), running headers,
  table-cell labels, footer text, dates without context.
- Consolidate near-duplicate headings (same text appearing on multiple pages).
- Infer end_page from the start_page of the next sibling; the last node's
  end_page is the document's last page.
- Generate clean human-readable titles. If a heading is "1.1. ITEM 1A. RISK FACTORS",
  use "Item 1A — Risk Factors" as title — keep the source-heading words verbatim,
  apply case transformation only.
- Do NOT include leaf-level subsection content; only the structural skeleton.
- Assign node_id sequentially as 4-digit zero-padded strings ("0001", "0002", ...).

- **Coverage rule (load-bearing):** every page in the document must belong to some node's [start_page, end_page] range. If detected headings leave a gap (e.g., headings at pages 3, 21, 23 but nothing for pages 4-20), CREATE a placeholder node titled by what you infer the gap covers (e.g., "Chairman's Letter" for the typical 4-20 gap in an annual report; "Buffett's Letter to Shareholders" if document title mentions Berkshire). Better to have a generically-titled node covering pages 4-20 than to leave those pages unreachable. The navigator at query time can ONLY land on a node that exists.
- **Annual-report structural priors:** Berkshire / financial annual reports follow a standard skeleton — (1) Cover + Table of Contents (~pages 1-3); (2) Chairman's / CEO's Letter to Shareholders (~10-25 pages, often the first content section, contains "Acquisition Criteria", per-business commentary, capital allocation discussion); (3) Operating segment overviews (insurance, railroad, energy, etc); (4) GAAP Financial Statements (balance sheet, income statement, cash flows); (5) Notes to Financial Statements; (6) Management's Discussion & Analysis (or 10-K filing if embedded); (7) Independent Auditor's Report; (8) Corporate Governance / Officers / Directors; (9) Operating Companies appendix. Use this as a sanity check — if your tree has TOC + Financial Statements but NO Chairman's Letter, you missed the most important content section. Promote a placeholder.
- **Do not let one section dominate.** If a candidate set has many ALL-CAPS lines for "TABLE OF CONTENTS" or "REPORT OF AUDITOR" and few candidates for Chairman's Letter, do not collapse the entire document under TOC. TOC is a small leaf (typically 1-3 pages); make it ONE node, not the parent of everything.

Output strict JSON only, one tree object. No markdown, no commentary."""


def build_tree(headings: list[dict], doc_title: str, last_page: int) -> dict:
    """One LLM call consolidates heading candidates into a structural tree."""
    user_msg = (
        f"Document title: {doc_title}\n"
        f"Last page in document: {last_page}\n\n"
        f"Heading candidates ({len(headings)} total):\n"
        + json.dumps(headings, indent=1)
    )
    resp = omlx.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": TREE_BUILDER_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
        max_tokens=6000,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content or "{}"
    return json.loads(content)


# ---------------------------------------------------------- Per-node summaries

SUMMARIZE_SYSTEM = """Summarize this document section in 100-150 words. The
summary is read by a navigation LLM deciding whether this section answers a
user query — so it MUST contain concrete facts the navigator can match against.

REQUIRED elements (every summary must include):
1. Three numeric facts verbatim from the section (with units): e.g.,
   "$364.5 billion in revenues", "27.8% common-share ownership of Occidental",
   "operating earnings of $37,350 million".
2. Five named entities verbatim: companies, people, regulations, financial
   instruments, segment names — quoted exactly as the source uses them.
3. One sentence of structural location: where this section sits in the document
   hierarchy (e.g., "Sub-section of Chairman's Letter / Form 10-K Item 8").

PROHIBITED:
- Do NOT start with "This section discusses" or "The section covers" — write
  declarative sentences with the facts up front.
- Do NOT use generic phrases like "various financial metrics" or "the company's
  operations" — name the metrics, name the operations.
- Do NOT exceed 150 words.

If the section is genuinely empty boilerplate, output exactly:
"Empty boilerplate section — refer to subsections."
"""


def summarize_node(node: dict, pages: list[dict]) -> str:
    """Pull text spanning node['start_page']..node['end_page'] and summarize.

    Head-truncate at 12000 chars — a 10-K's longest section fits, longer
    sections get the head where the topic sentence usually lives.
    """
    start = node.get("start_page", 1)
    end = node.get("end_page", start)
    text = "\n".join(
        p["text"] for p in pages if start <= p["page_num"] <= end
    )
    if len(text) > 12000:
        text = text[:12000]
    if not text.strip():
        return "Empty section (no extractable text)."

    resp = omlx.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SUMMARIZE_SYSTEM},
            {"role": "user", "content": text},
        ],
        temperature=0.0,
        max_tokens=400,
    )
    return (resp.choices[0].message.content or "").strip()


def add_summaries_recursive(node: dict, pages: list[dict]) -> None:
    """In-place: write a `summary` field to every node that has a page range.
    Recurse into children. Idempotent — re-running re-summarizes."""
    if "start_page" in node and "end_page" in node:
        node["summary"] = summarize_node(node, pages)
    for child in node.get("nodes", []):
        add_summaries_recursive(child, pages)


# -------------------------- Recursive node split (PageIndex pattern, opt #2)

SPLIT_SYSTEM = """You receive raw text from a multi-page section of a long PDF.
Split this section into 2-5 topical sub-sections by content shifts. Return
strict JSON: {"sub_sections": [{"title": "...", "start_page": N, "end_page": N},
...]}.

Rules:
- Sub-section titles must come verbatim from the text (case-insensitive
  substring of an actual heading line in the source).
- Pages must lie within the section's page range and not overlap.
- If the section is too uniform to split meaningfully (single topic across all
  pages), return: {"sub_sections": []}."""

MAX_PAGES_PER_LEAF = 5  # PageIndex equivalent: max_page_num_each_node
MAX_CHARS_PER_LEAF = 20000  # PageIndex equivalent: max_token_num_each_node (~20K tok)


def split_large_nodes(node: dict, pages: list[dict], doc_root: bool = True) -> None:
    """Walk the tree; for any leaf spanning > MAX_PAGES_PER_LEAF AND with
    > MAX_CHARS_PER_LEAF of text, ask the LLM to split it into 2-5 sub-sections.

    PageIndex-equivalent of process_large_node_recursively. Skipped on the
    document root (always too big). Idempotent — re-running on an already-split
    tree won't double-split because non-leaves are skipped.
    """
    children = node.get("nodes", [])
    if not children and not doc_root:
        start = node.get("start_page", 1)
        end = node.get("end_page", start)
        span = end - start + 1
        text = "\n".join(p["text"] for p in pages if start <= p["page_num"] <= end)
        if span <= MAX_PAGES_PER_LEAF or len(text) <= MAX_CHARS_PER_LEAF:
            return
        if not text.strip():
            return
        text_for_llm = text[:18000]  # cap input
        try:
            resp = omlx.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SPLIT_SYSTEM},
                    {"role": "user", "content":
                        f"Section: {node.get('title', 'Untitled')} "
                        f"(pages {start}-{end})\n\nText:\n{text_for_llm}"},
                ],
                temperature=0.0, max_tokens=600,
                response_format={"type": "json_object"},
            )
            decision = json.loads(resp.choices[0].message.content or "{}")
            subs = decision.get("sub_sections", [])
        except Exception as e:  # noqa: BLE001
            print(f"      split_large_nodes: skip {node.get('title', '?')}: "
                  f"{type(e).__name__}")
            return
        if not subs:
            return
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
            print(f"      split {node.get('title', '?')[:50]}: "
                  f"{len(new_children)} sub-sections")

    for child in node.get("nodes", []):
        split_large_nodes(child, pages, doc_root=False)


# ---------------------------------------------------------- Main entry

def count_nodes(node: dict) -> int:
    """Recursively count nodes in the tree (including the root)."""
    return 1 + sum(count_nodes(c) for c in node.get("nodes", []))


def tree_depth(node: dict) -> int:
    """Maximum depth of the tree (root depth = 1)."""
    children = node.get("nodes", [])
    if not children:
        return 1
    return 1 + max(tree_depth(c) for c in children)


def main() -> None:
    # Berkshire Hathaway 2023 Annual Report — known-stable PDF URL
    # (https://www.berkshirehathaway.com/2023ar/2023ar.pdf). SEC EDGAR
    # serves only iXBRL HTML; company IR sites are the reliable PDF source
    # but URLs rotate. Berkshire's URL has been stable for 5+ years.
    pdf_path = "data/brk-2023-ar.pdf"
    out_path = Path("data/tree.json")

    if not Path(pdf_path).exists():
        raise FileNotFoundError(
            f"Missing {pdf_path}. Run the curl from §1.2 first."
        )

    print(f"[1/4] Parsing {pdf_path} ...")
    pages = extract_pages(pdf_path)
    print(f"      {len(pages)} pages extracted.")

    print("[2/4] Detecting heading candidates ...")
    headings = detect_heading_candidates(pages)
    print(f"      {len(headings)} heading candidates (over-recall expected).")

    print("[3/4] Building tree (LLM call, ~10-25 s) ...")
    tree = build_tree(headings, "Berkshire Hathaway 2023 Annual Report", last_page=len(pages))

    print(f"      Tree skeleton: {count_nodes(tree)} nodes, depth={tree_depth(tree)}.")

    print(f"[4/5] Splitting large leaves (> {MAX_PAGES_PER_LEAF} pages or "
          f"> {MAX_CHARS_PER_LEAF} chars) — PageIndex pattern ...")
    split_large_nodes(tree, pages, doc_root=True)
    print(f"      After split: {count_nodes(tree)} nodes, depth={tree_depth(tree)}.")

    print(f"[5/5] Generating per-node summaries ({count_nodes(tree)} LLM calls) ...")
    add_summaries_recursive(tree, pages)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(tree, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path} — {count_nodes(tree)} nodes, depth {tree_depth(tree)}.")


if __name__ == "__main__":
    main()