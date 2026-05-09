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
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

# Bootstrap shared/tree_index onto sys.path
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "shared"))

from tree_index import (  # noqa: E402
    FACT_RICH_SUMMARIZE_SYSTEM,
    SPLIT_SYSTEM,
    split_large_nodes as _shared_split_large_nodes,
)

load_dotenv()
omlx = OpenAI(
    base_url=os.getenv("OMLX_BASE_URL"),
    api_key=os.getenv("OMLX_API_KEY"),
)
# MODEL_BUILD overrides MODEL_SONNET specifically for tree-build/summarize.
# Decoupling lets you build with a stronger model (e.g., Qwen3.6-35B-A3B-DWQ
# for richer fact extraction) while keeping the judge model independent.
MODEL = os.getenv("MODEL_BUILD") or os.getenv("MODEL_SONNET")

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
    content = _llm_call_with_retry(
        messages=[
            {"role": "system", "content": TREE_BUILDER_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=6000,
        response_format={"type": "json_object"},
    ) or "{}"
    return json.loads(content)


# ---------------------------------------------------------- Per-node summaries

# Reuse the shared fact-rich summarize prompt (lifted from this lab's W2.7
# optimization run; lives in shared/tree_index/prompts.py for cross-lab reuse).
SUMMARIZE_SYSTEM = FACT_RICH_SUMMARIZE_SYSTEM


_EXTRACT_SYSTEM = """First-pass extractor for tree-index summarization.
Read the section text and emit a strict JSON object with:
  {
    "title_phrase": "<the section's heading phrase verbatim, if visible
                      in the first 200 chars; else empty string>",
    "entities": ["<10-30 named entities verbatim — companies, people,
                   regulations, financial instruments, segment names,
                   product names>"],
    "aliases": ["<short forms / nicknames / acronyms paired with the
                  full names above; e.g. 'AMEX' for 'American Express'>"],
    "quoted_phrases": ["<distinctive multi-word phrases worth preserving:
                         'patience pays', 'wonderful businesses', etc.
                         Include scare-quoted phrases and metaphors>"],
    "numeric_facts": ["<3-5 numeric facts with units, verbatim from source:
                        '$364,482 million in revenues', '27.8% ownership'>"],
    "section_id": "<numbered identifier if present: 'Item 1A', 'Item 8',
                   'Schedule X', or empty string>"
  }

Output ONLY this JSON, no prose. Be exhaustive — downstream summarizer
needs all distinctive tokens preserved verbatim."""


def _llm_call_with_retry(messages: list, max_tokens: int,
                          response_format: dict | None = None) -> str:
    """LLM call with backoff retry on transient errors + post-call cooldown.
    vMLX returns 503 'Metal GPU working set too full' under accumulated load.
    Strategy:
      - Pre-call: progressive backoff retry on transient errors (up to 8
        attempts: 15, 30, 60, 90, 120, 180, 240, 300 s).
      - Post-call: brief cooldown sleep (2s) so GPU can release before next
        call. Trades ~5 min total build time for OOM elimination.
    """
    import time as _time
    last_err = None
    backoff_schedule = [15, 30, 60, 90, 120, 180, 240, 300]
    for attempt, sleep_s in enumerate(backoff_schedule):
        try:
            kwargs = dict(model=MODEL, messages=messages,
                          temperature=0.0, max_tokens=max_tokens)
            if response_format:
                kwargs["response_format"] = response_format
            r = omlx.chat.completions.create(**kwargs)
            content = (r.choices[0].message.content or "").strip()
            _time.sleep(2)  # cooldown: let GPU release before next call
            return content
        except Exception as e:                                # noqa: BLE001
            last_err = e
            msg = str(e)
            transient = ("503" in msg or "GPU working set" in msg
                         or "ConnectionError" in type(e).__name__
                         or "Connection" in msg)
            if not transient:
                raise
            print(f"  [build] transient {type(e).__name__} on attempt "
                  f"{attempt+1}/{len(backoff_schedule)}; sleep {sleep_s}s "
                  f"before retry. detail={msg[:120]}", flush=True)
            _time.sleep(sleep_s)
    raise RuntimeError(f"_llm_call_with_retry exhausted {len(backoff_schedule)} "
                       f"attempts; last_err={last_err}")


def _extract_facts(text: str) -> dict:
    """Pass 1 of multi-pass summarization: pull entities, quoted phrases,
    numeric facts, and the section's title phrase. JSON-mode response.
    max_tokens=400 caps GPU command-buffer size (smaller = less memory
    pressure on Apple Silicon)."""
    fallback = {"title_phrase": "", "entities": [], "aliases": [],
                "quoted_phrases": [], "numeric_facts": [], "section_id": ""}
    try:
        raw = _llm_call_with_retry(
            messages=[
                {"role": "system", "content": _EXTRACT_SYSTEM},
                {"role": "user", "content": text},
            ],
            max_tokens=400,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(raw or "{}")
        # Some models (Gemma in particular) emit a top-level list when asked
        # for JSON-object output. Guard against that here — fall back to
        # empty-fields dict so caller's `.get()` calls don't crash.
        if not isinstance(parsed, dict):
            return fallback
        # Coerce any field that should be a list but came back as something
        # else (string, dict) into a list — defensive against schema drift.
        for k in ("entities", "aliases", "quoted_phrases", "numeric_facts"):
            v = parsed.get(k)
            if v is None:
                parsed[k] = []
            elif isinstance(v, str):
                parsed[k] = [v]
            elif not isinstance(v, list):
                parsed[k] = []
        for k in ("title_phrase", "section_id"):
            v = parsed.get(k)
            if not isinstance(v, str):
                parsed[k] = ""
        return parsed
    except Exception:                                   # noqa: BLE001
        return fallback


def _compose_summary(text: str, facts: dict) -> str:
    """Pass 2: write the summary preserving Pass-1 vocabulary verbatim.
    max_tokens=400 keeps GPU command buffers small."""
    facts_block = json.dumps(facts, indent=2)
    user_msg = (
        f"Extracted facts (you MUST preserve every entity, alias, quoted "
        f"phrase, and numeric fact from this JSON in your SUMMARY block "
        f"verbatim):\n{facts_block}\n\n"
        f"Section text:\n{text}"
    )
    return _llm_call_with_retry(
        messages=[
            {"role": "system", "content": SUMMARIZE_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=400,
    )


def _parse_tags_block(summary_text: str) -> list[str]:
    """Extract the TAGS: line tokens from a multi-block summary output.
    Returns a list of comma-separated tokens, deduped, preserving order."""
    import re as _re
    m = _re.search(r"^\s*TAGS\s*:\s*(.+?)$", summary_text,
                   _re.MULTILINE | _re.DOTALL)
    if not m:
        return []
    raw = m.group(1).split("\n", 1)[0]  # only the first TAGS line
    seen: set = set()
    out: list[str] = []
    for tok in raw.split(","):
        t = tok.strip()
        if t and t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out


def summarize_node(node: dict, pages: list[dict]) -> tuple[str, list[str]]:
    """Two-pass summarization. Returns (summary_text, tags_list).

    Pass 1: extract entities, quoted phrases, numeric facts.
    Pass 2: compose summary that preserves Pass-1 vocabulary verbatim.
    The TAGS: line is parsed out for direct EntityIndex lookup.

    Head-truncate at 12000 chars — a 10-K's longest section fits, longer
    sections get the head where the topic sentence usually lives.
    """
    start = node.get("start_page", 1)
    end = node.get("end_page", start)
    text = "\n".join(
        p["text"] for p in pages if start <= p["page_num"] <= end
    )
    if len(text) > 8000:
        text = text[:8000]   # reduced from 12000 — smaller K/V cache, less GPU pressure
    if not text.strip():
        return ("Empty section (no extractable text).", [])

    facts = _extract_facts(text)
    summary = _compose_summary(text, facts)
    tags = _parse_tags_block(summary)
    # Augment tags with Pass-1 extraction (ensures verbatim preservation
    # even if compose pass dropped one). Dedupe case-insensitively.
    extra = (facts.get("entities", []) + facts.get("aliases", [])
             + facts.get("quoted_phrases", []))
    if facts.get("title_phrase"):
        extra = [facts["title_phrase"]] + extra
    if facts.get("section_id"):
        extra.append(facts["section_id"])
    seen = {t.lower() for t in tags}
    for t in extra:
        ts = (t or "").strip()
        if ts and ts.lower() not in seen:
            seen.add(ts.lower())
            tags.append(ts)
    return (summary, tags)


def add_summaries_recursive(node: dict, pages: list[dict]) -> None:
    """In-place: write `summary` and `tags` fields to every node that has a
    page range. Recurse into children. Idempotent — re-running rewrites both."""
    if "start_page" in node and "end_page" in node:
        summary, tags = summarize_node(node, pages)
        node["summary"] = summary
        node["tags"] = tags
    for child in node.get("nodes", []):
        add_summaries_recursive(child, pages)


# -------------------------- Recursive node split — uses shared/tree_index

MAX_PAGES_PER_LEAF = 5  # PageIndex equivalent: max_page_num_each_node
MAX_CHARS_PER_LEAF = 20000  # PageIndex equivalent: max_token_num_each_node (~20K tok)


def split_large_nodes(node: dict, pages: list[dict], doc_root: bool = True) -> None:
    """Thin wrapper around shared/tree_index.split_large_nodes — supplies this
    lab's model client + Berkshire-tuned thresholds. The actual split logic
    lives in shared/tree_index/builder.py for cross-lab reuse.
    """
    _shared_split_large_nodes(
        node, pages,
        model_client=omlx,
        model_name=MODEL or "",
        split_system_prompt=SPLIT_SYSTEM,
        max_pages=MAX_PAGES_PER_LEAF,
        max_chars=MAX_CHARS_PER_LEAF,
        doc_root=doc_root,
    )


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

    # Raise recursion limit upfront — the recursive PageIndex split has a known
    # bug producing depth-46 chains, and even depth-of-skeleton recursion through
    # nested generators (count_nodes, tree_depth) hits Python's default 1000
    # frame ceiling. 5000 is comfortable for any reasonable W2.7 tree.
    sys.setrecursionlimit(5000)

    print(f"      Tree skeleton: {count_nodes(tree)} nodes, depth={tree_depth(tree)}.")

    # SPLIT DISABLED 2026-05-09 — recursive split in shared/tree_index/builder.py
    # creates degenerate depth-46 chains that crash post-process. The 46-node
    # skeleton from Phase 3 is sufficient for multi-pass summarization
    # (matches W2.7's original 50-node published baseline).
    # print(f"[4/5] Splitting large leaves (> {MAX_PAGES_PER_LEAF} pages or "
    #       f"> {MAX_CHARS_PER_LEAF} chars) — PageIndex pattern ...")
    # split_large_nodes(tree, pages, doc_root=True)
    # print(f"      After split: {count_nodes(tree)} nodes, depth={tree_depth(tree)}.")

    print(f"[5/5] Generating per-node summaries ({count_nodes(tree)} LLM calls) ...")
    add_summaries_recursive(tree, pages)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(tree, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path} — {count_nodes(tree)} nodes, depth {tree_depth(tree)}.")


if __name__ == "__main__":
    main()