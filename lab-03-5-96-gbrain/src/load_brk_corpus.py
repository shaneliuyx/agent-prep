"""W3.5.96 Path B — load the W2.7 Berkshire 10-K corpus into GBrain as a richer,
EXACT-TERM-HEAVY test corpus for the auto-eval harness.

WHY this corpus: the 19-page entity brain is semantic-heavy, so pure vector wins
and RRF adds nothing (Phase 6). A 10-K is the opposite shape — dense with dollar
figures, segment names, subsidiaries, "Scorecard", "operating earnings" — exactly
where the KEYWORD arm earns its weight and hybrid-RRF should win. Loading it lets
`auto_eval.ts` test that hypothesis directly.

The W2.7 sections (`brk_corpus.json`) are ALREADY page-shaped — `{id, title, text}`
— so this is a DETERMINISTIC load (slug = sections/<id>), no LLM extraction and no
graph reconcile (we're testing retrieval, not wiring). After loading we embed, then
the existing `run_auto_eval()` measures keyword vs vector vs hybrid over them.

Run: python src/load_brk_corpus.py     (needs GBRAIN_DATABASE_URL + OLLAMA_* up)
Env: BRK_CORPUS=<path>  SLUG_PREFIX=sections  AUTO_EVAL=1
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess

from ingest_agent import _GBRAIN, _server_env, run_auto_eval

_DEFAULT_CORPUS = pathlib.Path(os.path.expanduser(
    "~/code/agent-prep/lab-02-7-pageindex/data/brk_corpus.json"))
# PageIndex tree (built by lab-02-7 build_tree.py) carries the STRUCTURE the flat
# brk_corpus.json drops: per-section page ranges + hierarchy. Joining it in is the
# "PageIndex + GBrain" combined ingest — it makes structural/"where-is-X" questions
# answerable from the page itself, with no per-query tree navigation.
_DEFAULT_TREE = pathlib.Path(os.path.expanduser(
    "~/code/agent-prep/lab-02-7-pageindex/data/tree.json"))
_SLUG_PREFIX = os.getenv("SLUG_PREFIX", "sections")
_ITEM_RE = re.compile(r"\bItem\s+\d+[A-Z]?\b")


def _clean_title(raw: str, fallback: str) -> str:
    """The W2.7 titles are breadcrumbs ("Berkshire ... Annual Report > Chairman's
    Letter"). The shared prefix repeats across all 44 sections, so using the whole
    breadcrumb makes keyword drown in boilerplate. Keep only the distinctive TAIL
    after the last '>' ("Chairman's Letter") as the page title / exact probe."""
    tail = raw.split(">")[-1].strip() if raw else ""
    return tail or fallback


def flatten_tree(node: dict, parent_title: str = "", inherited_item: str = "",
                 out: dict | None = None) -> dict:
    """node_id → {title, start_page, end_page, parent, item}. `item` is inherited from the
    nearest ancestor whose title matches `Item N` (so a sub-section of Item 8 still knows it's
    in Item 8). Used to enrich each GBrain page with its document location."""
    if out is None:
        out = {}
    title = str(node.get("title", ""))
    m = _ITEM_RE.search(title)
    item = m.group(0) if m else inherited_item
    nid = str(node.get("node_id", "")).strip()
    if nid:
        out[nid] = {"title": title, "start_page": node.get("start_page"),
                    "end_page": node.get("end_page"), "parent": parent_title, "item": item}
    for child in node.get("nodes", []):
        flatten_tree(child, title, item, out)
    return out


def build_pages(corpus: list[dict], prefix: str = _SLUG_PREFIX,
                tree_meta: dict | None = None, source: str = "") -> list[tuple[str, str]]:
    """Pure transform: W2.7 sections → (slug, GBrain-page-content) pairs.

    Each section becomes a markdown page with YAML frontmatter under `<prefix>/<id>`.
    The frontmatter `title:` is AUTHORITATIVE — `gbrain put` titles from frontmatter,
    NOT from the `# heading` (that seam is why a slug-only write gets titled "Brk 0002").

    PageIndex-enriched (when `tree_meta` is supplied): frontmatter gains `source` / `item` /
    `pages` / `parent`, AND a `**Location:**` breadcrumb is prepended to the BODY. The body
    line is the load-bearing bit — GBrain indexes title+body+timeline, so a location *in the
    body* is retrievable (keyword + vector) and injected into the reader's context, whereas
    unknown frontmatter keys are not. That is what makes "where are the Notes located?"
    answerable. Sections with no id or empty text are skipped.
    """
    pages: list[tuple[str, str]] = []
    tree_meta = tree_meta or {}
    for sec in corpus:
        sid = str(sec.get("id", "")).strip()
        text = str(sec.get("text", "")).strip()
        if not sid or not text:
            continue
        title = _clean_title(str(sec.get("title", "")), sid)
        esc = title.replace('"', '\\"')
        slug = f"{prefix}/{sid}"
        # Join by TITLE, not node_id: the on-disk brk_corpus.json and tree.json can carry
        # drifted node_ids (regenerated independently), but section titles are stable.
        meta = tree_meta.get(title, {})

        fm = [f'title: "{esc}"']
        loc = ""
        if source:
            fm.append(f'source: "{source}"')
        if meta.get("item"):
            fm.append(f'item: "{meta["item"]}"')
        if meta.get("parent"):
            fm.append(f'parent: "{_clean_title(meta["parent"], "").replace(chr(34), "")}"')
        if meta.get("start_page") and meta.get("end_page"):
            pages_str = f'{meta["start_page"]}-{meta["end_page"]}'
            fm.append(f'pages: "{pages_str}"')
            where = meta.get("item") or title
            loc = (f"**Location:** {where} — {title} · pages {pages_str}"
                   + (f" · source: {source}" if source else "") + "\n\n")

        content = f'---\n' + "\n".join(fm) + f'\n---\n\n# {title}\n\n{loc}{text}\n'
        pages.append((slug, content))
    return pages


def _put_page(slug: str, content: str) -> bool:
    """Write one page via the local `gbrain put` CLI (stdin = content)."""
    out = subprocess.run([_GBRAIN, "put", slug], input=content,
                         capture_output=True, text=True, env=_server_env())
    if out.returncode != 0:
        print(f">>> WARNING: put failed for {slug}: {(out.stderr or out.stdout).strip()[:200]}")
    return out.returncode == 0


def _embed_stale() -> str:
    """Embed newly-written chunks so the vector arm has vectors to search."""
    out = subprocess.run([_GBRAIN, "embed", "--stale"],
                         capture_output=True, text=True, env=_server_env())
    lines = [ln for ln in (out.stdout or out.stderr).splitlines() if ln.strip()]
    return lines[-1] if lines else "(no output)"


def main() -> None:
    corpus_path = pathlib.Path(os.getenv("BRK_CORPUS", str(_DEFAULT_CORPUS)))
    if not corpus_path.exists():
        raise SystemExit(f"corpus not found: {corpus_path} (set BRK_CORPUS=<path>)")

    corpus = json.loads(corpus_path.read_text())

    # PageIndex join: pull per-section page ranges + hierarchy from tree.json (if present).
    tree_path = pathlib.Path(os.getenv("BRK_TREE", str(_DEFAULT_TREE)))
    tree_meta, source = {}, ""
    if tree_path.exists():
        tree = json.loads(tree_path.read_text())
        # key by title (see build_pages: id join is unreliable across regenerated files)
        tree_meta = {m["title"].strip(): m for m in flatten_tree(tree).values() if m.get("title")}
        source = str(tree.get("title", "")).strip()
        print(f">>> PageIndex enrich: {tree_path.name} → {len(tree_meta)} nodes "
              f"(source '{source}')")
    else:
        print(f">>> no tree.json at {tree_path} — ingesting WITHOUT structural metadata")

    pages = build_pages(corpus, tree_meta=tree_meta, source=source)
    print(f">>> {corpus_path.name}: {len(corpus)} sections → {len(pages)} pages "
          f"(prefix '{_SLUG_PREFIX}')")

    written = sum(_put_page(slug, content) for slug, content in pages)
    print(f">>> wrote {written}/{len(pages)} pages")

    print(">>> embed --stale: " + _embed_stale())
    print(">>> auto-eval: " + run_auto_eval())


if __name__ == "__main__":
    main()
