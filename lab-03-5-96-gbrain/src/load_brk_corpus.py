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
import subprocess

from ingest_agent import _GBRAIN, _server_env, run_auto_eval

_DEFAULT_CORPUS = pathlib.Path(os.path.expanduser(
    "~/code/agent-prep/lab-02-7-pageindex/data/brk_corpus.json"))
_SLUG_PREFIX = os.getenv("SLUG_PREFIX", "sections")


def _clean_title(raw: str, fallback: str) -> str:
    """The W2.7 titles are breadcrumbs ("Berkshire ... Annual Report > Chairman's
    Letter"). The shared prefix repeats across all 44 sections, so using the whole
    breadcrumb makes keyword drown in boilerplate. Keep only the distinctive TAIL
    after the last '>' ("Chairman's Letter") as the page title / exact probe."""
    tail = raw.split(">")[-1].strip() if raw else ""
    return tail or fallback


def build_pages(corpus: list[dict], prefix: str = _SLUG_PREFIX) -> list[tuple[str, str]]:
    """Pure transform: W2.7 sections → (slug, GBrain-page-content) pairs.

    Each section becomes a markdown page with YAML frontmatter under `<prefix>/<id>`.
    The frontmatter `title:` is AUTHORITATIVE — `gbrain put` titles from frontmatter,
    NOT from the `# heading` (that seam is why a slug-only write gets titled
    "Brk 0002"). No wikilinks — this corpus is for retrieval eval, not the graph.
    Sections with no id or empty text are skipped.
    """
    pages: list[tuple[str, str]] = []
    for sec in corpus:
        sid = str(sec.get("id", "")).strip()
        text = str(sec.get("text", "")).strip()
        if not sid or not text:
            continue
        title = _clean_title(str(sec.get("title", "")), sid)
        esc = title.replace('"', '\\"')
        slug = f"{prefix}/{sid}"
        content = f'---\ntitle: "{esc}"\n---\n\n# {title}\n\n{text}\n'
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
    pages = build_pages(corpus)
    print(f">>> {corpus_path.name}: {len(corpus)} sections → {len(pages)} pages "
          f"(prefix '{_SLUG_PREFIX}')")

    written = sum(_put_page(slug, content) for slug, content in pages)
    print(f">>> wrote {written}/{len(pages)} pages")

    print(">>> embed --stale: " + _embed_stale())
    print(">>> auto-eval: " + run_auto_eval())


if __name__ == "__main__":
    main()
