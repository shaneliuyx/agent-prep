# src/fetch_corpus.py
"""Fetch a domain-bounded slice of Wikipedia via MediaWiki category traversal.

Production architecture for GraphRAG corpus building:
1. Stakeholder defines the domain ("our knowledge base", "tech industry", etc.).
2. Engineer translates the domain into a *mechanism* that selects articles —
   a category-tree walk, a search query, a full-domain dump, or an embedding-
   similarity sampler. The mechanism scales (add more categories, walk deeper)
   without re-engineering.
3. Mechanism runs, produces a corpus, graph build runs over it, queries are
   served against whatever entities the graph happens to contain.

This script implements the mechanism via Wikipedia's category system. We pick
3-4 broad categories that bound the "tech industry" domain, fetch the article
titles in each (up to PER_CATEGORY_LIMIT), dedupe, cap at MAX_ARTICLES, then
pull the plain-text extract for each. The categories are a *scoping decision*;
the specific entities that fall out of the walk are emergent. The corpus does
not know in advance whether "Mark Zuckerberg" or "Apple Inc." will be in it,
which matches how production corpus pipelines actually behave.

Trade-offs vs the original `train[:200]` slice:
- More HTTP requests (~2-3 minutes instead of ~30 seconds), but the corpus
  actually covers the domain we want to query.
- Category coverage is whatever Wikipedia editors decided. A production system
  using a controlled corpus (company docs, internal wiki) would not have this
  ambiguity.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import requests

WIKI_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "lab-02-5-graphrag/1.0 (educational; agent-prep curriculum)"
SHUFFLE_SEED = 42  # deterministic shuffle so reproducible across runs
REQUEST_SLEEP = 0.6  # ~100 req/min — well under MediaWiki's 200/min anon limit
MAX_RETRIES = 4

# Domain-bounding parameter. Each entry is a Wikipedia category whose direct
# members (ns=0 articles) we will pull. Adding categories or switching to a
# different domain (medicine, law, sports) is the way to scale this — not
# adding hand-picked article titles. To inspect a category before adding it:
# https://en.wikipedia.org/wiki/Category:<NAME>
SEED_CATEGORIES: list[str] = [
    "American_technology_company_founders",
    "Companies_based_in_Silicon_Valley",
    "Software_companies_of_the_United_States",
    "American_chief_executives_of_technology_companies",
]

PER_CATEGORY_PAGE = 500  # max anon page size for categorymembers
MAX_PAGES_PER_CATEGORY = 5  # cap pagination — bound worst-case round-trips
MAX_ARTICLES = 150
ARTICLE_TEXT_CHARS = 4000


def _api_get(params: dict) -> dict:
    """Wrap requests.get with retry-on-429.

    MediaWiki applies a per-IP courtesy limit (~200/min anonymous). Sustained
    traffic at the limit eventually trips a 429 with a Retry-After header.
    We honor that header when present and otherwise back off exponentially
    starting at 2 s. After MAX_RETRIES we give up — the caller decides
    whether the missing article is fatal."""
    headers = {"User-Agent": USER_AGENT}
    backoff = 2.0
    for attempt in range(1, MAX_RETRIES + 1):
        resp = requests.get(WIKI_API, params=params, headers=headers, timeout=20)
        if resp.status_code == 429:
            wait = float(resp.headers.get("Retry-After", backoff))
            print(f"  rate-limited (attempt {attempt}/{MAX_RETRIES}); sleeping {wait:.0f}s")
            time.sleep(wait)
            backoff *= 2
            continue
        resp.raise_for_status()
        return resp.json()
    raise requests.HTTPError(f"429 after {MAX_RETRIES} retries")


def fetch_category_members(category: str) -> list[str]:
    """Page through every article member (ns=0) of one Wikipedia category.

    `categorymembers` returns at most `cmlimit` items per request (anonymous
    cap = 500). For categories larger than 500 members the API returns a
    `continue.cmcontinue` token; passing it back as `cmcontinue` resumes
    where we left off. Without pagination, the implicit alphabetical-by-
    sortkey ordering systematically drops Z-tail entries — that's how the
    first iteration of this lab missed `Mark Zuckerberg` despite him sitting
    in `American_technology_company_founders`.

    MAX_PAGES_PER_CATEGORY caps pagination at a fixed worst-case round-trip
    budget. Most tech-domain categories fit in 1-3 pages."""
    titles: list[str] = []
    cmcontinue: str | None = None
    for _ in range(MAX_PAGES_PER_CATEGORY):
        params = {
            "action":      "query",
            "format":      "json",
            "list":        "categorymembers",
            "cmtitle":     f"Category:{category}",
            "cmnamespace": 0,
            "cmtype":      "page",
            "cmlimit":     PER_CATEGORY_PAGE,
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        body = _api_get(params)
        members = body.get("query", {}).get("categorymembers", [])
        titles.extend(m["title"] for m in members)
        cmcontinue = body.get("continue", {}).get("cmcontinue")
        if not cmcontinue:
            break
        time.sleep(REQUEST_SLEEP)  # polite between pagination round-trips
    return titles


def fetch_extract(title: str, max_chars: int = ARTICLE_TEXT_CHARS) -> dict | None:
    """Pull the plain-text extract for one Wikipedia title.

    `prop=extracts&explaintext=True` returns UTF-8 prose, not wikitext.
    `redirects=1` follows redirects so that e.g. "iPhone" → canonical
    "IPhone" doesn't silent-fail on title-form mismatch."""
    params = {
        "action":      "query",
        "format":      "json",
        "prop":        "extracts",
        "titles":      title,
        "explaintext": True,
        "redirects":   1,
    }
    pages = _api_get(params).get("query", {}).get("pages", {})
    if not pages:
        return None
    page = next(iter(pages.values()))
    if page.get("missing") is True or "extract" not in page:
        return None
    return {
        "id":    str(page["pageid"]),
        "title": page["title"],
        "text":  page["extract"][:max_chars],
    }


def collect_titles() -> list[str]:
    """Walk the seed categories, dedupe titles, shuffle, cap at MAX_ARTICLES.

    Wikipedia categorymembers returns titles in collation order (alphabetical
    by sortkey). Without shuffling, MAX_ARTICLES truncation systematically
    drops Z-tail entries — including, e.g., 'Mark Zuckerberg' falling outside
    the first 50 of `American_technology_company_founders`. Shuffling with a
    fixed seed (SHUFFLE_SEED) gives a deterministic random sample: same input
    produces same output across runs while spanning the alphabet."""
    seen: set[str] = set()
    titles: list[str] = []
    for category in SEED_CATEGORIES:
        try:
            members = fetch_category_members(category)
        except requests.RequestException as exc:
            print(f"  category {category!r}: HTTP error {exc}")
            continue
        added = 0
        for t in members:
            if t in seen:
                continue
            seen.add(t)
            titles.append(t)
            added += 1
        print(f"  category {category!r}: {len(members)} members (paginated), {added} new")
        time.sleep(REQUEST_SLEEP)
    rng = random.Random(SHUFFLE_SEED)
    rng.shuffle(titles)
    if len(titles) > MAX_ARTICLES:
        titles = titles[:MAX_ARTICLES]
    return titles


def main() -> None:
    print(f"Resolving seed categories ({len(SEED_CATEGORIES)})...")
    titles = collect_titles()
    print(f"\n{len(titles)} unique titles to fetch.\n")

    out: list[dict] = []
    missing: list[str] = []
    t0 = time.time()
    for i, title in enumerate(titles, 1):
        try:
            article = fetch_extract(title)
        except requests.RequestException as exc:
            print(f"[{i}/{len(titles)}] {title!r}: HTTP error {exc}")
            missing.append(title)
            continue
        if article is None:
            print(f"[{i}/{len(titles)}] {title!r}: not found / empty")
            missing.append(title)
            continue
        out.append(article)
        if i % 10 == 0:
            print(f"  fetched {i}/{len(titles)} ({time.time() - t0:.0f}s)")
        # Polite to MediaWiki — REQUEST_SLEEP keeps us well under the 200/min
        # anonymous courtesy limit. Sustained traffic still occasionally trips
        # 429; _api_get handles that with Retry-After + exponential backoff.
        time.sleep(REQUEST_SLEEP)

    Path("data").mkdir(exist_ok=True)
    Path("data/corpus.json").write_text(json.dumps(out, indent=2))
    elapsed = time.time() - t0
    print(f"\nWrote {len(out)} articles in {elapsed:.0f}s")
    if missing:
        print(f"Missing ({len(missing)}): {missing}")


if __name__ == "__main__":
    main()
