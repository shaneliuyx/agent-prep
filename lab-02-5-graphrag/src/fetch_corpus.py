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
import math
import random
import time
from pathlib import Path

import requests

WIKI_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "lab-02-5-graphrag/1.0 (educational; agent-prep curriculum)"
SHUFFLE_SEED = 42  # deterministic shuffle so reproducible across runs
REQUEST_SLEEP = 0.6  # ~100 req/min — well under MediaWiki's 200/min anon limit
MAX_RETRIES = 4
PAGEVIEWS_BATCH = 50  # MediaWiki query API caps at 50 titles per `titles=` arg
PAGEVIEWS_DAYS = 60  # max value pvipdays accepts

# Domain-bounding parameter. Each entry is a Wikipedia category whose direct
# members (ns=0 articles) we will pull. Adding categories or switching to a
# different domain (medicine, law, sports) is the way to scale this — not
# adding hand-picked article titles. To inspect a category before adding it:
# https://en.wikipedia.org/wiki/Category:<NAME>
SEED_CATEGORIES: list[str] = [
    # Founders — Wikipedia's taxonomy splits "tech founders" into multiple
    # subcategories. Mark Zuckerberg, Larry Page, Sergey Brin, Jeff Bezos,
    # Jack Dorsey land in "Internet company founders" (consumer Internet),
    # while semiconductor/hardware founders land in "technology company
    # founders" — without both, half the canonical names are uncovered.
    "American_technology_company_founders",
    "American_Internet_company_founders",
    # Executives — covers Tim Cook, Sundar Pichai, Satya Nadella, Andy Jassy,
    # who are not founders but are central nodes in the tech graph.
    "American_chief_executives_in_technology",
    # Wealth-class backstop — picks up tech billionaires that fall outside
    # the founder/CEO subcategories (e.g. early-employee equity, investors).
    "American_billionaires",
    # Companies — Silicon Valley + US software for organizational coverage.
    "Companies_based_in_Silicon_Valley",
    "Software_companies_of_the_United_States",
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


def fetch_pageviews_batch(titles: list[str]) -> dict[str, int]:
    """Pull last-60-day pageview totals for up to ~50 titles per batch.

    MediaWiki's `prop=pageviews&pvipdays=60` returns daily counts per article.
    We sum the daily counts to get a single importance score per article.

    Two non-obvious gotchas:

    1. Title-resolution mismatch (entry 12 in W2.5 bad-case journal). With
       `redirects=1` the API normalizes input titles ("Apple Inc." →
       "Apple Inc") and follows redirects ("iPhone" → "IPhone"). Response
       titles in `query.pages[*].title` are *canonical*, not input. Returning
       a dict keyed by *input* title requires walking the `normalized` +
       `redirects` arrays to map back.

    2. Per-property continue limit. `prop=pageviews` returns pageview data
       for only ~19 titles per call regardless of how many titles are in
       the request. The remaining titles appear in `query.pages` *without*
       a `pageviews` field — silently zero. The response includes a
       `continue.pvipcontinue` token to fetch the next slice of pageviews
       data; loop until that token disappears.

    Without (1), ~13% of inputs return correct pageviews; without (2),
    ~38% do. With both, ~98%+ return correct pageviews (rare misses are
    pages where MediaWiki's pageview backend is genuinely missing data
    — disambiguation pages, recently-renamed articles, etc.)."""
    out: dict[str, int] = {}

    # Build input → canonical mapping once from the first response.
    input_to_canonical: dict[str, str] = {t: t for t in titles}
    canonical_to_total: dict[str, int] = {}

    base_params = {
        "action":      "query",
        "format":      "json",
        "prop":        "pageviews",
        "titles":      "|".join(titles),
        "pvipdays":    PAGEVIEWS_DAYS,
        "redirects":   1,
    }
    pvipcontinue: str | None = None
    seen_canonical: set[str] = set()
    for round_idx in range(20):  # bounded to avoid infinite loop on API bug
        params = dict(base_params)
        if pvipcontinue:
            params["pvipcontinue"] = pvipcontinue
        body = _api_get(params)
        query = body.get("query", {})

        # First round only: capture title normalization/redirect mappings.
        if round_idx == 0:
            for n in query.get("normalized", []):
                from_t, to_t = n["from"], n["to"]
                for k, v in input_to_canonical.items():
                    if v == from_t:
                        input_to_canonical[k] = to_t
            for r in query.get("redirects", []):
                from_t, to_t = r["from"], r["to"]
                for k, v in input_to_canonical.items():
                    if v == from_t:
                        input_to_canonical[k] = to_t

        # Each round contributes pageviews for a slice of titles.
        for page in query.get("pages", {}).values():
            title = page.get("title", "")
            pv = page.get("pageviews")
            if pv is None:
                continue  # this title is in a later slice
            seen_canonical.add(title)
            canonical_to_total[title] = sum(
                v for v in pv.values() if isinstance(v, int)
            )

        cont = body.get("continue", {})
        new_pvipcontinue = cont.get("pvipcontinue")
        if not new_pvipcontinue or new_pvipcontinue == pvipcontinue:
            break
        pvipcontinue = new_pvipcontinue

    # Resolve each input title to canonical, look up its pageview total.
    for input_title in titles:
        canonical = input_to_canonical.get(input_title, input_title)
        out[input_title] = canonical_to_total.get(canonical, 0)
    return out


def weighted_sample_no_replacement(
    items: list[str],
    weights: dict[str, int],
    k: int,
    seed: int,
) -> list[str]:
    """A-ExpJ weighted reservoir sampling (Efraimidis & Spirakis, 2006).

    Without numpy: assign each item key_i = log(U_i) / weight_i where U_i ~
    Uniform(0,1), sort ascending, take top-k. Items with weight=0 are excluded
    (would divide by zero); items with very small weights are sampled
    proportionally. Deterministic with seeded RNG so the same input pool
    produces the same sample across runs.

    For weight distributions with a single zero or near-zero floor, we add 1
    to every weight to avoid the sampler ignoring any candidate entirely. The
    +1 floor still respects the heavy-tail (10M views + 1 ≈ 10M views) while
    keeping zero-pageview articles eligible at their long-tail probability."""
    rng = random.Random(seed)
    keyed: list[tuple[float, str]] = []
    for item in items:
        weight = weights.get(item, 0) + 1  # +1 floor — see docstring
        u = rng.random()
        # log(u) is negative; dividing by positive weight keeps it negative;
        # smaller (more negative) keys = lower-priority items. Sorting
        # descending means highest-weight items rise to the top.
        key = math.log(u) / weight
        keyed.append((key, item))
    keyed.sort(reverse=True)
    return [item for _, item in keyed[:k]]


def collect_titles() -> list[str]:
    """Walk the seed categories → dedupe → pageview-weighted sample → cap at k.

    Production-shape sampling: Wikipedia's category structure is uniform-
    enumerable but the entities inside are heavy-tailed by attention.
    Mid-tier B2B companies dominate the membership lists; canonical
    entities like 'Mark Zuckerberg' or 'Steve Jobs' sit alongside them
    with no built-in priority signal. Sampling uniformly makes canonical
    entities probability ~1/N where N is the deduped pool size — for our
    1472-candidate domain that's ~10% inclusion per famous name.

    Pageview-weighted A-ExpJ sampling instead biases inclusion by the
    same importance signal real corpus pipelines use. Mark Zuckerberg's
    Wikipedia page receives ~5M+ monthly pageviews; a typical mid-tier
    SaaS company gets ~5K. The 1000× ratio in the weighted sampler
    means famous entities are essentially guaranteed to be in the
    sample while leaving capacity for long-tail discovery."""
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

    # Batched pageview fetch — up to 50 titles per request.
    print(f"\nFetching pageviews for {len(titles)} candidates ...")
    weights: dict[str, int] = {}
    t_pv = time.time()
    for i in range(0, len(titles), PAGEVIEWS_BATCH):
        batch = titles[i : i + PAGEVIEWS_BATCH]
        try:
            weights.update(fetch_pageviews_batch(batch))
        except requests.RequestException as exc:
            print(f"  pageviews batch [{i}:{i+len(batch)}] HTTP error: {exc}")
        time.sleep(REQUEST_SLEEP)
    elapsed_pv = time.time() - t_pv
    n_zero = sum(1 for t in titles if weights.get(t, 0) == 0)
    p99 = sorted(weights.values())[int(len(weights) * 0.99)] if weights else 0
    p50 = sorted(weights.values())[len(weights) // 2] if weights else 0
    print(
        f"  fetched in {elapsed_pv:.0f}s — "
        f"p50={p50}, p99={p99}, zero-pageview={n_zero}/{len(titles)}"
    )

    # A-ExpJ weighted sample. Cap at k = MAX_ARTICLES.
    if len(titles) > MAX_ARTICLES:
        titles = weighted_sample_no_replacement(
            titles, weights, MAX_ARTICLES, SHUFFLE_SEED
        )
    print(f"  weighted-sampled {len(titles)} titles for fetch.")
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
