"""web_search — discover ranked results across backends.

Backend precedence (first available wins), generalized from agent-prep/shared/web_search.py
and improved with pi-web-toolkit's structured + paginated SearXNG behavior:

  1. SEARXNG_URL (default http://localhost:8080) — free self-hosted metasearch, best
     free-source ranking. Auto-pages up to 3 pages and de-dupes by URL.
  2. TAVILY_API_KEY — managed API (optional dep ``tavily-python``).
  3. DuckDuckGo — free, no key (optional dep ``ddgs``).

Improvements over the original list[str] version:
  - returns structured :class:`SearchResult` (title, url, snippet, engine, score)
  - SearXNG pagination + URL de-dup + query ``suggestions`` capture on empty results
  - disk cache keyed by backend+config so switching engine/language invalidates stale pools

Env: SEARXNG_URL, SEARXNG_LANGUAGE (default "en"), SEARXNG_ENGINES (optional allowlist),
     TAVILY_API_KEY, WEB_CACHE / WEB_CACHE_PATH (see _cache).
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from typing import Optional

from ._cache import cache_lookup, cache_store
from ._types import SearchResult

__all__ = ["web_search", "web_search_text", "SearchError"]

DEFAULT_SEARXNG_URL = "http://localhost:8080"
_MAX_PAGES = 3

# Captured suggestions from the most recent SearXNG call with empty results.
# Read it after a call that returned [] to refine and retry.
last_suggestions: list[str] = []


class SearchError(RuntimeError):
    """Raised when the selected backend fails (network, bad config, missing dep)."""


def _backend_id() -> str:
    """Cache-key prefix that captures determinism-affecting config, so switching
    engine / language / backend invalidates stale entries instead of replaying a
    different source's pool."""
    url = os.getenv("SEARXNG_URL", DEFAULT_SEARXNG_URL)
    if url:
        return (f"searxng:{url}:{os.getenv('SEARXNG_LANGUAGE', 'en')}"
                f":{os.getenv('SEARXNG_ENGINES', '')}")
    if os.getenv("TAVILY_API_KEY"):
        return "tavily"
    return "ddg"


def _cache_key(query: str, results: int, language: Optional[str]) -> str:
    return f"{_backend_id()}|n={results}|lang={language or ''}|{query}"


def _searxng_search(base_url: str, query: str, results: int,
                    language: Optional[str]) -> list[SearchResult]:
    """Query a SearXNG JSON API, auto-paging up to 3 pages and de-duping by URL.

    Pins ``language`` (env SEARXNG_LANGUAGE, default "en") and an optional ``engines``
    allowlist (env SEARXNG_ENGINES) so the same query returns a more stable pool —
    auto language detection and a rotating engine set are the two biggest variance sources.
    """
    global last_suggestions
    last_suggestions = []
    lang = language if language is not None else os.getenv("SEARXNG_LANGUAGE", "en")
    engines = os.getenv("SEARXNG_ENGINES", "").strip()

    seen: set[str] = set()
    out: list[SearchResult] = []
    base = base_url.rstrip("/")

    for page in range(1, _MAX_PAGES + 1):
        params = {"q": query, "format": "json", "pageno": str(page)}
        if lang:
            params["language"] = lang
        if engines:
            params["engines"] = engines
        url = base + "/search?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310 - operator-set URL
                data = json.load(resp)
        except Exception as err:  # noqa: BLE001 - surface as actionable SearchError
            raise SearchError(f"SearXNG request failed at {base}: {err}") from err

        page_results = data.get("results") or []
        if not page_results:
            sugg = data.get("suggestions") or []
            if sugg and not last_suggestions:
                last_suggestions = list(sugg)
            break

        for r in page_results:
            u = r.get("url")
            if not u or u in seen:
                continue
            seen.add(u)
            out.append(SearchResult(
                title=r.get("title", ""),
                url=u,
                snippet=(r.get("content") or "").strip(),
                engine=r.get("engine", ""),
                score=r.get("score"),
            ))
        if len(out) >= results:
            break

    return out[:results]


def _tavily_search(query: str, results: int) -> list[SearchResult]:
    try:
        from tavily import TavilyClient
    except ImportError as err:
        raise SearchError("TAVILY_API_KEY set but 'tavily-python' not installed "
                          "(pip install tavily-python)") from err
    resp = TavilyClient(api_key=os.environ["TAVILY_API_KEY"]).search(query, max_results=results)
    return [
        SearchResult(title=r.get("title", ""), url=r.get("url", ""),
                     snippet=(r.get("content") or "").strip(),
                     engine="tavily", score=r.get("score"))
        for r in resp.get("results", []) if r.get("url")
    ]


def _ddg_search(query: str, results: int) -> list[SearchResult]:
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS  # type: ignore
        except ImportError as err:
            raise SearchError("No search backend available: set SEARXNG_URL or TAVILY_API_KEY, "
                              "or `pip install ddgs`") from err
    with DDGS() as ddg:
        return [
            SearchResult(title=r.get("title", ""), url=r.get("href", ""),
                         snippet=(r.get("body") or "").strip(), engine="ddg")
            for r in ddg.text(query, max_results=results) if r.get("href")
        ]


def _live(query: str, results: int, language: Optional[str]) -> list[SearchResult]:
    url = os.getenv("SEARXNG_URL", DEFAULT_SEARXNG_URL)
    if url:
        return _searxng_search(url, query, results, language)
    if os.getenv("TAVILY_API_KEY"):
        return _tavily_search(query, results)
    return _ddg_search(query, results)


def web_search(query: str, *, results: int = 10,
               language: Optional[str] = None,
               use_cache: bool = True) -> list[SearchResult]:
    """Search the web and return up to ``results`` ranked :class:`SearchResult`.

    Args:
        query: the search query.
        results: max results to return (1-60). SearXNG auto-pages up to 3 pages.
        language: language code (e.g. "en", "de"). None → SEARXNG_LANGUAGE / backend default.
        use_cache: replay from disk cache when available (set False to force live).

    After an empty result, inspect ``web_toolkit.search.last_suggestions`` for
    SearXNG-provided query refinements.

    Raises:
        SearchError: backend network failure, bad config, or missing optional dep.
    """
    results = max(1, min(60, int(results)))
    key = _cache_key(query, results, language)
    if use_cache:
        hit = cache_lookup(key)
        if hit is not None:
            return [SearchResult(**d) for d in hit]
    found = _live(query, results, language)
    if use_cache and found:
        cache_store(key, [r.to_dict() for r in found])
    return found


def web_search_text(query: str, k: int = 4) -> list[str]:
    """Back-compat helper matching the original agent-prep API: return snippet strings."""
    return [r.snippet for r in web_search(query, results=k) if r.snippet]
