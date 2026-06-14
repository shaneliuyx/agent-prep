"""web_search — discover ranked results across backends (merged module).

This is the single canonical web-search home for the repo. It carries TWO contracts over
one backend+cache core:

  * `web_search(...) -> list[SearchResult]`   — agent-facing structured results (paginated,
    de-duped, scored). Use inside an agent action-space.
  * `web_search_text(query, k) -> list[str]`  — the legacy RAG web-fallback contract,
    promoted verbatim from the old `shared/web_search.py`: single-page content strings,
    SEARXNG_URL→TAVILY→DDG, and the SAME `web_cache_key` format so existing
    `.web_cache.json` pools replay identically (W3.7 CRAG reproducibility).

Plus the RAG primitives the W3.7 labs import directly: `rerank_results`, `cache_lookup`,
`cache_store`, `web_cache_key`, `web_cache_enabled`.

Backend precedence: SEARXNG_URL → TAVILY_API_KEY → DuckDuckGo. The structured path defaults
SEARXNG_URL to http://localhost:8080; the legacy path keeps the original behavior (no default,
so its cache key matches historical runs exactly).

Env: SEARXNG_URL, SEARXNG_LANGUAGE (default "en"), SEARXNG_ENGINES (optional allowlist),
     TAVILY_API_KEY, WEB_CACHE (1/0), WEB_CACHE_PATH.
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from typing import Any, Optional

from ._cache import cache_enabled, cache_lookup, cache_store
from ._types import SearchResult

__all__ = [
    # structured (agent) API
    "web_search", "SearchError",
    # legacy (RAG) API — promoted from shared/web_search.py
    "web_search_text", "rerank_results",
    "web_cache_key", "web_cache_enabled", "cache_lookup", "cache_store",
]

DEFAULT_SEARXNG_URL = "http://localhost:8080"
_MAX_PAGES = 3

# Suggestions captured from the most recent SearXNG call with empty results.
last_suggestions: list[str] = []


class SearchError(RuntimeError):
    """Raised when the selected backend fails (network, bad config, missing dep)."""


# ============================================================================
# Structured path (agent-facing) — returns list[SearchResult]
# ============================================================================

def _structured_backend_id() -> str:
    """Cache-key prefix capturing determinism-affecting config (structured namespace).
    Uses the localhost default so agent calls work out of the box."""
    url = os.getenv("SEARXNG_URL", DEFAULT_SEARXNG_URL)
    if url:
        return (f"searxng:{url}:{os.getenv('SEARXNG_LANGUAGE', 'en')}"
                f":{os.getenv('SEARXNG_ENGINES', '')}")
    if os.getenv("TAVILY_API_KEY"):
        return "tavily"
    return "ddg"


def _structured_cache_key(query: str, results: int, language: Optional[str]) -> str:
    # Distinct namespace ("v2|...") so the structured pool never collides with the
    # legacy web_cache_key entries that the W3.7 labs replay.
    return f"v2|{_structured_backend_id()}|n={results}|lang={language or ''}|{query}"


def _searxng_structured(base_url: str, query: str, results: int,
                        language: Optional[str]) -> list[SearchResult]:
    """Query SearXNG JSON API, auto-paging up to 3 pages and de-duping by URL.

    Pins ``language`` (SEARXNG_LANGUAGE, default "en") and an optional ``engines`` allowlist
    (SEARXNG_ENGINES) so the same query returns a more stable pool — header-inferred language
    and a rotating engine set are the two biggest variance sources.
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
                title=r.get("title", ""), url=u,
                snippet=(r.get("content") or "").strip(),
                engine=r.get("engine", ""), score=r.get("score"),
            ))
        if len(out) >= results:
            break

    return out[:results]


def _tavily_structured(query: str, results: int) -> list[SearchResult]:
    try:
        from tavily import TavilyClient
    except ImportError as err:
        raise SearchError("TAVILY_API_KEY set but 'tavily-python' not installed "
                          "(pip install tavily-python)") from err
    resp = TavilyClient(api_key=os.environ["TAVILY_API_KEY"]).search(query, max_results=results)
    return [SearchResult(title=r.get("title", ""), url=r.get("url", ""),
                         snippet=(r.get("content") or "").strip(),
                         engine="tavily", score=r.get("score"))
            for r in resp.get("results", []) if r.get("url")]


def _ddg_structured(query: str, results: int) -> list[SearchResult]:
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS  # type: ignore
        except ImportError as err:
            raise SearchError("No search backend available: set SEARXNG_URL or TAVILY_API_KEY, "
                              "or `pip install ddgs`") from err
    with DDGS() as ddg:
        return [SearchResult(title=r.get("title", ""), url=r.get("href", ""),
                             snippet=(r.get("body") or "").strip(), engine="ddg")
                for r in ddg.text(query, max_results=results) if r.get("href")]


def _live_structured(query: str, results: int, language: Optional[str]) -> list[SearchResult]:
    url = os.getenv("SEARXNG_URL", DEFAULT_SEARXNG_URL)
    if url:
        return _searxng_structured(url, query, results, language)
    if os.getenv("TAVILY_API_KEY"):
        return _tavily_structured(query, results)
    return _ddg_structured(query, results)


def web_search(query: str, *, results: int = 10, language: Optional[str] = None,
               use_cache: bool = True) -> list[SearchResult]:
    """Search the web and return up to ``results`` ranked :class:`SearchResult`.

    SearXNG auto-pages up to 3 pages and de-dupes by URL. After an empty result, inspect
    ``web_toolkit.search.last_suggestions`` for SearXNG-provided query refinements.

    Raises:
        SearchError: backend network failure, bad config, or missing optional dep.
    """
    results = max(1, min(60, int(results)))
    key = _structured_cache_key(query, results, language)
    if use_cache:
        hit = cache_lookup(key)
        if hit is not None:
            return [SearchResult(**d) for d in hit]
    found = _live_structured(query, results, language)
    if use_cache and found:
        cache_store(key, [r.to_dict() for r in found])
    return found


# ============================================================================
# Legacy path (RAG web-fallback) — promoted verbatim from shared/web_search.py.
# Single-page, content strings, ORIGINAL cache key → existing .web_cache.json
# pools replay identically (W3.7 reproducibility). Do not "modernize" this path.
# ============================================================================

def web_cache_enabled() -> bool:
    """The WEB_CACHE env toggle (alias of the cache module's switch)."""
    return cache_enabled()


def web_cache_key(query: str, k: int) -> str:
    """Per-query legacy cache key — backend + determinism-config aware, IDENTICAL to the
    original shared/web_search.py so historical cache entries still hit. Note: no localhost
    default (matches the env the W3.7 runs used)."""
    if os.getenv("SEARXNG_URL"):
        backend = (f"searxng:{os.environ['SEARXNG_URL']}:{os.getenv('SEARXNG_LANGUAGE', 'en')}"
                   f":{os.getenv('SEARXNG_ENGINES', '')}")
    else:
        backend = "tavily" if os.getenv("TAVILY_API_KEY") else "ddg"
    return f"{backend}|k={k}|{query}"


def _searxng_legacy(base_url: str, query: str, k: int) -> list[str]:
    """Single-page SearXNG content-string fetch — the original behavior."""
    params = {"q": query, "format": "json",
              "language": os.getenv("SEARXNG_LANGUAGE", "en")}
    engines = os.getenv("SEARXNG_ENGINES", "").strip()
    if engines:
        params["engines"] = engines
    url = base_url.rstrip("/") + "/search?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=15) as resp:  # noqa: S310 - operator-set local URL
        results = json.load(resp).get("results", [])
    return [r["content"] for r in results[:k] if r.get("content")]


def _live_legacy(query: str, k: int) -> list[str]:
    """Original backend precedence (no localhost default): SEARXNG_URL → TAVILY → DDG."""
    if os.getenv("SEARXNG_URL"):
        return _searxng_legacy(os.environ["SEARXNG_URL"], query, k)
    if os.getenv("TAVILY_API_KEY"):
        from tavily import TavilyClient
        resp = TavilyClient(api_key=os.environ["TAVILY_API_KEY"]).search(query, max_results=k)
        return [r["content"] for r in resp.get("results", []) if r.get("content")]
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS  # type: ignore
    with DDGS() as ddg:
        return [r["body"] for r in ddg.text(query, max_results=k) if r.get("body")]


def web_search_text(query: str, k: int = 4) -> list[str]:
    """Cached web-search returning content STRINGS — the legacy RAG web-fallback contract
    (formerly ``shared/web_search.py:web_search``). Cache hit replays; miss fetches live +
    persists, using the original ``web_cache_key`` so historical pools replay identically.
    WEB_CACHE=0 disables (always live)."""
    key = web_cache_key(query, k)
    hit = cache_lookup(key)
    if hit is not None:
        return hit
    docs = _live_legacy(query, k)
    cache_store(key, docs)
    return docs


def rerank_results(query: str, docs: list[str], top_k: int, reranker: Any) -> list[str]:
    """Rerank web-result STRINGS against ``query`` with a cross-encoder, keep top_k.

    The ACCURACY half of the web fallback: a fanned-out comparison reranks each sub-query's
    docs against THAT sub-query so the passage holding that entity's figure surfaces. The
    reranker MODEL stays in ``rag_hybrid`` and is passed IN, so this module stays torch-free.
    ``reranker`` is a rag_hybrid CrossEncoderReranker."""
    if not docs:
        return []
    reranker._ensure_loaded()
    scores = reranker._model.predict([(query, d) for d in docs],
                                     batch_size=reranker.cfg.spec.batch_size)
    return [d for d, _ in sorted(zip(docs, scores), key=lambda x: -x[1])][:top_k]
