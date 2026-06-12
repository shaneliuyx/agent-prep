"""Cached web-search backend (SearXNG → Tavily → DuckDuckGo) — shared infra.

Promoted from lab-03.7-agentic-rag (W3.7 Agentic RAG §3.3) where the CRAG web fallback first
needed it. Both `baseline_handrolled.py` (hand-rolled CRAG) and `crag_variant.py` (LangGraph CRAG)
import it, so the backend + cache live in exactly one place.

Public API:
  web_search(query, k=4)        — cached backend call; precedence SEARXNG_URL → TAVILY_API_KEY → DDG.
  cache_lookup(key) / cache_store(key, docs)
                                — generic disk-cache access for callers that cache at a HIGHER level
                                  than one query (e.g. a fanned-out planned query whose whole doc
                                  pool should replay as one unit). Honor the WEB_CACHE toggle.
  web_cache_key(query, k)       — the per-query cache key (backend + determinism-config aware).
  web_cache_enabled()           — the WEB_CACHE env toggle.

Determinism + accuracy rationale (full story: W3.7 chapter §3.3.1): a metasearch is inherently
non-deterministic — the same query returns a different doc pool run-to-run because engines
bot-block / time out / rotate and language is header-inferred. So SearXNG language + engines are
pinned via env, and results are cached on disk for reproducible runs. Accuracy (each sub-search
yielding its figure, not one entity dominating) is the CALLER's job via per-sub-query rerank — see
`baseline_handrolled.web_search_planned`.

Env: SEARXNG_URL, SEARXNG_LANGUAGE (default "en"), SEARXNG_ENGINES (optional allowlist),
     TAVILY_API_KEY, WEB_CACHE (1/0, default 1), WEB_CACHE_PATH (default ./.web_cache.json).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_WEB_CACHE_FILE = Path(os.getenv("WEB_CACHE_PATH", str(Path.cwd() / ".web_cache.json")))


def web_cache_enabled() -> bool:
    return os.getenv("WEB_CACHE", "1") == "1"


def _read_cache() -> dict[str, list[str]]:
    try:
        return json.loads(_WEB_CACHE_FILE.read_text()) if _WEB_CACHE_FILE.exists() else {}
    except (json.JSONDecodeError, OSError):
        return {}


def cache_lookup(key: str) -> list[str] | None:
    """Return cached docs for `key`, or None on miss / cache-disabled."""
    if not web_cache_enabled():
        return None
    return _read_cache().get(key)


def cache_store(key: str, docs: list[str]) -> None:
    """Persist `docs` under `key` (no-op when cache disabled). Best-effort: a write failure must
    never break the caller's answer."""
    if not web_cache_enabled():
        return
    cache = _read_cache()
    cache[key] = docs
    try:
        _WEB_CACHE_FILE.write_text(json.dumps(cache))
    except OSError:
        pass


def web_cache_key(query: str, k: int) -> str:
    """Key includes the backend + its determinism-affecting config, so switching engine / language /
    backend invalidates stale entries instead of silently replaying a different source's pool."""
    if os.getenv("SEARXNG_URL"):
        backend = (f"searxng:{os.environ['SEARXNG_URL']}:{os.getenv('SEARXNG_LANGUAGE', 'en')}"
                   f":{os.getenv('SEARXNG_ENGINES', '')}")
    else:
        backend = "tavily" if os.getenv("TAVILY_API_KEY") else "ddg"
    return f"{backend}|k={k}|{query}"


def _searxng_search(base_url: str, query: str, k: int) -> list[str]:
    """Query a SearXNG instance's JSON API (free, self-hosted, no key). Aggregates many engines
    (Google, Startpage, ...) and ranks free encyclopedic / press sources ABOVE the Statista paywall
    snippet that Tavily and DuckDuckGo surface first. stdlib urllib only (no curl / no dep).

    STABILITY: pin `language` (env SEARXNG_LANGUAGE, default "en") and an optional `engines`
    allowlist (env SEARXNG_ENGINES, e.g. "google,duckduckgo,wikipedia") so the same query returns a
    more stable pool — auto language detection from headers and a rotating engine set are the two
    biggest variance sources. (Pinning a SMALL engine set can backfire: Google's scraper is
    intermittently bot-blocked, collapsing a tiny pool to empty — so the allowlist defaults to all.)
    """
    import urllib.parse
    import urllib.request
    params = {"q": query, "format": "json",
              "language": os.getenv("SEARXNG_LANGUAGE", "en")}  # pin lang → no header-based drift
    engines = os.getenv("SEARXNG_ENGINES", "").strip()
    if engines:
        params["engines"] = engines                            # pin engine set → stable pool
    url = base_url.rstrip("/") + "/search?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=15) as resp:  # noqa: S310 — operator-set local URL
        results = json.load(resp).get("results", [])
    return [r["content"] for r in results[:k] if r.get("content")]


def _web_search_live(query: str, k: int) -> list[str]:
    """The actual network call, backend precedence:
      1. SEARXNG_URL  — free self-hosted metasearch, best free-source ranking
      2. TAVILY_API_KEY — managed API (good, but ranks paywalled aggregators high)
      3. DuckDuckGo   — free, no key, weakest ranking
    ImportError (missing tavily/ddgs) propagates — the caller's config-bug handler turns it into a
    loud, actionable error rather than a silent abstain."""
    if os.getenv("SEARXNG_URL"):
        return _searxng_search(os.environ["SEARXNG_URL"], query, k)
    if os.getenv("TAVILY_API_KEY"):
        # Official Tavily SDK (tavily-python) — replaces the sunset
        # langchain_community.TavilySearchResults wrapper (deprecated in LC 0.3.25).
        from tavily import TavilyClient
        resp = TavilyClient(api_key=os.environ["TAVILY_API_KEY"]).search(query, max_results=k)
        return [r["content"] for r in resp.get("results", []) if r.get("content")]
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS
    with DDGS() as ddg:
        return [r["body"] for r in ddg.text(query, max_results=k) if r.get("body")]


def web_search(query: str, k: int = 4) -> list[str]:
    """Cached wrapper over _web_search_live for reproducible runs. Cache hit → replay; miss → fetch
    live + persist. WEB_CACHE=0 disables (always live)."""
    key = web_cache_key(query, k)
    hit = cache_lookup(key)
    if hit is not None:
        return hit
    docs = _web_search_live(query, k)
    cache_store(key, docs)
    return docs


def rerank_results(query: str, docs: list[str], top_k: int, reranker: Any) -> list[str]:
    """Rerank web-result STRINGS against `query` with a cross-encoder, keep top_k.

    This is the ACCURACY half of the web fallback. A fanned-out comparison must rerank each
    sub-query's docs against THAT sub-query (not the broad "compare X and Y" query) so the passage
    holding that entity's figure surfaces and neither entity dominates — see
    `baseline_handrolled.web_search_planned`. The reranker MODEL stays in rag_hybrid (a shared
    package); it's passed IN here so this module stays light (pure stdlib + optional web libs, no
    torch import unless the caller already loaded one). `reranker` is a rag_hybrid
    CrossEncoderReranker; the one place that reaches into its predict() lives here, not scattered."""
    if not docs:
        return []
    reranker._ensure_loaded()
    scores = reranker._model.predict([(query, d) for d in docs],
                                     batch_size=reranker.cfg.spec.batch_size)
    return [d for d, _ in sorted(zip(docs, scores), key=lambda x: -x[1])][:top_k]
