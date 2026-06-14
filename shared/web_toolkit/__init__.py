"""web_toolkit — a small, dependency-light web research library for custom agents.

Four tools, structured results, CLI-driven backends (no heavyweight Python deps):

    from web_toolkit import web_search, web_fetch, web_batch_fetch, web_browse

    hits = web_search("rust async runtime comparison", results=5)   # -> list[SearchResult]
    page = web_fetch(hits[0].url)                                    # -> FetchResult
    batch = web_batch_fetch([h.url for h in hits[:3]])               # -> BatchFetchResult
    res = web_browse("https://news.ycombinator.com",                # -> BrowseResult
                     [{"type": "scroll", "direction": "bottom"}],
                     selector=".titleline")

Backends:
  - web_search       SearXNG (default http://localhost:8080) -> Tavily -> DuckDuckGo
  - web_fetch        scrapling CLI (browser fetch, HTTP GET fallback, stealthy mode)
  - web_batch_fetch  scrapling CLI, bounded parallelism
  - web_browse       agent-browser CLI (click/fill/scroll/wait then extract)

Design: results are typed dataclasses (``.to_dict()`` for JSON tool payloads); search is
disk-cached for reproducibility; external tools are invoked as subprocesses so the library
needs only the CLIs on PATH, not their Python modules.
"""
from __future__ import annotations

from ._types import (
    BatchFetchResult,
    BrowseAction,
    BrowseResult,
    FetchResult,
    SearchResult,
)
from .browse import BrowseError, agent_browser_available, web_browse
from .fetch import FetchError, scrapling_available, web_batch_fetch, web_fetch
from .search import SearchError, web_search, web_search_text

__version__ = "0.1.0"

__all__ = [
    # tools
    "web_search",
    "web_search_text",
    "web_fetch",
    "web_batch_fetch",
    "web_browse",
    # result types
    "SearchResult",
    "FetchResult",
    "BatchFetchResult",
    "BrowseResult",
    "BrowseAction",
    # errors
    "SearchError",
    "FetchError",
    "BrowseError",
    # capability probes
    "scrapling_available",
    "agent_browser_available",
    "__version__",
]
