"""Shared result types for web_toolkit.

Structured dataclasses returned by every tool so downstream agent code can rely on
typed fields (url, content, ok, error, ...) instead of parsing free-form text. Each
type is JSON-serializable via :meth:`to_dict` for caching, logging, or tool-call
return payloads.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional, TypedDict

__all__ = [
    "SearchResult",
    "FetchResult",
    "BatchFetchResult",
    "BrowseResult",
    "BrowseAction",
]


@dataclass(slots=True)
class SearchResult:
    """One ranked web-search hit (SearXNG/Tavily/DDG normalized to a common shape)."""

    title: str
    url: str
    snippet: str = ""
    engine: str = ""
    score: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FetchResult:
    """Outcome of fetching a single page to clean markdown."""

    url: str
    ok: bool
    content: str = ""
    bytes: int = 0
    error: str = ""
    stealthy: bool = False
    selector: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BatchFetchResult:
    """Aggregated outcome of fetching several pages in parallel."""

    results: list[FetchResult] = field(default_factory=list)

    @property
    def succeeded(self) -> int:
        return sum(1 for r in self.results if r.ok)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.ok)

    def ok_results(self) -> list[FetchResult]:
        return [r for r in self.results if r.ok]

    def to_dict(self) -> dict[str, Any]:
        return {
            "succeeded": self.succeeded,
            "failed": self.failed,
            "results": [r.to_dict() for r in self.results],
        }


@dataclass(slots=True)
class BrowseResult:
    """Outcome of an interactive browser session (agent-browser)."""

    url: str
    ok: bool
    title: str = ""
    content: str = ""
    steps: list[str] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BrowseAction(TypedDict, total=False):
    """One step in a :func:`web_browse` action chain.

    ``type`` is required; the remaining fields depend on the action:
      - click / fill / type / wait_selector: ``selector`` (and ``value`` for fill/type)
      - press: ``key`` (and optional ``selector`` to focus first)
      - wait: ``ms``
      - scroll: ``direction`` ("down"|"up"|"bottom"|"top") and optional ``amount`` px
      - wait_selector: optional ``state`` ("attached"|"visible"|"hidden")
    """

    type: Literal["click", "fill", "type", "press", "wait", "wait_selector", "scroll"]
    selector: str
    value: str
    key: str
    ms: int
    direction: Literal["down", "up", "bottom", "top"]
    amount: int
    state: Literal["attached", "visible", "hidden"]
