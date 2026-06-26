"""web_fetch / web_batch_fetch — read pages as clean markdown.

Two-tier strategy, best-first with graceful degradation:

  1. **scrapling CLI** (if on PATH) — the strong path: ``scrapling extract fetch``
     (browser fetcher) falling back to ``scrapling extract get`` (plain HTTP); a
     ``stealthy`` mode (``stealthy-fetch``) bypasses anti-bot with NO GET fallback.
  2. **stdlib HTTP** (always available) — when scrapling is absent OR its fetch
     errors, a pure ``urllib`` GET + minimal HTML→markdown converter runs. No
     install required; it just can't execute JS or evade anti-bot.

So web_fetch works out of the box; scrapling is an *optional* upgrade for
JS-rendered / anti-bot pages. Optional CSS ``selector`` extracts a region (scrapling
path only — the stdlib fallback fetches the whole page and notes selector ignored).

Env: SCRAPLING_BIN (override the ``scrapling`` executable), WEB_FETCH_UA (override
the stdlib fallback User-Agent).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional

from ._cache import cache_enabled, cache_lookup, cache_store
from ._types import BatchFetchResult, FetchResult

__all__ = ["web_fetch", "web_batch_fetch", "FetchError", "scrapling_available"]

_SCRAPLING = os.getenv("SCRAPLING_BIN", "scrapling")
_DEFAULT_TIMEOUT = 60
_UA = os.getenv(
    "WEB_FETCH_UA",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
)


class FetchError(RuntimeError):
    """Raised only for an unsupported URL scheme.

    Per-page failures (404, blocked, timeout) use ``FetchResult(ok=False, error=...)``.
    """


def scrapling_available() -> bool:
    return shutil.which(_SCRAPLING) is not None


# -- scrapling path --------------------------------------------------------------


class _ScraplingMissing(RuntimeError):
    """Internal: scrapling CLI not found mid-run → caller falls back to stdlib HTTP."""


def _run_scrapling(cmd: str, url: str, out: Path, selector: Optional[str],
                   timeout: int) -> tuple[bool, str]:
    # Output format is inferred from the OUTPUT_FILE extension (.md → markdown).
    args = [_SCRAPLING, "extract", cmd, url, str(out)]
    if selector:
        args += ["--css-selector", selector]
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as err:
        raise _ScraplingMissing(str(err)) from err
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout}s"
    if proc.returncode == 0:
        return True, ""
    return False, (proc.stderr or proc.stdout or f"exit {proc.returncode}").strip()


def _fetch_to_file(url: str, out: Path, selector: Optional[str], stealthy: bool,
                   timeout: int) -> tuple[bool, str]:
    cmd = "stealthy-fetch" if stealthy else "fetch"
    ok, err = _run_scrapling(cmd, url, out, selector, timeout)
    if ok:
        return True, ""
    if stealthy:  # stealthy mode does not fall back to plain GET
        return False, err
    ok2, err2 = _run_scrapling("get", url, out, selector, timeout)
    return (True, "") if ok2 else (False, err or err2)


def _fetch_via_scrapling(url: str, selector: Optional[str], stealthy: bool,
                         timeout: int) -> FetchResult:
    tmpdir = Path(tempfile.mkdtemp(prefix="web-fetch-"))
    out = tmpdir / "page.md"
    try:
        ok, err = _fetch_to_file(url, out, selector, stealthy, timeout)
        if not ok:
            return FetchResult(url=url, ok=False, error=err, stealthy=stealthy, selector=selector)
        content = out.read_text(encoding="utf-8", errors="replace") if out.exists() else ""
        return FetchResult(url=url, ok=True, content=content,
                           bytes=len(content.encode("utf-8")), stealthy=stealthy, selector=selector)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# -- stdlib fallback path --------------------------------------------------------

_SKIP_TAGS = {"script", "style", "noscript", "template", "svg", "head"}
_BLOCK_TAGS = {"p", "div", "section", "article", "header", "footer", "br",
               "ul", "ol", "table", "tr", "blockquote", "pre"}
_HEADINGS = {f"h{i}": "#" * i for i in range(1, 7)}


class _MarkdownExtractor(HTMLParser):
    """Minimal HTML→markdown: keeps headings, links, list items, paragraphs.

    Deliberately small — not a full HTML renderer. Drops scripts/styles, preserves
    link targets (agents need the URLs) and block structure for readability.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0
        self._href: Optional[str] = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag in _HEADINGS:
            self._parts.append(f"\n\n{_HEADINGS[tag]} ")
        elif tag == "li":
            self._parts.append("\n- ")
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n\n")
        elif tag == "a":
            self._href = dict(attrs).get("href")
            self._parts.append("[")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return
        if tag == "a":
            href = self._href or ""
            self._parts.append(f"]({href})" if href else "]")
            self._href = None
        elif tag in _HEADINGS or tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        self._parts.append(data)

    def markdown(self) -> str:
        text = "".join(self._parts)
        text = re.sub(r"[ \t]+", " ", text)        # collapse runs of spaces/tabs
        text = re.sub(r"\n[ \t]+", "\n", text)     # strip leading indent per line
        text = re.sub(r"\n{3,}", "\n\n", text)     # cap blank-line runs
        return text.strip()


def _fetch_via_http(url: str, selector: Optional[str], timeout: int) -> FetchResult:
    req = urllib.request.Request(url, headers={"User-Agent": _UA, "Accept": "text/html,*/*"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - scheme guarded
            charset = resp.headers.get_content_charset() or "utf-8"
            raw = resp.read()
    except urllib.error.HTTPError as err:
        return FetchResult(url=url, ok=False, error=f"HTTP {err.code} {err.reason}", selector=selector)
    except (urllib.error.URLError, ValueError, OSError) as err:
        return FetchResult(url=url, ok=False, error=f"fetch failed: {err}", selector=selector)

    html = raw.decode(charset, errors="replace")
    parser = _MarkdownExtractor()
    parser.feed(html)
    content = parser.markdown()
    if selector:
        content = f"_(selector '{selector}' ignored: stdlib fallback fetches full page)_\n\n{content}"
    return FetchResult(url=url, ok=True, content=content,
                       bytes=len(content.encode("utf-8")), selector=selector)


# -- public API ------------------------------------------------------------------


def _guard_scheme(url: str) -> None:
    if not url.lower().startswith(("http://", "https://")):
        raise FetchError(f"unsupported URL scheme (need http/https): {url!r}")


def web_fetch(url: str, *, selector: Optional[str] = None, stealthy: bool = False,
              timeout: int = _DEFAULT_TIMEOUT, use_cache: bool = True) -> FetchResult:
    """Fetch a single page and return its content as markdown in a :class:`FetchResult`.

    Tries the scrapling CLI when present; otherwise (or if scrapling fails to start)
    degrades to a pure-stdlib ``urllib`` GET + HTML→markdown converter.

    Args:
        url: full URL (must be http/https).
        selector: optional CSS selector (scrapling path only; ignored by the fallback).
        stealthy: anti-bot browser mode (scrapling only; ignored by the fallback).
        timeout: per-fetch timeout in seconds.
        use_cache: when True (default), read/write from the shared .web_cache.json
            disk cache (keyed by url+selector). Set False to force a live fetch.

    Raises:
        FetchError: the URL scheme is unsupported. Network/HTTP failures return
            ``FetchResult(ok=False, error=...)`` instead of raising.
    """
    _guard_scheme(url)
    _cache_key = f"fetch:{url}:{selector}"
    if use_cache and cache_enabled():
        hit = cache_lookup(_cache_key)
        if hit is not None:
            return FetchResult(**hit)
    if scrapling_available():
        try:
            result = _fetch_via_scrapling(url, selector, stealthy, timeout)
        except _ScraplingMissing:
            result = _fetch_via_http(url, selector, timeout)
    else:
        result = _fetch_via_http(url, selector, timeout)
    if use_cache and cache_enabled() and result.ok:
        cache_store(_cache_key, {
            "url": result.url, "ok": result.ok,
            "content": result.content, "bytes": result.bytes,
            "selector": result.selector, "stealthy": result.stealthy,
        })
    return result


def web_batch_fetch(urls: list[str], *, selector: Optional[str] = None,
                    stealthy: bool = False, max_concurrency: int = 3,
                    timeout: int = _DEFAULT_TIMEOUT) -> BatchFetchResult:
    """Fetch several pages in parallel (order preserved). 2-5 URLs recommended.

    A page that fails is recorded as ``ok=False`` and does not abort the others.

    Args:
        urls: URLs to fetch (1-15 sensible).
        selector: CSS selector applied to ALL pages (scrapling path only).
        stealthy: anti-bot mode for all requests (scrapling only).
        max_concurrency: max parallel fetches (clamped to 1-8).
        timeout: per-fetch timeout in seconds.
    """
    if not urls:
        return BatchFetchResult(results=[])
    workers = max(1, min(8, int(max_concurrency), len(urls)))

    def _one(u: str) -> FetchResult:
        try:
            return web_fetch(u, selector=selector, stealthy=stealthy, timeout=timeout)
        except FetchError as err:  # bad scheme etc. — record, don't abort the batch
            return FetchResult(url=u, ok=False, error=str(err), selector=selector)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_one, urls))  # preserves input order
    return BatchFetchResult(results=results)
