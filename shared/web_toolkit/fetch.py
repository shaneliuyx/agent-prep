"""web_fetch / web_batch_fetch — read pages as clean markdown via the scrapling CLI.

Ports pi-web-toolkit's scrapling strategy to Python subprocess (env-independent: needs
only the ``scrapling`` CLI on PATH, not the Python module):

  - default: ``scrapling extract fetch <url> <out> --ai-targeted`` (browser fetcher),
    falling back to ``scrapling extract get`` (plain HTTP) on failure.
  - stealthy: ``scrapling extract stealthy-fetch ...`` with NO GET fallback (anti-bot).
  - optional CSS ``selector`` extracts only a region (``--css-selector``).

``web_batch_fetch`` runs several fetches concurrently with a bounded worker pool; a
failed page is reported (ok=False, error) but never aborts the batch.

Env: SCRAPLING_BIN (override the ``scrapling`` executable name/path).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from ._types import BatchFetchResult, FetchResult

__all__ = ["web_fetch", "web_batch_fetch", "FetchError", "scrapling_available"]

_SCRAPLING = os.getenv("SCRAPLING_BIN", "scrapling")
_DEFAULT_TIMEOUT = 60


class FetchError(RuntimeError):
    """Raised when scrapling is unavailable (the per-URL failures use FetchResult.ok)."""


def scrapling_available() -> bool:
    return shutil.which(_SCRAPLING) is not None


def _run_scrapling(cmd: str, url: str, out: Path, selector: Optional[str],
                   timeout: int) -> tuple[bool, str]:
    # Output format is inferred from the OUTPUT_FILE extension (.md → markdown).
    args = [_SCRAPLING, "extract", cmd, url, str(out)]
    if selector:
        args += ["--css-selector", selector]
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as err:
        raise FetchError(
            f"'{_SCRAPLING}' not found. Install with: pip install \"scrapling[all]\" "
            "&& scrapling install (or set SCRAPLING_BIN)."
        ) from err
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


def web_fetch(url: str, *, selector: Optional[str] = None, stealthy: bool = False,
              timeout: int = _DEFAULT_TIMEOUT) -> FetchResult:
    """Fetch a single page and return its content as markdown in a :class:`FetchResult`.

    Args:
        url: full URL (include https://).
        selector: optional CSS selector to extract only a region.
        stealthy: anti-bot browser mode (no HTTP GET fallback).
        timeout: per-fetch timeout in seconds.

    Raises:
        FetchError: the scrapling CLI is not installed. Per-page failures (404, blocked)
            return ``FetchResult(ok=False, error=...)`` instead of raising.
    """
    tmpdir = Path(tempfile.mkdtemp(prefix="web-fetch-"))
    out = tmpdir / "page.md"
    try:
        ok, err = _fetch_to_file(url, out, selector, stealthy, timeout)
        if not ok:
            return FetchResult(url=url, ok=False, error=err, stealthy=stealthy, selector=selector)
        content = out.read_text(encoding="utf-8", errors="replace") if out.exists() else ""
        return FetchResult(url=url, ok=True, content=content, bytes=len(content.encode("utf-8")),
                           stealthy=stealthy, selector=selector)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def web_batch_fetch(urls: list[str], *, selector: Optional[str] = None,
                    stealthy: bool = False, max_concurrency: int = 3,
                    timeout: int = _DEFAULT_TIMEOUT) -> BatchFetchResult:
    """Fetch several pages in parallel (order preserved). 2-5 URLs recommended.

    A page that fails is recorded as ``ok=False`` and does not abort the others.

    Args:
        urls: URLs to fetch (1-15 sensible).
        selector: CSS selector applied to ALL pages.
        stealthy: anti-bot mode for all requests.
        max_concurrency: max parallel fetches (clamped to 1-8).
        timeout: per-fetch timeout in seconds.

    Raises:
        FetchError: the scrapling CLI is not installed.
    """
    if not urls:
        return BatchFetchResult(results=[])
    if not scrapling_available():
        raise FetchError(
            f"'{_SCRAPLING}' not found. Install with: pip install \"scrapling[all]\" "
            "&& scrapling install (or set SCRAPLING_BIN)."
        )
    workers = max(1, min(8, int(max_concurrency), len(urls)))

    def _one(u: str) -> FetchResult:
        return web_fetch(u, selector=selector, stealthy=stealthy, timeout=timeout)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_one, urls))  # preserves input order
    return BatchFetchResult(results=results)
