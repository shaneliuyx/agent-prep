"""Disk cache for non-deterministic backends (promoted from agent-prep/shared/web_search.py).

A metasearch is inherently non-deterministic: the same query returns a different pool
run-to-run because engines bot-block, time out, or rotate. Caching results on disk gives
reproducible runs and avoids hammering backends during development. Generalized here to
store any JSON-serializable value (list of result dicts), not just list[str].

Env:
  WEB_CACHE       1/0, default 1 (set 0 to always go live)
  WEB_CACHE_PATH  cache file, default ./.web_cache.json
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

__all__ = ["cache_enabled", "cache_lookup", "cache_store"]


def _cache_file() -> Path:
    return Path(os.getenv("WEB_CACHE_PATH", str(Path.cwd() / ".web_cache.json")))


def cache_enabled() -> bool:
    return os.getenv("WEB_CACHE", "1") == "1"


def _read() -> dict[str, Any]:
    f = _cache_file()
    try:
        return json.loads(f.read_text()) if f.exists() else {}
    except (json.JSONDecodeError, OSError):
        return {}


def cache_lookup(key: str) -> Optional[Any]:
    """Return cached value for ``key``, or None on miss / cache-disabled."""
    if not cache_enabled():
        return None
    return _read().get(key)


def cache_store(key: str, value: Any) -> None:
    """Persist ``value`` under ``key`` (no-op when cache disabled).

    Best-effort: a write failure must never break the caller's result.
    """
    if not cache_enabled():
        return
    data = _read()
    data[key] = value
    try:
        _cache_file().write_text(json.dumps(data))
    except OSError:
        pass
