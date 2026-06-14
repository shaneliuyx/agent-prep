"""Smoke + unit tests for web_toolkit.

Network/CLI-dependent paths are exercised against live backends only when available
(SearXNG / scrapling / agent-browser); otherwise those tests are skipped. Pure logic
(command building, caching, result types, backend selection) is always tested offline.
"""
from __future__ import annotations

import os
import urllib.request

import pytest

from web_toolkit import (
    BatchFetchResult,
    FetchResult,
    SearchResult,
    agent_browser_available,
    scrapling_available,
    web_search,
)
from web_toolkit.browse import build_action_commands
from web_toolkit.search import DEFAULT_SEARXNG_URL


# ---- offline logic ----------------------------------------------------------

def test_result_types_serialize():
    r = SearchResult(title="t", url="https://x", snippet="s", engine="e", score=1.0)
    assert r.to_dict()["url"] == "https://x"
    b = BatchFetchResult(results=[FetchResult("https://a", True, "hi"),
                                  FetchResult("https://b", False, error="boom")])
    assert (b.succeeded, b.failed) == (1, 1)
    assert [r.url for r in b.ok_results()] == ["https://a"]


def test_build_action_commands_maps_actions():
    cmds = build_action_commands(
        [{"type": "fill", "selector": "#q", "value": "hi"},
         {"type": "press", "key": "Enter"},
         {"type": "scroll", "direction": "bottom"},
         {"type": "wait", "ms": 500}],
    )
    assert ["fill", "#q", "hi"] in cmds
    assert ["press", "Enter"] in cmds
    assert ["eval", "window.scrollTo(0, document.body.scrollHeight)"] in cmds
    assert ["wait", "500"] in cmds


def test_build_action_commands_empty():
    assert build_action_commands([]) == []


def test_search_cache_roundtrip(tmp_path, monkeypatch):
    """A cached search replays from disk without hitting the network."""
    monkeypatch.setenv("WEB_CACHE", "1")
    monkeypatch.setenv("WEB_CACHE_PATH", str(tmp_path / "c.json"))
    monkeypatch.setenv("SEARXNG_URL", "http://unused.invalid")

    import web_toolkit.search as s
    calls = {"n": 0}

    def fake_live(query, results, language):
        calls["n"] += 1
        return [SearchResult(title="cached", url="https://c", snippet="hit")]

    monkeypatch.setattr(s, "_live", fake_live)
    first = web_search("q", results=3)
    second = web_search("q", results=3)
    assert calls["n"] == 1            # second served from cache
    assert first[0].url == second[0].url == "https://c"


# ---- live (skipped when backend unavailable) --------------------------------

def _searxng_up() -> bool:
    url = os.getenv("SEARXNG_URL", DEFAULT_SEARXNG_URL)
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/search?q=test&format=json", timeout=5):
            return True
    except Exception:
        return False


@pytest.mark.skipif(not _searxng_up(), reason="no SearXNG instance reachable")
def test_live_search(monkeypatch, tmp_path):
    monkeypatch.setenv("WEB_CACHE", "0")  # force live
    results = web_search("python programming language", results=5)
    assert results, "expected at least one result from live SearXNG"
    assert all(r.url.startswith("http") for r in results)


@pytest.mark.skipif(not scrapling_available(), reason="scrapling CLI not installed")
def test_live_fetch():
    from web_toolkit import web_fetch
    res = web_fetch("https://example.com")
    assert res.ok and "example" in res.content.lower()


@pytest.mark.skipif(not agent_browser_available(), reason="agent-browser CLI not installed")
def test_live_browse():
    from web_toolkit import web_browse
    res = web_browse("https://example.com", actions=[], selector="h1")
    assert res.ok and res.title
