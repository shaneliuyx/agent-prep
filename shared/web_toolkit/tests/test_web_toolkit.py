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

    monkeypatch.setattr(s, "_live_structured", fake_live)
    first = web_search("q", results=3)
    second = web_search("q", results=3)
    assert calls["n"] == 1            # second served from cache
    assert first[0].url == second[0].url == "https://c"


# ---- legacy path (merged from web_search.py) — reproducibility ----------------

def test_web_cache_key_legacy_format(monkeypatch):
    """The legacy key format must be byte-identical to the old web_search.py so existing
    .web_cache.json pools still hit (W3.7 reproducibility)."""
    from web_toolkit import web_cache_key
    monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
    monkeypatch.setenv("SEARXNG_LANGUAGE", "en")
    monkeypatch.delenv("SEARXNG_ENGINES", raising=False)
    assert web_cache_key("foo bar", 4) == "searxng:http://localhost:8080:en:|k=4|foo bar"
    monkeypatch.delenv("SEARXNG_URL", raising=False)
    monkeypatch.setenv("TAVILY_API_KEY", "x")
    assert web_cache_key("q", 2) == "tavily|k=2|q"


def test_web_search_text_replays_legacy_cache(tmp_path, monkeypatch):
    """web_search_text replays an existing legacy-key cache entry WITHOUT hitting the network —
    the guarantee that a finished W3.7 run reproduces after the merge."""
    import json as _json
    from web_toolkit import web_search_text, web_cache_key
    monkeypatch.setenv("WEB_CACHE", "1")
    monkeypatch.setenv("WEB_CACHE_PATH", str(tmp_path / "c.json"))
    monkeypatch.setenv("SEARXNG_URL", "http://localhost:8080")
    monkeypatch.setenv("SEARXNG_LANGUAGE", "en")
    monkeypatch.delenv("SEARXNG_ENGINES", raising=False)

    # Seed a cache entry exactly as the old web_search.py would have written it.
    key = web_cache_key("berkshire revenue", 4)
    (tmp_path / "c.json").write_text(_json.dumps({key: ["doc-A", "doc-B"]}))

    # No network: if it tried to fetch localhost it would connection-refuse in CI.
    out = web_search_text("berkshire revenue", k=4)
    assert out == ["doc-A", "doc-B"]


def test_rerank_results_orders_by_score():
    from web_toolkit import rerank_results

    class FakeReranker:
        class _cfg:
            class spec:
                batch_size = 8
        cfg = _cfg()
        _model = None
        def _ensure_loaded(self):
            self._model = self
        def predict(self, pairs, batch_size):  # higher = more relevant
            return [len(d) for _, d in pairs]   # rank by doc length, descending

    out = rerank_results("q", ["aa", "aaaa", "a"], top_k=2, reranker=FakeReranker())
    assert out == ["aaaa", "aa"]


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
