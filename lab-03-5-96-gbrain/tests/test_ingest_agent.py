"""Tests for ingest_agent's deterministic logic: read_sources hardening + the
extract_pages cache. No LLM (the OpenAI client is stubbed), no services.

Run: uv run --with pytest python -m pytest tests/test_ingest_agent.py -v
"""
from __future__ import annotations

import ingest_agent as ia


# ── read_sources: skip dotted parts + non-UTF-8; header each file ────────────
def test_read_sources_skips_dotfiles_and_binary(tmp_path, monkeypatch):
    src = tmp_path / "sources"
    (src / "emails").mkdir(parents=True)
    (src / "emails" / "a.txt").write_text("hello alice")
    (src / "transcripts").mkdir()
    (src / "transcripts" / "b.txt").write_text("dinner notes")
    (src / ".DS_Store").write_bytes(b"\x00\x01\x02 binary")          # dotted binary
    (src / ".omc-state").mkdir()
    (src / ".omc-state" / "inner.txt").write_text("tool state")      # inside dotted dir
    monkeypatch.setattr(ia, "SOURCES", src)

    out = ia.read_sources()
    assert "hello alice" in out and "dinner notes" in out
    assert "binary" not in out and "tool state" not in out, "dotted paths skipped"
    # one header per kept file (relative to sources' parent)
    assert out.count("=====") == 4   # 2 files × 2 header markers ("===== path =====")
    assert "sources/emails/a.txt" in out


def test_read_sources_empty_when_only_dotted(tmp_path, monkeypatch):
    src = tmp_path / "sources"; src.mkdir()
    (src / ".DS_Store").write_bytes(b"\x00")
    monkeypatch.setattr(ia, "SOURCES", src)
    assert ia.read_sources() == ""


# ── extract_pages cache: LLM called once, second call served from cache ──────
class _StubOpenAI:
    """Minimal OpenAI stand-in; counts .chat.completions.create calls."""
    calls = 0

    def __init__(self, *a, **k):
        self.chat = self
        self.completions = self

    def create(self, *a, **k):
        type(self).calls += 1
        content = '{"pages":[{"slug":"people/alice-chen","content":"# Alice Chen"}]}'
        msg = type("M", (), {"content": content})()
        choice = type("Ch", (), {"message": msg})()
        return type("R", (), {"choices": [choice]})()


def test_extract_pages_caches_and_validates(monkeypatch):
    monkeypatch.setattr(ia, "OpenAI", _StubOpenAI)
    monkeypatch.setattr(ia, "_PAGES_CACHE", None)   # fresh cache
    _StubOpenAI.calls = 0

    p1 = ia.extract_pages("raw text one")
    assert p1 == [{"slug": "people/alice-chen", "content": "# Alice Chen"}]
    assert _StubOpenAI.calls == 1, "first call hits the LLM"

    p2 = ia.extract_pages("DIFFERENT raw text")     # cache ignores raw after first
    assert p2 == p1
    assert _StubOpenAI.calls == 1, "second call served from cache (no LLM)"


def test_extract_pages_drops_incomplete_pages(monkeypatch):
    class _PartialStub(_StubOpenAI):
        def create(self, *a, **k):
            type(self).calls += 1
            content = '{"pages":[{"slug":"people/a","content":"# A"},{"slug":"","content":"x"},{"slug":"people/b"}]}'
            msg = type("M", (), {"content": content})()
            choice = type("Ch", (), {"message": msg})()
            return type("R", (), {"choices": [choice]})()

    monkeypatch.setattr(ia, "OpenAI", _PartialStub)
    monkeypatch.setattr(ia, "_PAGES_CACHE", None)
    pages = ia.extract_pages("raw")
    assert pages == [{"slug": "people/a", "content": "# A"}], "drop pages missing slug or content"
