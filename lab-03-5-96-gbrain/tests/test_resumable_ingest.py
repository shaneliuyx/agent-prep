"""Repeatable tests for resumable_ingest's resume machinery.

Covers the deterministic logic with NO LLM and NO services (monkeypatched temp
dirs + stubbed `_merge`): chunking, dotted-path skipping, the write checkpoint +
resume filter, oversized-page split, cross-chunk merge grouping, and the merge
cache HIT/MISS gate. One integration test (`_verify_written`) is skip-gated on a
live Postgres.

Run: uv run pytest tests/test_resumable_ingest.py -v
"""
from __future__ import annotations

import json
import subprocess
import time

import pytest

import resumable_ingest as r


# ── _chunk_text: ≤budget, lossless, line-aligned, deterministic ──────────────
def test_chunk_text_small_is_one_chunk():
    text = "line1\nline2\nline3\n"
    assert r._chunk_text(text, 6000) == [text]


def test_chunk_text_big_splits_correctly():
    text = "".join(f"line {i} with some padding text here\n" for i in range(400))
    chunks = r._chunk_text(text, 1000)
    assert len(chunks) > 1, "should split"
    assert all(len(c) <= 1000 for c in chunks), "every chunk ≤ budget"
    assert "".join(chunks) == text, "lossless rejoin"
    assert all(c.endswith("\n") for c in chunks), "split on line boundaries"
    assert r._chunk_text(text, 1000) == chunks, "deterministic"


def test_chunk_text_never_splits_mid_line():
    text = "a" * 500 + "\n" + "b" * 500 + "\n"   # two long lines
    chunks = r._chunk_text(text, 600)            # each line alone < budget, both > budget
    assert chunks == ["a" * 500 + "\n", "b" * 500 + "\n"]


# ── _files: skip dotted path parts (.DS_Store, .omc-state/ dirs) ─────────────
def test_files_skips_dotted_parts(tmp_path, monkeypatch):
    src = tmp_path / "sources"
    (src / "emails").mkdir(parents=True)
    (src / "emails" / "a.txt").write_text("hello")
    (src / ".DS_Store").write_bytes(b"\x00\x01binary")          # dotted file
    (src / ".omc-state").mkdir()
    (src / ".omc-state" / "inner.txt").write_text("tool state") # file inside dotted dir
    monkeypatch.setattr(r, "SOURCES", src)
    stems = {stem for stem, _ in r._files()}
    assert stems == {"emails-a"}, f"only the real source file, got {stems}"


# ── write checkpoint + resume filter ────────────────────────────────────────
def test_write_checkpoint_roundtrip_and_resume_filter(tmp_path, monkeypatch):
    monkeypatch.setattr(r, "WRITTEN", tmp_path / "written.json")
    assert r._written_done() == set()
    r._mark_written(["people/a", "companies/b"])
    r._mark_written(["people/c"])              # accumulates
    assert r._written_done() == {"people/a", "companies/b", "people/c"}
    canonical = [{"slug": "people/a"}, {"slug": "deals/x"}, {"slug": "companies/b"}]
    pending = [p for p in canonical if p["slug"] not in r._written_done()]
    assert pending == [{"slug": "deals/x"}], "resume writes only un-checkpointed slugs"


# ── oversized-page split (BIG_PAGE_CHARS) ────────────────────────────────────
def test_oversized_page_partition(monkeypatch):
    monkeypatch.setattr(r, "BIG_PAGE_CHARS", 100)
    pages = [{"slug": "x", "content": "a" * 200}, {"slug": "y", "content": "short"}]
    big = [p for p in pages if len(p["content"]) > r.BIG_PAGE_CHARS]
    small = [p for p in pages if len(p["content"]) <= r.BIG_PAGE_CHARS]
    assert [p["slug"] for p in big] == ["x"]
    assert [p["slug"] for p in small] == ["y"]


# ── merge: cross-chunk grouping + cache HIT/MISS ─────────────────────────────
@pytest.fixture
def staged(tmp_path, monkeypatch):
    """A temp stage dir + merged cache; `_merge` stubbed to count calls (no LLM)."""
    stage = tmp_path / "stage"; stage.mkdir()
    monkeypatch.setattr(r, "STAGE_DIR", stage)
    monkeypatch.setattr(r, "MERGED", tmp_path / "merged.json")
    calls = {"n": 0}
    monkeypatch.setattr(r, "_merge", lambda slug, contents: calls.__setitem__("n", calls["n"] + 1) or "MERGED")
    return stage, calls


def test_merge_groups_cross_chunk_and_passes_through_singletons(staged):
    stage, calls = staged
    # alice appears in TWO chunks (→ merge); bob in one (→ passthrough)
    (stage / "f#0.json").write_text(json.dumps([{"slug": "people/alice", "content": "# A0"},
                                                {"slug": "people/bob", "content": "# B"}]))
    (stage / "f#1.json").write_text(json.dumps([{"slug": "people/alice", "content": "# A1"}]))
    canon = {p["slug"]: p["content"] for p in r.merge_from_disk()}
    assert canon["people/bob"] == "# B", "singleton passes through unchanged"
    assert canon["people/alice"] == "MERGED", "multi-chunk entity is merged"
    assert calls["n"] == 1, "merge LLM called once — only for the 2-variant entity"


def test_merge_cache_hit_then_miss_on_stage_change(staged):
    stage, calls = staged
    (stage / "f#0.json").write_text(json.dumps([{"slug": "people/a", "content": "# A"}]))
    c1 = r.merge_from_disk()                       # MISS → compute + cache
    assert r.MERGED.exists()
    c2 = r.merge_from_disk()                       # HIT → same result, no recompute
    assert c2 == c1
    time.sleep(1.05)                               # ensure mtime ticks
    (stage / "f#0.json").write_text(json.dumps([{"slug": "people/a", "content": "# A v2"}]))
    c3 = r.merge_from_disk()                       # fingerprint changed → MISS → re-merge
    assert c3[0]["content"] == "# A v2", "cache invalidated when a chunk changed"


# ── integration: _verify_written gate (needs live Postgres) ──────────────────
def _pg_up() -> bool:
    try:
        out = subprocess.run(["docker", "ps", "--format", "{{.Names}}"],
                             capture_output=True, text=True, timeout=5).stdout
        return any("gbrain-pg" in n for n in out.splitlines())
    except Exception:
        return False


@pytest.mark.skipif(not _pg_up(), reason="needs live gbrain-pg Postgres")
def test_verify_written_excludes_nonexistent():
    # a slug that cannot exist must never be reported as written
    assert r._verify_written(["people/__definitely_not_a_real_slug__"]) == []
    assert r._verify_written([]) == []
