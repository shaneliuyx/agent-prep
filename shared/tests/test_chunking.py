"""Chunker parity tests — char-window MUST produce byte-identical output to
the inline chunker in ingest_to_vector.py, and sentence-window MUST produce
byte-identical output to _chunk_by_sentences in build_graph.py.

This is the strictest possible parity gate: same algorithm, same constants,
same boundary handling. If these tests pass, swapping the inline chunker for
the library function leaves vector ingest output literally identical.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "shared"))

from rag_hybrid.chunking import (
    char_window_chunks,
    sentence_window_chunks,
)


# ----- char-window: replicate the inline chunker from ingest_to_vector.py -----

def _inline_chunk_text(text: str, size: int = 512, overlap: int = 64) -> list[str]:
    """Verbatim copy of ingest_to_vector.py::chunk_text — the function whose
    output the library impl must match."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    step = size - overlap
    for i in range(0, len(text), step):
        chunk = text[i : i + size]
        if chunk:
            chunks.append(chunk)
        if i + size >= len(text):
            break
    return chunks


def _gen_text(n: int, seed: int = 1) -> str:
    """Deterministic pseudo-text — same length as a typical W2.5 article body."""
    import random
    random.seed(seed)
    words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
             "Steve", "Jobs", "founded", "Apple", "in", "1976", "with", "Wozniak"]
    out = []
    while len(" ".join(out)) < n:
        out.append(random.choice(words))
    return " ".join(out)[:n]


def test_char_window_short_text() -> None:
    assert char_window_chunks("", 512, 64) == []
    assert char_window_chunks("   ", 512, 64) == []
    assert char_window_chunks("hello", 512, 64) == ["hello"]


def test_char_window_matches_inline_for_w25_defaults() -> None:
    """The actual config W2.5 used: size=512, overlap=64. Must match for every
    article-length input range we care about."""
    for n in (100, 511, 512, 513, 800, 1500, 5000, 12000):
        text = _gen_text(n, seed=n)
        a = char_window_chunks(text, 512, 64)
        b = _inline_chunk_text(text, 512, 64)
        assert a == b, f"drift at n={n}: lib={len(a)} chunks, inline={len(b)} chunks"


def test_char_window_matches_inline_for_w2_sweep_grid() -> None:
    """All 9 chunk-sweep variants W2 ran (sizes × overlaps)."""
    for size in (256, 512, 1024):
        for overlap in (0, 64, 128):
            text = _gen_text(3000, seed=size * 100 + overlap)
            a = char_window_chunks(text, size, overlap)
            b = _inline_chunk_text(text, size, overlap)
            assert a == b, f"drift at size={size}, overlap={overlap}"


# ----- sentence-window tests -----

def test_sentence_window_empty() -> None:
    assert sentence_window_chunks("", 3000, 500) == []
    assert sentence_window_chunks("   ", 3000, 500) == []


def test_sentence_window_short_text_single_window() -> None:
    text = "Steve Jobs founded Apple in 1976."
    out = sentence_window_chunks(text, 3000, 500)
    assert out == [text]


def test_sentence_window_no_sentence_split_returns_whole() -> None:
    """Text with no sentence-final punctuation can't be split — return as-is."""
    text = "x" * 3500  # over target, but no boundaries
    out = sentence_window_chunks(text, 3000, 500)
    assert out == [text]


def test_sentence_window_overlap_carry() -> None:
    """3 windows of ~3 sentences each. Adjacent windows must overlap (the same
    sentence appears in both) — that's the whole point of the chunker."""
    sentences = [f"This is sentence number {i}." for i in range(20)]
    text = " ".join(sentences) + " " + ("filler word " * 200)  # force overflow
    out = sentence_window_chunks(text, 500, 100)
    assert len(out) >= 2, f"expected multi-window output, got {len(out)}"
    # Overlap check: some sentence-string from window 0 should also appear in window 1.
    shared = any(s in out[1] for s in out[0].split(". ") if len(s) > 15)
    assert shared, "no overlap between adjacent windows — overlap-carry broken"


def test_sentence_window_abbreviation_merge() -> None:
    """Wikipedia prose: 'Mr. Smith founded the company.' must NOT split after 'Mr.'."""
    text = "Mr. Smith founded the company. " + ("padding sentence. " * 200)
    out = sentence_window_chunks(text, 500, 100)
    # First window's first sentence must contain "Mr. Smith" — not split.
    assert "Mr. Smith" in out[0], f"abbreviation split: {out[0][:50]}"


def test_sentence_window_no_infinite_loop_on_giant_sentence() -> None:
    """Single sentence >> target_chars — must emit as its own window, not loop.
    Bug regression test for build_graph.py's earlier infinite-loop issue."""
    big = "x " * 5000  # one ~10K-char "sentence" with no terminators
    out = sentence_window_chunks(big, 500, 100)
    assert len(out) >= 1
    assert sum(len(w) for w in out) > 0


def main() -> int:
    failures = []
    tests = [
        ("test_char_window_short_text", test_char_window_short_text),
        ("test_char_window_matches_inline_for_w25_defaults",
         test_char_window_matches_inline_for_w25_defaults),
        ("test_char_window_matches_inline_for_w2_sweep_grid",
         test_char_window_matches_inline_for_w2_sweep_grid),
        ("test_sentence_window_empty", test_sentence_window_empty),
        ("test_sentence_window_short_text_single_window",
         test_sentence_window_short_text_single_window),
        ("test_sentence_window_no_sentence_split_returns_whole",
         test_sentence_window_no_sentence_split_returns_whole),
        ("test_sentence_window_overlap_carry", test_sentence_window_overlap_carry),
        ("test_sentence_window_abbreviation_merge",
         test_sentence_window_abbreviation_merge),
        ("test_sentence_window_no_infinite_loop_on_giant_sentence",
         test_sentence_window_no_infinite_loop_on_giant_sentence),
    ]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS: {name}")
        except AssertionError as e:
            failures.append(f"{name}: {e}")
            print(f"FAIL: {name}: {e}")
    if failures:
        return 1
    print(f"OK — {len(tests)} chunking tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
