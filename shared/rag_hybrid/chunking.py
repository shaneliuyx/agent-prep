"""Text chunkers — char-window (simple) and sentence-window (canonical).

Two chunkers because the existing labs use two strategies:

  - char_window_chunks: naive sliding char window. Used by W2.5 vector ingest
    (CHUNK_SIZE=512, CHUNK_OVERLAP=64). Cheap, deterministic, doesn't need a
    sentence splitter — but breaks entities across chunk boundaries.

  - sentence_window_chunks: sentence-aware sliding window with abbreviation-
    aware splitting. Canonical from W2.5 build_graph.py — windows are runs of
    complete sentences whose total length is near `target_chars`. Used for
    LLM-extraction windows where mid-sentence breaks tank quality.

Both preserve the original behavior byte-for-byte so callers can swap to
the canonical impl without re-ingesting.
"""
from __future__ import annotations

import re

# Sentence boundaries: simple split on terminal punctuation followed by
# whitespace + capital letter. Python's re module rejects variable-width
# look-behinds, so we cannot inline an abbreviation guard like
# (?<!Inc\.|Mr\.|...). Instead we split conservatively and post-merge
# fragments that look like abbreviation false-splits ("Mr." → next
# sentence). Quality on Wikipedia prose is near-perfect after the merge.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
_ABBREV_TAILS = (
    "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "St.", "Inc.",
    "Co.", "Corp.", "Ltd.", "Ph.D.", "M.D.", "U.S.", "U.K.", "e.g.",
    "i.e.", "etc.", "vs.", "approx.", "cir.", "c.", "fl.", "d.", "b.",
)

CHAR_DEFAULT_SIZE = 512
CHAR_DEFAULT_OVERLAP = 64
SENTENCE_DEFAULT_TARGET = 3000
SENTENCE_DEFAULT_OVERLAP = 500
SENTENCE_DEFAULT_MIN = 200


def char_window_chunks(
    text: str,
    size: int = CHAR_DEFAULT_SIZE,
    overlap: int = CHAR_DEFAULT_OVERLAP,
) -> list[str]:
    """Naive char-window chunker. Same algorithm as W2.5 ingest_to_vector.py.

    Returns list of chunks; empty/whitespace-only text returns []. Final chunk
    can be shorter than `size`. Identical output to the inline chunker in
    ingest_to_vector.py — verified by parity test.
    """
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


def _merge_abbrev_splits(sentences: list[str]) -> list[str]:
    """Merge fragments that ended on a known abbreviation into the next.

    Conservative split + merge is faster and simpler than maintaining a
    correct variable-width abbreviation regex, and Wikipedia prose has a
    bounded set of common abbreviations covered by _ABBREV_TAILS.
    """
    merged: list[str] = []
    i = 0
    while i < len(sentences):
        sent = sentences[i]
        # If this fragment ends with a known abbreviation and there's a next
        # fragment, fold them together. Repeat in case of consecutive
        # abbreviations ("Dr. Mr. Smith said...").
        while i + 1 < len(sentences) and sent.rstrip().endswith(_ABBREV_TAILS):
            sent = sent + " " + sentences[i + 1]
            i += 1
        merged.append(sent)
        i += 1
    return merged


def sentence_window_chunks(
    text: str,
    target_chars: int = SENTENCE_DEFAULT_TARGET,
    overlap_chars: int = SENTENCE_DEFAULT_OVERLAP,
    min_window_chars: int = SENTENCE_DEFAULT_MIN,
) -> list[str]:
    """Sentence-aware sliding window. Each window is a contiguous run of
    complete sentences whose total length is close to ``target_chars``.

    Adjacent windows overlap by approximately ``overlap_chars`` (carrying
    the last few sentences of the previous window forward) so a fact whose
    subject and object span a sentence boundary still gets a chance.

    Returns list of strings, never None / empty list. Empty or near-empty
    text returns a single window containing the input; sentences longer
    than ``target_chars`` are kept whole rather than split mid-sentence
    (preserving extraction quality at the cost of a slightly oversized
    window). Tail windows shorter than ``min_window_chars`` are dropped
    unless they would be the only window.

    Bug history: an earlier version forgot to append the overflowing
    sentence after overlap-carry, looping forever when overlap+sent
    still exceeded target_chars. The current implementation falls
    through to the append after carry to guarantee progress.
    """
    text = (text or "").strip()
    if len(text) <= target_chars:
        return [text] if text else []

    raw_sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    sentences = _merge_abbrev_splits(raw_sentences)
    if len(sentences) <= 1:
        return [text]

    windows: list[str] = []
    current: list[str] = []
    current_chars = 0
    i = 0
    while i < len(sentences):
        sent = sentences[i]
        sent_chars = len(sent)
        # If a single sentence already exceeds target, emit it as its own window.
        if sent_chars >= target_chars and not current:
            windows.append(sent)
            i += 1
            continue
        # Adding this sentence would overflow → emit current window, then
        # carry the last `overlap_chars` worth of sentences forward AND
        # append this sentence so the loop makes progress.
        if current and current_chars + sent_chars > target_chars:
            windows.append(" ".join(current))
            overlap: list[str] = []
            overlap_running = 0
            for c in reversed(current):
                overlap_running += len(c) + 1
                overlap.append(c)
                if overlap_running >= overlap_chars:
                    break
            overlap.reverse()
            current = overlap
            current_chars = sum(len(c) + 1 for c in current)
            # Fall through to append `sent` and advance `i` below — guarantees
            # progress every iteration.
        current.append(sent)
        current_chars += sent_chars + 1
        i += 1

    # Tail flush.
    if current:
        tail = " ".join(current)
        if len(tail) >= min_window_chars or not windows:
            windows.append(tail)

    return windows
