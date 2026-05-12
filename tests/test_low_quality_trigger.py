"""Tests for the composite-signal low-quality trigger that determines
when AgenticTreeRetriever should fire chunk-level fallback.

Replaces the simple `(empty OR "insufficient context")` check with a
broader heuristic that catches refusal-class answers, pseudo-refusals
(too short to be substantive), explicit refusal phrases, and ungrounded
answers (no [page N] citation).

PRODUCTION SIGNAL ONLY — never reads judge scores or any test-time
oracle. Goodhart-safe.
"""
from __future__ import annotations

from tree_index.agentic import _is_low_quality


# ===== Tier 1: HIGH CONFIDENCE — should fire =====

def test_empty_answer_is_low_quality() -> None:
    assert _is_low_quality("") is True


def test_whitespace_only_is_low_quality() -> None:
    assert _is_low_quality("   \n\t  ") is True


def test_literal_insufficient_context_is_low_quality() -> None:
    assert _is_low_quality("insufficient context") is True


def test_insufficient_context_in_longer_answer_is_low_quality() -> None:
    """Even when wrapped in prose, the literal phrase signals refusal."""
    msg = ("The provided document is the Berkshire Hathaway 2023 Annual "
           "Report which does not cover this topic. insufficient context")
    assert _is_low_quality(msg) is True


def test_idk_pattern_is_low_quality() -> None:
    msg = "I don't have information about that in this document."
    assert _is_low_quality(msg) is True


def test_cannot_find_pattern_is_low_quality() -> None:
    msg = "I cannot find any reference to that in the provided text."
    assert _is_low_quality(msg) is True


def test_unable_pattern_is_low_quality() -> None:
    msg = "I am unable to locate that information in the document."
    assert _is_low_quality(msg) is True


def test_document_does_not_pattern_is_low_quality() -> None:
    msg = "The document does not contain information about Apple's 2024 revenue."
    assert _is_low_quality(msg) is True


def test_no_information_about_pattern_is_low_quality() -> None:
    msg = "There is no information about Microsoft in this annual report."
    assert _is_low_quality(msg) is True


def test_short_answer_is_low_quality() -> None:
    """< 80 chars suggests pseudo-refusal."""
    assert _is_low_quality("Yes, in Item 1.") is True
    assert _is_low_quality("$364B revenue.") is True


# ===== Tier 2: MEDIUM CONFIDENCE — fires only with co-signal =====

def test_long_ungrounded_answer_without_citation_is_low_quality() -> None:
    """> 80 chars but no [page N] / [pages X-Y] citation suggests
    ungrounded synthesis."""
    msg = ("Berkshire Hathaway is a holding company led by Warren Buffett "
           "that owns several insurance and operating businesses across "
           "many sectors of the American economy.")
    assert _is_low_quality(msg) is True


# ===== Tier 1 negatives: substantive grounded answers should NOT fire =====

def test_substantive_factoid_with_citation_is_high_quality() -> None:
    msg = ("Berkshire's total revenues in 2023 were $364,482 million "
           "($364.5 billion) per the Consolidated Statements of Earnings "
           "[page 96].")
    assert _is_low_quality(msg) is False


def test_substantive_synthesis_with_pages_citation_is_high_quality() -> None:
    msg = ("Buffett describes Berkshire's not-so-secret weapon as the "
           "ability to immediately respond to market seizures with both "
           "huge sums and certainty of performance, citing historical "
           "panics in 1914, 2001, and 2008 [pages 9-9].")
    assert _is_low_quality(msg) is False


def test_substantive_citation_with_item_reference() -> None:
    """[pages 49-51] form should count as citation."""
    msg = ("Risk Factors are covered by Item 1A of Form 10-K [pages 49-51]. "
           "This is the standard SEC structure where Item 1A follows Item 1 "
           "(Business Description).")
    assert _is_low_quality(msg) is False


# ===== Tier 3 negatives: hedging language should NOT fire =====

def test_hedging_language_in_substantive_answer_does_not_trigger() -> None:
    """'approximately'/'about' are common in legitimate factoids; must
    not fire on its own. Co-signal of citation suppresses it."""
    msg = ("Berkshire owns approximately 27.8% of Occidental Petroleum's "
           "common shares as of year-end 2023 [page 11].")
    assert _is_low_quality(msg) is False


def test_might_in_substantive_answer_does_not_trigger() -> None:
    msg = ("BNSF Railway operating results are covered in Item 1 (Business "
           "Description) at pages 30-34, where Buffett notes the figures "
           "might vary year over year [pages 30-34].")
    assert _is_low_quality(msg) is False
