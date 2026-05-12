"""Tests for AgenticTreeRetriever's chunk-level fallback hook.

Hook fires when:
- agent loop reaches max_iterations AND final_answer is empty/refusal
- AgenticTreeRetriever was constructed with `page_vector_index=PageVectorIndex(...)`

Recovers Q9-class regressions where the variant generator paraphrased a
distinctive document term away and the agent never fetched the right page.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from tree_index.agentic import AgenticTreeRetriever
from tree_index.page_vector_index import PageVectorIndex


def _minimal_tree() -> dict:
    """3-leaf tree: pages 1, 2, 3 each one section."""
    return {
        "node_id": "root", "title": "Root",
        "start_page": 1, "end_page": 3, "summary": "",
        "nodes": [
            {"node_id": f"000{i+1}", "title": f"Section {i+1}",
             "start_page": i+1, "end_page": i+1,
             "summary": f"Section {i+1} summary", "tags": [], "nodes": []}
            for i in range(3)
        ],
    }


def _page_provider_factory():
    """Returns a page_provider that maps page N → 'page N text'."""
    return lambda s, e: "\n\n".join(f"[page {p}]\nbody for page {p}"
                                      for p in range(s, e + 1))


def _identity_dense_embedder(dim: int = 4):
    def encode(text: str) -> np.ndarray:
        v = np.zeros(dim, dtype=np.float32)
        v[(ord(text[0]) if text else 0) % dim] = 1.0
        return v
    return encode


def _make_pvi(num_pages: int = 3, dim: int = 4) -> PageVectorIndex:
    """Build a small PageVectorIndex over `num_pages` distinct dense vecs."""
    page_vecs = np.eye(num_pages, dim, dtype=np.float32)
    return PageVectorIndex(
        dense_embeddings=page_vecs,
        embedder=_identity_dense_embedder(dim=dim),
    )


def test_constructor_accepts_page_vector_index_kwarg() -> None:
    """AgenticTreeRetriever must accept the new optional kwarg without
    breaking existing callers."""
    pvi = _make_pvi()
    retriever = AgenticTreeRetriever(
        tree=_minimal_tree(),
        page_provider=_page_provider_factory(),
        model_client=MagicMock(),
        model_name="test-model",
        system_prompt="test prompt",
        page_vector_index=pvi,
    )
    assert retriever.page_vector_index is pvi


def test_constructor_defaults_page_vector_index_to_none() -> None:
    """Existing callers (no kwarg) must still work — back-compat."""
    retriever = AgenticTreeRetriever(
        tree=_minimal_tree(),
        page_provider=_page_provider_factory(),
        model_client=MagicMock(),
        model_name="test-model",
        system_prompt="test prompt",
    )
    assert retriever.page_vector_index is None


def test_chunk_level_fallback_returns_empty_when_no_index() -> None:
    """Fallback method must safely no-op when page_vector_index is None."""
    retriever = AgenticTreeRetriever(
        tree=_minimal_tree(),
        page_provider=_page_provider_factory(),
        model_client=MagicMock(),
        model_name="test-model",
        system_prompt="test prompt",
    )
    result = retriever._chunk_level_fallback("any query")
    assert result == "", f"expected empty string when no index, got {result!r}"


def test_chunk_level_fallback_calls_llm_with_top_pages() -> None:
    """When pvi is present, fallback fetches top pages via vector match
    and submits them to the LLM with a strict 'answer only from passages'
    prompt. Returns the LLM's answer."""
    pvi = _make_pvi()

    # Mock LLM client to return a non-empty answer
    fake_resp = MagicMock()
    fake_resp.choices = [MagicMock()]
    fake_resp.choices[0].message.content = "The answer is on page 2."
    mock_client = MagicMock()
    mock_client.chat.completions.create = MagicMock(return_value=fake_resp)

    retriever = AgenticTreeRetriever(
        tree=_minimal_tree(),
        page_provider=_page_provider_factory(),
        model_client=mock_client,
        model_name="test-model",
        system_prompt="test prompt",
        page_vector_index=pvi,
    )
    result = retriever._chunk_level_fallback("Apple")  # ord('A')=65, %4=1 → page 2

    assert result == "The answer is on page 2."
    # Confirm exactly one LLM call was made
    assert mock_client.chat.completions.create.call_count == 1
    # Confirm passages were included in the prompt
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    user_msg = call_kwargs["messages"][-1]["content"]
    assert "page 2" in user_msg, f"expected 'page 2' in user prompt, got {user_msg!r}"


def test_chunk_level_fallback_returns_empty_on_llm_refusal() -> None:
    """When fallback LLM returns 'insufficient context', propagate empty
    string — caller decides whether to fall back further or accept refusal."""
    pvi = _make_pvi()

    fake_resp = MagicMock()
    fake_resp.choices = [MagicMock()]
    fake_resp.choices[0].message.content = "insufficient context"
    mock_client = MagicMock()
    mock_client.chat.completions.create = MagicMock(return_value=fake_resp)

    retriever = AgenticTreeRetriever(
        tree=_minimal_tree(),
        page_provider=_page_provider_factory(),
        model_client=mock_client,
        model_name="test-model",
        system_prompt="test prompt",
        page_vector_index=pvi,
    )
    result = retriever._chunk_level_fallback("Apple")
    assert result == "", \
        f"refusal answer must propagate as empty string, got {result!r}"


def test_neighbor_page_expansion_dedup_to_contiguous_ranges() -> None:
    """When top-K page hits are [13, 151, 57], expand each by ±1 neighbor,
    dedup, and emit contiguous (start, end) ranges. Catches multi-page
    topics (e.g. Japanese trading houses spans pages 12-13)."""
    from tree_index.agentic import _expand_with_neighbors
    ranges = _expand_with_neighbors([13, 151, 57], window=1)
    assert ranges == [(12, 14), (56, 58), (150, 152)], \
        f"unexpected ranges: {ranges}"


def test_neighbor_page_expansion_merges_overlapping_neighbors() -> None:
    """Adjacent pages with overlapping windows must merge into one range."""
    from tree_index.agentic import _expand_with_neighbors
    # Pages 13 and 14 with window=1 → {12,13,14, 13,14,15} → merged (12, 15)
    ranges = _expand_with_neighbors([13, 14], window=1)
    assert ranges == [(12, 15)], f"unexpected ranges: {ranges}"


def test_neighbor_page_expansion_clips_at_page_one() -> None:
    """Page 1 with window=2 must clip lower bound to 1, not 0 or negative."""
    from tree_index.agentic import _expand_with_neighbors
    ranges = _expand_with_neighbors([1, 2], window=2)
    assert ranges[0][0] == 1, f"lower bound must be 1, got {ranges[0]}"


def test_chunk_level_fallback_uses_neighbor_expansion(monkeypatch) -> None:
    """End-to-end: when search returns page 13, fallback fetches pages
    12-14 (the neighbor window), not just page 13."""
    pvi = _make_pvi()

    fake_resp = MagicMock()
    fake_resp.choices = [MagicMock()]
    fake_resp.choices[0].message.content = "answer with 9% stake."
    mock_client = MagicMock()
    mock_client.chat.completions.create = MagicMock(return_value=fake_resp)

    fetched_ranges: list[tuple[int, int]] = []

    def tracking_provider(s: int, e: int) -> str:
        fetched_ranges.append((s, e))
        return f"[pages {s}-{e}]\nbody"

    retriever = AgenticTreeRetriever(
        tree=_minimal_tree(),
        page_provider=tracking_provider,
        model_client=mock_client,
        model_name="test-model",
        system_prompt="test prompt",
        page_vector_index=pvi,
    )
    # Force search to return page 2 (synthetic): with window=1 should
    # fetch range (1, 3)
    retriever._chunk_level_fallback("Apple")
    # 3-page synthetic PVI returns all 3 pages [2, 1, 3] for top_k=3.
    # Neighbor expansion window=1: {1,2,3} ∪ {1,2} ∪ {2,3,4} = {1,2,3,4}
    # → contiguous range (1, 4).
    assert (1, 4) in fetched_ranges, f"expected (1,4), got {fetched_ranges}"


def test_chunk_level_fallback_returns_empty_on_llm_error() -> None:
    """When LLM call raises, fallback must swallow the error and return
    empty string. Don't crash the user query just because fallback failed."""
    pvi = _make_pvi()
    mock_client = MagicMock()
    mock_client.chat.completions.create = MagicMock(
        side_effect=RuntimeError("LLM down")
    )

    retriever = AgenticTreeRetriever(
        tree=_minimal_tree(),
        page_provider=_page_provider_factory(),
        model_client=mock_client,
        model_name="test-model",
        system_prompt="test prompt",
        page_vector_index=pvi,
    )
    result = retriever._chunk_level_fallback("Apple")
    assert result == "", \
        f"LLM error must produce empty fallback, got {result!r}"
