"""Tests for ground-truth-based judge (Option C — GT scoring for the 4
ambiguous cross-section questions where entity-recall has been unreliable).

GT judge takes (question, gt_passage_or_answer, pass_criteria, candidate_answer)
and returns (passed: bool, rationale: str). Replaces entity-bag scoring
for these specific questions while leaving the entity-recall judge intact
for the other 12 questions.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Inject lab src onto path once at import time
_src_dir = str(
    Path(__file__).resolve().parents[1] / "lab-02-7-pageindex" / "src"
)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import gt_judge  # noqa: E402


def _mock_judge_response(content: str) -> MagicMock:
    """Build a fake OpenAI chat.completion response object with given content."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


def test_score_against_ground_truth_passes_when_judge_says_yes() -> None:
    client = MagicMock()
    client.chat.completions.create = MagicMock(return_value=_mock_judge_response(
        '{"passed": true, "rationale": "candidate captures the core idea"}'
    ))

    passed, rationale = gt_judge.score_against_ground_truth(
        client=client,
        model="test-model",
        question="What is X?",
        gt_answer="X is the answer",
        pass_criteria="Must mention X",
        candidate_answer="The answer is X.",
    )
    assert passed is True
    assert "core idea" in rationale


def test_score_against_ground_truth_fails_when_judge_says_no() -> None:
    client = MagicMock()
    client.chat.completions.create = MagicMock(return_value=_mock_judge_response(
        '{"passed": false, "rationale": "candidate is off-topic"}'
    ))

    passed, _ = gt_judge.score_against_ground_truth(
        client=client, model="test-model",
        question="What is X?", gt_answer="X is the answer",
        pass_criteria="Must mention X", candidate_answer="Something unrelated.",
    )
    assert passed is False


def test_score_against_ground_truth_handles_malformed_json() -> None:
    """Judge LLM may return non-JSON. Function should not crash; return
    a conservative fail with the parse error in rationale."""
    client = MagicMock()
    client.chat.completions.create = MagicMock(return_value=_mock_judge_response(
        "Sorry, I cannot judge this."  # not JSON
    ))

    passed, rationale = gt_judge.score_against_ground_truth(
        client=client, model="test-model",
        question="Q", gt_answer="GT", pass_criteria="C", candidate_answer="A",
    )
    assert passed is False
    assert "parse" in rationale.lower() or "error" in rationale.lower() \
        or "JSONDecodeError" in rationale or "json" in rationale.lower()


def test_score_against_ground_truth_handles_llm_exception() -> None:
    """LLM call failure should produce conservative fail, not propagate."""
    client = MagicMock()
    client.chat.completions.create = MagicMock(
        side_effect=RuntimeError("connection refused")
    )

    passed, rationale = gt_judge.score_against_ground_truth(
        client=client, model="test-model",
        question="Q", gt_answer="GT", pass_criteria="C", candidate_answer="A",
    )
    assert passed is False
    assert "error" in rationale.lower() or "exception" in rationale.lower() \
        or "RuntimeError" in rationale


def test_score_against_ground_truth_passes_question_in_user_message() -> None:
    """Verify the LLM call wires question, gt_answer, pass_criteria,
    candidate_answer into the user message — no field silently dropped."""
    client = MagicMock()
    client.chat.completions.create = MagicMock(return_value=_mock_judge_response(
        '{"passed": true, "rationale": "ok"}'
    ))

    gt_judge.score_against_ground_truth(
        client=client, model="test-model",
        question="UNIQUE-Q-MARKER",
        gt_answer="UNIQUE-GT-MARKER",
        pass_criteria="UNIQUE-CRIT-MARKER",
        candidate_answer="UNIQUE-ANS-MARKER",
    )
    call_kwargs = client.chat.completions.create.call_args.kwargs
    user_msg = call_kwargs["messages"][-1]["content"]
    assert "UNIQUE-Q-MARKER" in user_msg
    assert "UNIQUE-GT-MARKER" in user_msg
    assert "UNIQUE-CRIT-MARKER" in user_msg
    assert "UNIQUE-ANS-MARKER" in user_msg


def test_load_ground_truth_returns_dict_keyed_by_q_id() -> None:
    """Loader reads the eval_ground_truth.json artifact and returns a
    dict keyed by question id (e.g., 'Q3', 'Q9', 'Q11', 'Q12')."""
    gt = gt_judge.load_ground_truth(
        Path(__file__).resolve().parents[1]
        / "lab-02-7-pageindex" / "data" / "eval_ground_truth.json"
    )
    assert "Q3" in gt
    assert "Q9" in gt
    assert "Q11" in gt
    assert "Q12" in gt
    for qid in ["Q3", "Q9", "Q11", "Q12"]:
        entry = gt[qid]
        assert "q" in entry
        assert "gt_answer" in entry
        assert "pass_criteria" in entry


def test_load_ground_truth_raises_helpfully_when_missing(tmp_path: Path) -> None:
    missing = tmp_path / "no_such.json"
    with pytest.raises(FileNotFoundError, match="ground.truth"):
        gt_judge.load_ground_truth(missing)
