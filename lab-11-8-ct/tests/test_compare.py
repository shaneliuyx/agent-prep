"""Offline tests for eval-gate comparator."""
from src.compare import compare


def test_pass_when_candidate_better():
    code, msg = compare({"ragas_faithfulness": 0.85},
                       {"ragas_faithfulness": 0.80},
                       "ragas_faithfulness", tolerance=0.02)
    assert code == 0
    assert "PASS" in msg


def test_pass_when_within_tolerance():
    """Small regression within tolerance passes."""
    code, _ = compare({"ragas_faithfulness": 0.795},
                     {"ragas_faithfulness": 0.80},
                     "ragas_faithfulness", tolerance=0.02)
    assert code == 0


def test_fail_when_regressed_beyond_tolerance():
    code, msg = compare({"ragas_faithfulness": 0.75},
                       {"ragas_faithfulness": 0.80},
                       "ragas_faithfulness", tolerance=0.02)
    assert code == 1
    assert "FAIL" in msg


def test_missing_metric_returns_code_2():
    code, _ = compare({}, {"ragas_faithfulness": 0.80},
                     "ragas_faithfulness", tolerance=0.02)
    assert code == 2
