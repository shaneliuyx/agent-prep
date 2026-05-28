# tests/test_voting.py
from voting import voting_run, aggregate_majority, SolverResult

def test_majority_agreement_high_confidence():
    """All 3 agree → confidence = 1.0."""
    results = [SolverResult(i, "raw", "42") for i in range(3)]
    agg = aggregate_majority(results)
    assert agg["confidence"] == 1.0 and agg["answer"] == "42"

def test_majority_split_lower_confidence():
    """2-of-3 split → confidence = 0.667."""
    results = [SolverResult(0, "raw", "yes"), SolverResult(1, "raw", "yes"), SolverResult(2, "raw", "no")]
    agg = aggregate_majority(results)
    assert agg["confidence"] == 2 / 3 and agg["answer"] == "yes"

def test_voting_returns_three_solver_answers():
    """End-to-end run produces 3 solver answers + aggregate."""
    out = voting_run("Is 2 + 2 = 4?", aggregator="majority")
    assert len(out["solver_answers"]) == 3 and "answer" in out["aggregate"]

def test_llm_judge_returns_answer_in_voted_set():
    """LLM-judge picks one of the solver answers."""
    out = voting_run("What color is the sky?", aggregator="llm-judge")
    assert out["aggregate"]["answer"] in set(out["solver_answers"])
