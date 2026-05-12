"""Ground-truth-based judge — alternative to entity-recall scoring for
the 4 cross-section questions where the entity-bag judge has been
demonstrably unreliable (Q3, Q9, Q11, Q12).

Approach: instead of asking "what fraction of expected entities does the
candidate mention", ask "given the question, the ground-truth answer
extracted from the source PDF, and the pass criteria, does the candidate
answer pass?". The judge LLM returns a binary pass/fail + rationale.

Used in Phase 7 follow-up to test the hypothesis that Q3-class
"long partial answer" failures are eval-design artifacts (entity bag
includes terms from OTHER sections of the document) rather than
retrieval failures.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


GT_JUDGE_SYSTEM = """You are an evaluation judge. You receive:
  - A question about a document
  - A ground-truth answer extracted directly from the document
  - The pass criteria for what counts as a correct candidate answer
  - A candidate answer to evaluate

Decide whether the candidate answer PASSES the criteria. Be strict:
the candidate must capture the substantive answer, not just mention
related keywords.

Output strict JSON only:
  {"passed": true | false, "rationale": "<one-sentence explanation>"}

Use the EXACT JSON schema above. No markdown, no commentary outside the JSON."""


def score_against_ground_truth(
    *,
    client,
    model: str,
    question: str,
    gt_answer: str,
    pass_criteria: str,
    candidate_answer: str,
) -> tuple[bool, str]:
    """Run a binary pass/fail judge on (question, GT, candidate).

    Returns (passed, rationale). Conservative on errors: parse failures
    and LLM exceptions return (False, error_message) so a broken judge
    can never inflate a score.
    """
    user_msg = (
        f"Question:\n{question}\n\n"
        f"Ground-truth answer (from source document):\n{gt_answer}\n\n"
        f"Pass criteria:\n{pass_criteria}\n\n"
        f"Candidate answer:\n{candidate_answer}"
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.0,
            max_tokens=300,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": GT_JUDGE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
        )
    except Exception as e:                                  # noqa: BLE001

        return False, f"LLM error: {type(e).__name__}: {e}"

    content = (resp.choices[0].message.content or "").strip()
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as e:
        return False, f"JSON parse error: {e}; raw={content[:120]!r}"

    if not isinstance(parsed, dict) or "passed" not in parsed:
        return False, f"missing 'passed' field; raw={content[:120]!r}"
    passed = bool(parsed["passed"])
    rationale = str(parsed.get("rationale", ""))
    return passed, rationale


def load_ground_truth(path: Path) -> dict[str, dict[str, Any]]:
    """Load eval_ground_truth.json. Returns dict keyed by question id
    (e.g., 'Q3', 'Q9'). Strips the leading '_doc' meta-key if present.

    Raises FileNotFoundError with a helpful hint if the file is missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"ground truth file not found at {path}. "
            f"Curate via lab-02-7-pageindex/data/eval_ground_truth.json"
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if not k.startswith("_")}
