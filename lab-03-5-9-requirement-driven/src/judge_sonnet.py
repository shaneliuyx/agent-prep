"""LongMemEval judge — claude-sonnet-4-6 via local Claude-Code-router on :8317.

Scope: §8.7 evaluation step ONLY. Compares a predicted answer against the
gold answer for a LongMemEval probe question, returning a binary correctness
verdict + short reasoning. Independent module — does NOT touch the lab's
consolidation / dedup / embedding pipeline (those keep their original
local-MLX clients).

Why a dedicated judge and not a shared LLM client:
    The consolidation pipeline (``src/consolidation.py``) requires strict
    structured-output prompts (one-sentence facts, JSON fact lists). The
    proxy at :8317 mangles those contracts because it routes through Claude
    Code's interactive system prompt. The judge step is conversational by
    nature — it asks "is P equivalent to A given Q?" — so the proxy's
    contract drift doesn't bite here.

Usage::

    from src.judge_sonnet import judge
    verdict = judge(
        question="What did Alice ask about deployments?",
        gold_answer="She asked about the Terraform plan for a new endpoint.",
        predicted_answer="Alice's deployment question was about Terraform.",
    )
    # -> {"correct": True, "score": 1.0, "reason": "..."}
"""
import json
import os
from openai import OpenAI

_BASE_URL = os.getenv("JUDGE_BASE_URL", "http://localhost:8317/v1")
_API_KEY = os.getenv("JUDGE_API_KEY", "dummy-not-used")
_MODEL = os.getenv("JUDGE_MODEL", "claude-sonnet-4-6")

_JUDGE_TEMPLATE = """Task: long-term-memory benchmark judging.

Decide whether the PREDICTED answer is semantically equivalent to the GOLD
answer for the QUESTION. The predicted answer may use different wording —
that is fine. It is wrong if it contradicts the gold answer, omits a
critical fact named in the gold answer, or adds a hallucinated fact not
implied by the gold answer.

Respond with ONE JSON object EXACTLY matching this schema:
{{"correct": true|false, "reason": "<one-sentence rationale, max 30 words>"}}
Output ONLY the JSON object. No preamble, no markdown, no code fences.

QUESTION: {question}
GOLD: {gold}
PREDICTED: {predicted}"""


def _client() -> OpenAI:
    return OpenAI(base_url=_BASE_URL, api_key=_API_KEY)


def judge(question: str, gold_answer: str, predicted_answer: str) -> dict:
    """Score one predicted answer against the gold reference.

    Returns ``{"correct": bool, "score": float, "reason": str}``. ``score``
    is ``1.0`` when correct, ``0.0`` otherwise — kept as a float so callers
    can aggregate with mean() without coercion.

    Falls back to ``{"correct": False, "score": 0.0, "reason": "<parse_error>"}``
    if the judge model returns malformed JSON. This is intentional: a
    judge-side parse failure should NOT silently inflate the score.

    Implementation note: the entire instruction is folded into a single
    user-role message. The Claude-Code-router proxy on :8317 injects its
    own system prompt whenever the ``system`` role is set on a request,
    which breaks JSON-only output contracts. User-only payload honors
    structured-output contracts cleanly (verified 2026-05-25 smoke test).
    """
    user_payload = _JUDGE_TEMPLATE.format(
        question=question, gold=gold_answer, predicted=predicted_answer
    )
    from src.llm_retry import chat_with_retry  # judge is on VibeProxy → ride 503 cooldowns
    resp = chat_with_retry(
        _client(),
        model=_MODEL,
        messages=[{"role": "user", "content": user_payload}],
        temperature=0.0,
        max_tokens=200,
    )
    raw = (resp.choices[0].message.content or "").strip()
    try:
        parsed = json.loads(raw)
        correct = bool(parsed.get("correct", False))
        return {
            "correct": correct,
            "score": 1.0 if correct else 0.0,
            "reason": str(parsed.get("reason", ""))[:200],
        }
    except (json.JSONDecodeError, AttributeError) as exc:
        return {
            "correct": False,
            "score": 0.0,
            "reason": f"<parse_error: {exc.__class__.__name__}; raw={raw[:80]!r}>",
        }
