"""Quick iteration harness for LongMemEval compose + judge prompts.

Bypasses Qdrant + consolidation. Hardcodes context + question + gold,
then calls the compose model + judge to see exactly what each
produces. Lets you iterate on COMPOSE_SYSTEM + JUDGE_PROMPT in
seconds instead of waiting for full eval runs.

Run:
    uv run python scripts/test_llm_io.py
    MODEL_HAIKU=<override> uv run python scripts/test_llm_io.py
"""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI

# Import the prompts from the runner so we test the SAME prompts the
# eval uses. Keeps this harness in sync with chapter-shipping runner.
from scripts.run_longmemeval_oracle import COMPOSE_SYSTEM, JUDGE_PROMPT, parse_verdict


# ─── Test fixtures: 3 LongMemEval-shaped questions ───
# Each fixture: context lines (mimicking what query_context() returns)
# + question + expected gold answer + a label for output organization.

FIXTURES = [
    {
        "label": "EASY — direct fact in context",
        "context": [
            "User mentioned the GPS system was not functioning correctly after the first service.",
            "User said the car was returned to the dealership for diagnostics.",
            "User noted the dealer fixed the GPS by updating the firmware.",
        ],
        "question": "What was the first issue I had with my new car after its first service?",
        "gold": "GPS system not functioning correctly",
        "expected_verdict": "CORRECT",
    },
    {
        "label": "MEDIUM — multi-hop ordering",
        "context": [
            "User attended the 'Data Analysis using Python' webinar on January 10.",
            "User attended the 'Effective Time Management' workshop on January 15.",
            "User said the webinar was useful for understanding pandas.",
        ],
        "question": "Which event did I attend first, the 'Effective Time Management' workshop or the 'Data Analysis using Python' webinar?",
        "gold": "'Data Analysis using Python' webinar",
        "expected_verdict": "CORRECT",
    },
    {
        # Tests TWO things at once:
        #   1. Agent must abstain → expected_agent_answer = NO_ANSWER_IN_CONTEXT
        #   2. Judge must mark abstention-vs-concrete-gold as INCORRECT
        #      (eval rule: abstaining when answerable = wrong).
        # Both ✅ means the fixture passed as designed.
        "label": "HARD — empty / off-topic context (agent abstains, judge rejects)",
        "context": [
            "User deployed a new version of the API to production.",
            "User configured Terraform with VPC peering to the data-lake account.",
        ],
        "question": "Which vehicle did I take care of first in February, the bike or the car?",
        "gold": "bike",
        "expected_agent_answer": "NO_ANSWER_IN_CONTEXT",
        "expected_verdict": "INCORRECT",
    },
]


def extract_answer(raw: str) -> str:
    """Same parser the runner uses. Take LAST <answer> match (not first)
    to skip CoT prompt-template echo. Reject sentinel `...` and empty
    content. Fall back to last non-empty line."""
    matches = re.findall(r"<answer>(.*?)</answer>", raw, re.DOTALL | re.IGNORECASE)
    for cand in reversed(matches):
        cand = cand.strip()
        if cand and cand != "...":
            return cand
    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    return lines[-1] if lines else raw


def run_one_fixture(fix: dict, llm: OpenAI, model: str, judge_model: str) -> bool:
    print(f"\n{'='*70}")
    print(f"{fix['label']}")
    print(f"{'='*70}")
    print(f"QUESTION: {fix['question']}")
    print(f"GOLD:     {fix['gold']}")
    print(f"CONTEXT:")
    for line in fix["context"]:
        print(f"  - {line}")

    # ─── Compose ───
    ctx = "\n".join(f"- {l}" for l in fix["context"])
    t0 = time.perf_counter()
    resp = llm.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": COMPOSE_SYSTEM},
            {"role": "user",
             "content": f"Context:\n{ctx}\n\nQuestion: {fix['question']}"},
        ],
        temperature=0.0,
        max_tokens=600,
    )
    compose_s = time.perf_counter() - t0
    raw_compose = (resp.choices[0].message.content or "").strip()
    agent_answer = extract_answer(raw_compose)

    print(f"\n--- COMPOSE ({compose_s:.1f}s) ---")
    print(f"RAW (first 600 chars):\n{raw_compose[:600]}")
    print(f"\nEXTRACTED: {agent_answer!r}")

    # ─── Judge ───
    t1 = time.perf_counter()
    judge_resp = llm.chat.completions.create(
        model=judge_model,
        messages=[{"role": "user",
                   "content": JUDGE_PROMPT.format(
                       question=fix["question"],
                       gold=fix["gold"],
                       answer=agent_answer,
                   )}],
        temperature=0.0,
        max_tokens=400,
    )
    judge_s = time.perf_counter() - t1
    judge_raw = (judge_resp.choices[0].message.content or "").strip()
    verdict = parse_verdict(judge_raw)

    print(f"\n--- JUDGE ({judge_s:.1f}s) ---")
    print(f"RAW (first 400 chars):\n{judge_raw[:400]}")
    print(f"\nVERDICT: {verdict}")

    # ─── Pass/fail vs expected ───
    expected_verdict = fix.get("expected_verdict")
    expected_agent = fix.get("expected_agent_answer")
    checks = []
    if expected_agent is not None:
        agent_ok = agent_answer.strip().upper() == expected_agent.strip().upper()
        checks.append(("agent_answer", expected_agent, agent_answer, agent_ok))
    if expected_verdict is not None:
        verdict_ok = verdict == expected_verdict
        checks.append(("verdict", expected_verdict, verdict, verdict_ok))

    if not checks:
        print("\n[no expected_* fields → not asserted]")
        return True

    all_ok = all(ok for _, _, _, ok in checks)
    print(f"\n--- EXPECTATION CHECK ---")
    for name, want, got, ok in checks:
        flag = "PASS" if ok else "FAIL"
        print(f"  [{flag}] {name}: want={want!r} got={got!r}")
    print(f"\nFIXTURE: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


def main() -> None:
    llm = OpenAI(base_url=os.getenv("OMLX_BASE_URL"), api_key=os.getenv("OMLX_API_KEY"))
    model = os.getenv("MODEL_HAIKU", "Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit")
    judge_model = os.getenv("MODEL_JUDGE", model)

    print(f"Compose model: {model}")
    print(f"Judge model:   {judge_model}")
    print(f"OMLX_BASE_URL: {os.getenv('OMLX_BASE_URL')}")

    passed = 0
    failed = 0
    errors = 0
    for fix in FIXTURES:
        try:
            ok = run_one_fixture(fix, llm, model, judge_model)
            if ok:
                passed += 1
            else:
                failed += 1
        except Exception as e:                                       # noqa: BLE001
            print(f"\nERROR on {fix['label']}: {type(e).__name__}: {e}")
            errors += 1

    total = passed + failed + errors
    print(f"\n{'='*70}")
    print(f"SMOKE RESULT: {passed}/{total} passed ({failed} failed, {errors} errors)")
    print(f"{'='*70}")
    sys.exit(0 if failed == 0 and errors == 0 else 1)


if __name__ == "__main__":
    main()
