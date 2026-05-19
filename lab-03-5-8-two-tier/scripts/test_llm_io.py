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
from scripts.run_longmemeval_oracle import COMPOSE_SYSTEM, JUDGE_PROMPT


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
    },
    {
        "label": "HARD — empty / off-topic context (should abstain)",
        "context": [
            "User deployed a new version of the API to production.",
            "User configured Terraform with VPC peering to the data-lake account.",
        ],
        "question": "Which vehicle did I take care of first in February, the bike or the car?",
        "gold": "bike",  # gold says bike but context has zero vehicle info → agent should abstain
    },
]


def extract_answer(raw: str) -> str:
    """Same parser the runner uses. Extract from <answer>...</answer>
    tags; fall back to last non-empty line."""
    m = re.search(r"<answer>(.*?)</answer>", raw, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    return lines[-1] if lines else raw


def parse_verdict(judge_raw: str) -> str:
    """Same parser the runner uses for the judge output."""
    judge_upper = judge_raw.upper()
    last_correct = judge_upper.rfind("CORRECT")
    last_incorrect = judge_upper.rfind("INCORRECT")
    if last_incorrect > -1 and last_incorrect + 2 >= last_correct:
        return "INCORRECT"
    if last_correct > -1:
        return "CORRECT"
    return "UNKNOWN"


def run_one_fixture(fix: dict, llm: OpenAI, model: str, judge_model: str) -> None:
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


def main() -> None:
    llm = OpenAI(base_url=os.getenv("OMLX_BASE_URL"), api_key=os.getenv("OMLX_API_KEY"))
    model = os.getenv("MODEL_HAIKU", "Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit")
    judge_model = os.getenv("MODEL_JUDGE", model)

    print(f"Compose model: {model}")
    print(f"Judge model:   {judge_model}")
    print(f"OMLX_BASE_URL: {os.getenv('OMLX_BASE_URL')}")

    for fix in FIXTURES:
        try:
            run_one_fixture(fix, llm, model, judge_model)
        except Exception as e:                                       # noqa: BLE001
            print(f"\nERROR on {fix['label']}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
