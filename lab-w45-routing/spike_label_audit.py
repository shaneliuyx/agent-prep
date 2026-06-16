"""SPIKE (throwaway) — prove the tier ceiling is LABELS, not capacity (non-gaming).

Method (independent re-adjudication, NOT relabel-to-predictions):
  1. A neutral judge (Opus, rubric-only, blind to original labels, no few-shot)
     re-labels each eval row's tier from scratch.
  2. agreement(original, independent) — low => the boundary is genuinely subjective.
  3. On CONSENSUS rows (original == independent), is the shipped 4B's tier accuracy
     ~100%, and do its misses cluster on DISPUTED rows? If yes, the ceiling is the
     ambiguous rows (labels), not model capability.

Run:  .venv/bin/python spike_label_audit.py
"""
import json
import re

from openai import OpenAI

from src.probes import load_probes, train_eval_split
from src.router import classify

JUDGE_MODEL = "claude-opus-4-8"
JUDGE_BASE = "http://localhost:8317/v1"
JUDGE_KEY = "vibeproxy"

# Rubric ONLY — no few-shot, no original label. A fresh, independent tier judgment.
RUBRIC = """You are grading task difficulty for LLM routing. Assign ONE tier:
  haiku  — trivial: arithmetic, factual recall, single-fact lookup, short summary.
  sonnet — moderate: single-file code work, concept explanation, light design.
  opus   — hard: multi-component architecture, deep cross-file debugging, multi-step
           planning, synthesis under ambiguity.
Judge ONLY the difficulty the task demands. Output one JSON object on one line:
{"tier": "haiku" | "sonnet" | "opus"}. No prose."""


def judge_tier(client: OpenAI, prompt: str) -> str | None:
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[  # user-only roles: VibeProxy cloaks on a caller `system` role (BCJ-5)
            {"role": "user", "content": RUBRIC},
            {"role": "assistant", "content": "Understood. JSON only."},
            {"role": "user", "content": f"Task:\n{prompt}"},
        ],
        max_tokens=256,
    )
    raw = resp.choices[0].message.content or ""
    m = re.search(r"\{.*?\}", raw, re.S)
    if not m:
        return None
    try:
        t = json.loads(m.group(0)).get("tier")
        return t if t in ("haiku", "sonnet", "opus") else None
    except Exception:
        return None


def main() -> None:
    _, ev = train_eval_split(load_probes())
    judge = OpenAI(base_url=JUDGE_BASE, api_key=JUDGE_KEY)

    rows = []
    for r in ev:
        indep = judge_tier(judge, r["prompt"])
        pred = classify(r["prompt"]).tier  # shipped 4B few-shot
        rows.append({
            "prompt": r["prompt"][:60],
            "orig": r["expected_tier"],
            "indep": indep,
            "pred4b": pred,
        })

    n = len(rows)
    judged = [x for x in rows if x["indep"]]
    agree = sum(x["orig"] == x["indep"] for x in judged)
    consensus = [x for x in judged if x["orig"] == x["indep"]]
    disputed = [x for x in judged if x["orig"] != x["indep"]]

    pred_all = sum(x["pred4b"] == x["orig"] for x in rows)
    pred_consensus = sum(x["pred4b"] == x["orig"] for x in consensus)
    pred_disputed = sum(x["pred4b"] == x["orig"] for x in disputed)

    print(f"eval rows: {n} | judge parsed: {len(judged)}")
    print(f"original<->independent tier agreement: {agree}/{len(judged)} ({agree/len(judged):.0%})")
    print(f"4B tier acc — ALL: {pred_all}/{n} ({pred_all/n:.0%})")
    print(f"4B tier acc — CONSENSUS rows: {pred_consensus}/{len(consensus)} "
          f"({pred_consensus/max(len(consensus),1):.0%} of {len(consensus)})")
    print(f"4B tier acc — DISPUTED rows:  {pred_disputed}/{len(disputed)} "
          f"({pred_disputed/max(len(disputed),1):.0%} of {len(disputed)})")
    print("\nDISPUTED rows (orig != independent judge):")
    for x in disputed:
        print(f"  orig={x['orig']:6s} judge={x['indep']:6s} 4b={x['pred4b']:6s} | {x['prompt']}")


if __name__ == "__main__":
    main()
