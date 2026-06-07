"""verify_arch.py — does the recommended architecture actually hold on this corpus?

The "measured policy, three layers" design rests on ONE empirical claim: the cheap,
deterministic SELECTOR metric (discounted grounding@C) predicts ANSWER quality well
enough to pick the right arm without an LLM in the hot loop. This script tests that
claim directly — the CALIBRATOR step — and confirms the SELECTOR's pick is the
answer-quality winner (the GATE's job, run here as a one-off audit).

For every arm (keyword/vector/hybrid) × every tenk golden question (those carry
pass_criteria), it pairs:
  - the cheap metric : discounted grounding@C  (from results/arm_scores.json, zero-LLM)
  - the real objective: answer PASS/FAIL        (generate from that arm's top-C context,
                                                 judge against pass_criteria — strong model)
Then reports:
  1. correlation(grounding, answer-pass) across all arm×question cells  → does the cheap
     surrogate track the objective? (the architecture's load-bearing assumption)
  2. per-arm mean grounding vs answer pass-rate → does the SELECTOR's winner (max grounding)
     also win on answers? (selector validity)

Inputs: results/arm_scores.json (from `bun src/route_eval.ts`) + the W2.7 ground truth.
Run with a STRONG generator so the answer signal isn't generation-noise-limited:
  OPENROUTER_BASE_URL=http://localhost:8317/v1 OPENROUTER_API_KEY=vibeproxy \
  CHAT_MODEL=claude-opus-4-5-20251101 uv run python src/verify_arch.py
"""
from __future__ import annotations

import json
import pathlib
import statistics

from answer_route_ab import LLMUnavailable, _answer, _judge, _pass_criteria_by_q
from ground_truth_ab import _client

_ARMS = ("keyword", "vector", "hybrid")
_ARM_SCORES = pathlib.Path(__file__).resolve().parent.parent / "results" / "arm_scores.json"


def _correlation(xs: list[float], ys: list[float]) -> float | None:
    """Pearson r; None if either series is constant (correlation undefined)."""
    if len(xs) < 2 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    return statistics.correlation(xs, ys)


def main() -> None:
    dump = json.loads(_ARM_SCORES.read_text())
    criteria_by_q = _pass_criteria_by_q()
    client = _client()

    g_all: list[float] = []        # grounding per arm×question cell
    p_all: list[float] = []        # answer pass (0/1) per cell
    per_arm: dict[str, dict[str, list[float]]] = {a: {"g": [], "p": []} for a in _ARMS}

    for item in dump:
        q = item["q"].strip()
        criteria = criteria_by_q.get(q)
        if criteria is None:
            continue  # entity questions have no rubric — skip
        for arm in _ARMS:
            grounding = float(item["grounding"][arm])
            try:
                passed = _judge(client, _answer(client, item["slugs"][arm], q), criteria)
            except LLMUnavailable as exc:
                print(f"  [skip] {arm:7s} {q[:44]} ({exc})")
                continue
            p = 1.0 if passed else 0.0
            g_all.append(grounding)
            p_all.append(p)
            per_arm[arm]["g"].append(grounding)
            per_arm[arm]["p"].append(p)
            print(f"  {arm:7s} grnd={grounding:.3f} ans={'P' if passed else 'F'}  {q[:44]}")

    n = len(g_all)
    corr = _correlation(g_all, p_all)
    print(f"\n=== CALIBRATOR — does the cheap metric predict answer quality? ===")
    print(f"  cells judged: {n}")
    print(f"  corr(discounted grounding@C, answer-pass) = "
          f"{'n/a' if corr is None else f'{corr:+.3f}'}")

    print(f"\n=== SELECTOR — is the max-grounding arm the answer-quality winner? ===")
    print(f"  {'arm':8s} {'mean grnd':>10s} {'ans pass':>9s}")
    arm_grnd = {a: statistics.fmean(per_arm[a]["g"]) if per_arm[a]["g"] else 0.0 for a in _ARMS}
    arm_pass = {a: statistics.fmean(per_arm[a]["p"]) if per_arm[a]["p"] else 0.0 for a in _ARMS}
    for a in _ARMS:
        print(f"  {a:8s} {arm_grnd[a]:10.3f} {arm_pass[a]:9.3f}")
    g_winner = max(_ARMS, key=lambda a: arm_grnd[a])
    p_winner = max(_ARMS, key=lambda a: arm_pass[a])
    verdict = "CONFIRMED" if g_winner == p_winner else "MISMATCH"
    print(f"\n  selector pick (max grounding) = {g_winner};  answer winner = {p_winner}"
          f"  → {verdict}")


if __name__ == "__main__":
    main()
