"""answer_route_ab.py — does per-query ROUTING beat global hybrid at ANSWER time?

route_eval.ts already showed routing has ZERO *grounding* headroom on this corpus
(hybrid weakly dominates per-query). This script tests the one effect grounding can't
see: discounted grounding takes the *max* coverage over the top-C, so it ignores
whether the OTHER chunks in the budget are distractors. Two contexts with the same
best chunk score identically — but the generator reads all C, and hybrid's fused set
may carry distractors a clean single-arm set avoids. So we judge the ANSWERS.

For each tenk golden question (those carry `pass_criteria` in eval_ground_truth.json):
  - build the GLOBAL-hybrid context (top-C pages) and the ROUTED best-arm context,
  - generate an answer from each (temp 0, VibeProxy → Claude),
  - LLM-judge each PASS/FAIL against the question's pass_criteria,
  - compare pass rates. Questions where routed_slugs == global_slugs are identical by
    construction (reported separately — they can't show a difference).

Inputs: results/route_slugs.json (from `bun src/route_eval.ts`) + the W2.7 ground truth.
Run: uv run python src/answer_route_ab.py
"""
from __future__ import annotations

import json
import os
import pathlib
import time

from openai import APIConnectionError, APIError

from ground_truth_ab import _MEMZERO_TMPL, _ask, _client, gbrain_get


class LLMUnavailable(RuntimeError):
    """The chat endpoint refused after retries — skip this question, don't crash the run."""


def _resilient(fn, *args, retries: int = 4, backoff: float = 2.0):
    """Retry an LLM call through transient connection drops (the local 14B MLX chat
    server falls over under memory pressure). Raise LLMUnavailable if it never recovers."""
    for attempt in range(retries):
        try:
            return fn(*args)
        except (APIConnectionError, APIError) as exc:
            if attempt == retries - 1:
                raise LLMUnavailable(str(exc)) from exc
            time.sleep(backoff * (attempt + 1))
    raise LLMUnavailable("unreachable")

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_SLUGS = _ROOT / "results" / "route_slugs.json"
_GROUND_TRUTH = pathlib.Path(
    "~/code/agent-prep/lab-02-7-pageindex/data/eval_ground_truth.json").expanduser()

_JUDGE_TMPL = (
    "You are grading a candidate answer against a rubric. Reply with EXACTLY one word: "
    "PASS or FAIL.\n\nRUBRIC (what makes the answer correct):\n{criteria}\n\n"
    "CANDIDATE ANSWER:\n{answer}\n\nVerdict (PASS or FAIL):"
)


def _pass_criteria_by_q() -> dict[str, str]:
    """Map question text → pass_criteria from the W2.7 ground-truth (tenk questions)."""
    raw = json.loads(_GROUND_TRUTH.read_text())
    out: dict[str, str] = {}
    for key, entry in raw.items():
        if key.startswith("_") or not isinstance(entry, dict):
            continue
        if entry.get("q") and entry.get("pass_criteria"):
            out[entry["q"].strip()] = entry["pass_criteria"]
    return out


class SnippetRegression(RuntimeError):
    """A reader injected a truncated `query --json` snippet instead of a full
    `gbrain get` body — the failure mode the lab fixed once and must not regress to."""


_MIN_BODY_CHARS = 80   # a `gbrain get` page body; a `query --json` snippet is a short fragment
# Optional per-body char cap. Default 0 = full bodies (unchanged). Set MAX_BODY_CHARS to
# fit a small-context generator (e.g. the local 14B chokes on ~70K-token 10-K sections).
_MAX_BODY_CHARS = int(os.getenv("MAX_BODY_CHARS", "0"))
_body_check_logged = False


def build_context(slugs: list[str]) -> str:
    """Assemble reader context from FULL `gbrain get` page bodies — never the truncated
    `query --json` snippet (which lacks the grounding the generator needs; ground_truth_ab
    gotcha #1). Guards the seam: logs body sizes once so a regression is visible, and raises
    SnippetRegression (fail loud) if any injected body is suspiciously short. MAX_BODY_CHARS
    (opt-in) caps each body for small-context generators."""
    global _body_check_logged
    bodies = [gbrain_get(s) for s in slugs]
    if _MAX_BODY_CHARS:
        bodies = [b[:_MAX_BODY_CHARS] for b in bodies]
    sizes = [len(b.strip()) for b in bodies]
    if not _body_check_logged:
        print(f"  [reader] full-body check: {len(slugs)} slugs, body chars={sizes} "
              f"(via `gbrain get`, not `query --json` snippets)")
        _body_check_logged = True
    short = [(s, n) for s, n in zip(slugs, sizes) if n < _MIN_BODY_CHARS]
    if short:
        raise SnippetRegression(
            f"injected body too short {short} — reader must pull full `gbrain get` bodies, "
            f"not `query --json` snippets")
    return "\n\n".join(bodies)


def _answer(client, slugs: list[str], q: str) -> str:
    ctx = build_context(slugs)
    msg = [{"role": "user", "content": _MEMZERO_TMPL.format(ctx=ctx, q=q)}]
    ans, _ = _resilient(_ask, client, msg)
    return ans


def _judge(client, answer: str, criteria: str) -> bool:
    msg = [{"role": "user", "content": _JUDGE_TMPL.format(criteria=criteria, answer=answer)}]
    verdict, _ = _resilient(_ask, client, msg)
    return verdict.strip().upper().startswith("PASS")


def main() -> None:
    dump = json.loads(_SLUGS.read_text())
    criteria_by_q = _pass_criteria_by_q()
    client = _client()

    rows: list[tuple[str, bool, bool, bool]] = []  # (q, differs, global_pass, routed_pass)
    for item in dump:
        q = item["q"].strip()
        criteria = criteria_by_q.get(q)
        if criteria is None:
            continue  # entity questions have no pass_criteria — skip (judge needs a rubric)
        differs = item["routed_slugs"] != item["global_slugs"]
        try:
            g_pass = _judge(client, _answer(client, item["global_slugs"], q), criteria)
            r_pass = _judge(client, _answer(client, item["routed_slugs"], q), criteria)
        except LLMUnavailable as exc:
            print(f"  [skip] chat endpoint down — {q[:52]} ({exc})")
            continue
        rows.append((q, differs, g_pass, r_pass))
        tag = "DIFF" if differs else "same"
        print(f"  [{tag}] global={'P' if g_pass else 'F'} routed={'P' if r_pass else 'F'}  {q[:52]}")

    n = len(rows)
    g_rate = sum(g for _, _, g, _ in rows) / n if n else 0.0
    r_rate = sum(r for _, _, _, r in rows) / n if n else 0.0
    diff = [row for row in rows if row[1]]
    print(f"\njudged {n} tenk questions ({len(diff)} have a routed≠global context)")
    print(f"  global (hybrid) pass rate : {g_rate:.3f}")
    print(f"  routed (per-query) pass   : {r_rate:.3f}   Δ {r_rate - g_rate:+.3f}")
    if diff:
        dg = sum(g for _, _, g, _ in diff) / len(diff)
        dr = sum(r for _, _, _, r in diff) / len(diff)
        print(f"  on the {len(diff)} DIFF-context questions: global {dg:.3f} → routed {dr:.3f}  Δ {dr - dg:+.3f}")


if __name__ == "__main__":
    main()
