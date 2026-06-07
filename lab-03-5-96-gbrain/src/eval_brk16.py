"""eval_brk16.py — run GBrain on ALL 16 W2.7 questions; answer pass-rate vs `pass_criteria`.

The full W2.7 eval (12 in-document + 4 out-of-document refusals), retrieved through GBrain's
hybrid arm (C=3, full bodies), answered + judged against the same rubric W2.7 used. Run it
BEFORE and AFTER the PageIndex-structure ingest enrichment to see whether the structural
"where-is-X" questions (esp. the Notes question) start passing — without regressing the rest.

Reuses shared/ (per AGENTS.md): llm (resolve/chat/judge/load_pass_criteria) + gbrain_cli
(gbrain_query_slugs/build_context). Generator + judge default to Opus so a failure is a
retrieval-representation gap, not a weak-generator artifact.

Run (services: gbrain-pg, oMLX :8000 for query embed, VibeProxy :8317 for gen/judge):
  GEN=opus JUDGE=opus OPENROUTER_BASE_URL=http://localhost:8317/v1 OPENROUTER_API_KEY=vibeproxy \
  uv run python src/eval_brk16.py
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/Users/yuxinliu/code/agent-prep/shared")
from gbrain_cli import build_context  # noqa: E402
from llm import LLMUnavailable, chat, judge, load_pass_criteria, resilient, resolve  # noqa: E402

_GT = "/Users/yuxinliu/code/agent-prep/lab-02-7-pageindex/data/eval_ground_truth.json"
_PROMPT = ("Using ONLY the notes below, answer the question. If a fact is not in the notes, "
           "say 'insufficient context'.\n\nNOTES:\n{ctx}\n\nQUESTION: {q}")
_C = 3


def _slugs(q: str, limit: int) -> list[str]:
    # local import so a missing gbrain CLI surfaces clearly
    from gbrain_cli import gbrain_query_slugs
    return gbrain_query_slugs(q, limit)


def main() -> None:
    criteria_by_q = load_pass_criteria(_GT)
    gen_c, gen_m, gen_n = resolve("GEN", "opus")
    judge_c, judge_m, judge_n = resolve("JUDGE", "opus")
    print(f"eval_brk16: {len(criteria_by_q)} questions · gen={gen_n} · judge={judge_n} · C={_C}\n")

    rows: list[tuple[str, bool]] = []
    for q, criteria in criteria_by_q.items():
        slugs = _slugs(q, _C)
        try:
            ctx = build_context(slugs) if slugs else ""
            answer = resilient(chat, gen_c, _PROMPT.format(ctx=ctx, q=q), gen_m)
            ok = judge(judge_c, answer, criteria, judge_m)
        except LLMUnavailable as exc:
            print(f"  [skip] {q[:56]} ({exc})")
            continue
        rows.append((q, ok))
        print(f"  {'P' if ok else 'F'}  {q[:66]}")

    n = len(rows)
    passed = sum(1 for _, ok in rows if ok)
    print(f"\ngt_pass: {passed}/{n} = {passed / n:.3f}" if n else "\nno rows")
    notes = [ok for q, ok in rows if "Notes to Consolidated" in q]
    if notes:
        print(f"structural 'Notes located?' question: {'PASS' if notes[0] else 'FAIL'}")


if __name__ == "__main__":
    main()
