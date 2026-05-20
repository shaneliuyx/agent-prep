"""Re-judge saved LongMemEval answer-sets with a SINGLE fixed judge model.

The eval runner defaults MODEL_JUDGE = compose model, so cross-model
accuracy numbers are confounded with judge strictness — a stricter judge
marks its own model's answers down harder. This pass re-runs ONLY the
judge over stored (question, gold, agent_answer) triples with one fixed
judge model, producing a clean judge-controlled comparison. No re-compose.

Run:
    COMPOSE_BASE_URL=http://localhost:8317/v1 COMPOSE_API_KEY=dummy \\
    MODEL_JUDGE=claude-opus-4-7 DISABLE_TEMPERATURE=1 \\
      uv run python scripts/rejudge.py results/a.json results/b.json ...
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI

from scripts.run_longmemeval_oracle import JUDGE_PROMPT, parse_verdict

# Opus 4.7 thinking deprecates `temperature`; gate as in the runner.
_TEMP_KW: dict = {} if os.getenv("DISABLE_TEMPERATURE") == "1" else {"temperature": 0.0}


def rejudge_file(path: str, llm: OpenAI, judge_model: str) -> dict:
    """Re-judge every answerable question in one results JSON.

    Errored questions (no agent_answer) cannot be judged — they are
    counted but stay non-correct. Writes a sibling *_rejudged.json with
    the per-question rejudge_verdict for later per-Q analysis.
    """
    data = json.loads(Path(path).read_text())
    rows = data.get("per_question", [])
    n_total = len(rows)
    correct = 0
    errors = 0
    for i, r in enumerate(rows, 1):
        if r.get("error") or not r.get("agent_answer"):
            errors += 1
            r["rejudge_verdict"] = "ERROR"
            continue
        resp = llm.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                question=r["question"], gold=r["gold"], answer=r["agent_answer"])}],
            max_tokens=400,
            **_TEMP_KW,
        )
        verdict = parse_verdict((resp.choices[0].message.content or "").strip())
        r["rejudge_verdict"] = verdict
        if verdict == "CORRECT":
            correct += 1
        if i % 20 == 0:
            print(f"    {i}/{n_total}", flush=True)

    judged = n_total - errors
    out_path = Path(path).with_name(Path(path).stem + "_rejudged.json")
    out_path.write_text(json.dumps(data, indent=2))

    return {
        "path": str(path),
        "n_total": n_total,
        "judged": judged,
        "errors": errors,
        "correct": correct,
        "acc_raw": round(correct / n_total, 3) if n_total else 0.0,
        "acc_excl_errors": round(correct / judged, 3) if judged else 0.0,
    }


def main() -> None:
    paths = sys.argv[1:]
    if not paths:
        print("usage: rejudge.py <results.json> [<results.json> ...]")
        sys.exit(1)

    llm = OpenAI(
        base_url=os.getenv("COMPOSE_BASE_URL") or os.getenv("OMLX_BASE_URL"),
        api_key=os.getenv("COMPOSE_API_KEY") or os.getenv("OMLX_API_KEY"),
        timeout=300.0,
        max_retries=10,
    )
    judge_model = os.getenv("MODEL_JUDGE", "claude-opus-4-7")
    print(f"Fixed judge: {judge_model}")
    print(f"Judge endpoint: {os.getenv('COMPOSE_BASE_URL') or os.getenv('OMLX_BASE_URL')}\n")

    results = []
    for p in paths:
        print(f"re-judging {p} ...", flush=True)
        t0 = time.perf_counter()
        res = rejudge_file(p, llm, judge_model)
        res["wall_s"] = round(time.perf_counter() - t0, 1)
        results.append(res)
        print(f"  → {res['correct']}/{res['n_total']} = {res['acc_raw']*100:.0f}% "
              f"raw | {res['acc_excl_errors']*100:.0f}% excl-errors "
              f"(judged {res['judged']}, errors {res['errors']}, {res['wall_s']}s)\n")

    report = {"fixed_judge": judge_model, "files": results}
    out = Path("results/rejudge_report.json")
    out.write_text(json.dumps(report, indent=2))
    print(f"Wrote {out}")
    print(f"\n{'='*70}\nJUDGE-CONTROLLED COMPARISON (fixed judge: {judge_model})\n{'='*70}")
    for r in results:
        name = Path(r["path"]).stem
        print(f"  {name:<46} {r['acc_raw']*100:5.1f}%  "
              f"({r['correct']}/{r['n_total']}, {r['errors']} err)")


if __name__ == "__main__":
    main()
