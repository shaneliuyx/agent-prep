"""Batch-rejudge unjudged cells in results_w358.jsonl.

The eval driver defers judging to `correct=None` when the VibeProxy-hosted judge
503s under load (so a scoring failure never crashes a 2-hr run). This script
rejudges those cells AFTER the run, once VibeProxy's auth is cool — judging is
seconds-cheap and redoable, unlike the imprints. The judge() call already has
503 retry/backoff (src/llm_retry), so run this when load has stopped.

Usage (from lab root):
    uv run python -m scripts.rejudge            # rejudge correct=None cells, write back
    uv run python -m scripts.rejudge --all      # re-judge EVERY ok cell (full re-score)
    uv run python -m scripts.rejudge --dry-run  # show what would change, write nothing
"""
from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv

load_dotenv()

from src.judge_sonnet import judge
from src.run_longmemeval_slice import ALL_BACKENDS, QUALITY_EXCLUDE, RESULTS_PATH


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="re-judge every ok cell, not just correct=None")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not RESULTS_PATH.exists():
        raise SystemExit(f"no results at {RESULTS_PATH}")
    records = [json.loads(ln) for ln in RESULTS_PATH.read_text().splitlines() if ln.strip()]

    rejudged = 0
    for rec in records:
        for backend in ALL_BACKENDS:
            cell = rec.get(backend)
            if not isinstance(cell, dict) or cell.get("status") != "ok":
                continue
            needs = cell.get("correct") is None if not args.all else True
            if not needs:
                continue
            verdict = judge(rec["question"], rec["gold"], cell.get("predicted", ""))
            print(f"  {rec['question_id']:>12} [{backend:11}] "
                  f"{str(cell.get('correct')):>5} -> {verdict['correct']}  {verdict['reason'][:70]}")
            if not args.dry_run:
                cell.update(verdict)
            rejudged += 1

    if not args.dry_run:
        RESULTS_PATH.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    # accuracy summary per backend (judged cells only)
    print(f"\n>>> rejudged {rejudged} cell(s){' (dry-run, not written)' if args.dry_run else ''}")
    excluded = [r["question_id"] for r in records if r["question_id"] in QUALITY_EXCLUDE]
    if excluded:
        print(f">>> EXCLUDED from accuracy (broken gold): {excluded}")
    print(">>> per-backend accuracy (judged cells, excluding broken-gold questions):")
    for backend in ALL_BACKENDS:
        cells = [r[backend] for r in records
                 if r["question_id"] not in QUALITY_EXCLUDE
                 and isinstance(r.get(backend), dict) and r[backend].get("correct") is not None]
        if cells:
            n_ok = sum(1 for c in cells if c.get("correct"))
            print(f"    {backend:12} {n_ok}/{len(cells)} = {100*n_ok/len(cells):.0f}%")


if __name__ == "__main__":
    main()
