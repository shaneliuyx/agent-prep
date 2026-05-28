"""§8.7.2 — Build the LongMemEval slice used by W3.5.8's EverCore-vs-Qdrant
cross-validation eval.

Downloads ``longmemeval_oracle.json`` (cleaned 2025-09 release) from
Hugging Face, filters to two question_types (``multi-session`` +
``knowledge-update``), takes the first N per type, and writes the slice
to ``data/longmemeval_slice_w358.json``.

Run from lab root::

    uv run python scripts/build_slice.py

The script is idempotent: re-running skips the download if the oracle
file is already present, and overwrites the slice file with the same
deterministic selection (no random sampling — the first N in source
order, so re-runs reproduce byte-for-byte).
"""
from __future__ import annotations

import json
import pathlib
import time
import urllib.request
from collections import Counter

ORACLE_URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/"
    "resolve/main/longmemeval_oracle.json"
)
TARGET_TYPES = ("multi-session", "knowledge-update")
PER_TYPE = 10

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
ORACLE_PATH = DATA_DIR / "longmemeval_oracle.json"
SLICE_PATH = DATA_DIR / "longmemeval_slice_w358.json"


def ensure_oracle() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    if ORACLE_PATH.exists():
        print(f"cached  {ORACLE_PATH.name}  {ORACLE_PATH.stat().st_size/1e6:.1f}MB")
        return
    t0 = time.time()
    urllib.request.urlretrieve(ORACLE_URL, ORACLE_PATH)
    print(f"fetched {ORACLE_PATH.name}  {ORACLE_PATH.stat().st_size/1e6:.1f}MB in {time.time()-t0:.1f}s")


def build_slice() -> list[dict]:
    qs = json.loads(ORACLE_PATH.read_text())
    seen: Counter[str] = Counter()
    picked: list[dict] = []
    for q in qs:
        qt = q["question_type"]
        if qt in TARGET_TYPES and seen[qt] < PER_TYPE and not q["question_id"].endswith("_abs"):
            picked.append(q)
            seen[qt] += 1
        if len(picked) == PER_TYPE * len(TARGET_TYPES):
            break
    return picked


def report(picked: list[dict]) -> None:
    sess = [len(q["haystack_sessions"]) for q in picked]
    turns = [sum(len(s) for s in q["haystack_sessions"]) for q in picked]
    ans_types = Counter(type(q["answer"]).__name__ for q in picked)
    print(f"slice: {len(picked)} questions  "
          f"({dict(Counter(q['question_type'] for q in picked))})")
    print(f"haystack sessions/q: min={min(sess)} median={sorted(sess)[len(sess)//2]} max={max(sess)}")
    print(f"haystack turns/q:    min={min(turns)} median={sorted(turns)[len(turns)//2]} max={max(turns)}")
    print(f"answer types in slice: {dict(ans_types)} (judge must cast to str)")


def main() -> None:
    ensure_oracle()
    picked = build_slice()
    SLICE_PATH.write_text(json.dumps(picked, indent=2))
    print(f"wrote   {SLICE_PATH.name}  {SLICE_PATH.stat().st_size/1e3:.1f}KB")
    report(picked)


if __name__ == "__main__":
    main()
