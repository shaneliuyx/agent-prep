"""Build a LongMemEval slice for the W3.5.9 / W3.5.10 memory-backend eval.

Downloads ``longmemeval_oracle.json`` (cleaned 2025-09 release) from Hugging
Face and writes a deterministic slice. The ORIGINAL 2-axis slice (W3.5.8/9)
covered only ``multi-session`` + ``knowledge-update`` — both *current-value*
question shapes. To build + measure a UNIVERSAL memory solution we need all six
LongMemEval axes (single-session-user / -assistant / -preference,
multi-session, knowledge-update, temporal-reasoning) plus the abstention
overlay, so the eval can reward the read-time operators (as-of, count, Yes/No,
preference, abstain) a universal reader must handle.

Run from lab root::

    # default: 2-axis legacy slice (byte-for-byte reproduces w358)
    uv run python scripts/build_slice.py

    # all 6 axes, 4 questions each, new file (does NOT clobber w358)
    uv run python scripts/build_slice.py --all-axes --per-type 4 \
        --out data/longmemeval_slice_6axis.json

    # add the abstention overlay (tests the reader's "I don't know" path)
    uv run python scripts/build_slice.py --all-axes --per-type 4 \
        --include-abstention --abs-per-type 2 --out data/longmemeval_slice_6axis.json

Deterministic: first N per type in source order — re-runs reproduce byte-for-byte
(no random sampling).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time
import urllib.request
from collections import Counter

ORACLE_URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/"
    "resolve/main/longmemeval_oracle.json"
)

# The two legacy axes (default) vs the full six. Names match the oracle's
# `question_type` field exactly.
LEGACY_TYPES = ("multi-session", "knowledge-update")
ALL_AXES = (
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "knowledge-update",
    "temporal-reasoning",
)

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
ORACLE_PATH = DATA_DIR / "longmemeval_oracle.json"
LEGACY_SLICE = DATA_DIR / "longmemeval_slice_w358.json"


def ensure_oracle() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    if ORACLE_PATH.exists():
        print(f"cached  {ORACLE_PATH.name}  {ORACLE_PATH.stat().st_size/1e6:.1f}MB")
        return
    t0 = time.time()
    print(f"downloading {ORACLE_PATH.name} (one-time) ...")
    urllib.request.urlretrieve(ORACLE_URL, ORACLE_PATH)
    print(f"fetched {ORACLE_PATH.name}  {ORACLE_PATH.stat().st_size/1e6:.1f}MB in {time.time()-t0:.1f}s")


def _is_abs(q: dict) -> bool:
    """Abstention questions carry the same question_type but an `_abs`-suffixed id;
    their gold is an abstention answer (the fact is NOT in the haystack)."""
    return q["question_id"].endswith("_abs")


def build_slice(
    types: tuple[str, ...], per_type: int,
    include_abstention: bool, abs_per_type: int,
) -> list[dict]:
    """Pick the first `per_type` non-abstention questions per type (source order),
    then optionally the first `abs_per_type` abstention questions per type."""
    qs = json.loads(ORACLE_PATH.read_text())
    base_seen: Counter[str] = Counter()
    abs_seen: Counter[str] = Counter()
    picked: list[dict] = []

    # Pass 1 — base (non-abstention) questions.
    for q in qs:
        qt = q["question_type"]
        if qt in types and not _is_abs(q) and base_seen[qt] < per_type:
            picked.append(q)
            base_seen[qt] += 1

    # Pass 2 — abstention overlay (separate cap so it doesn't displace base).
    if include_abstention and abs_per_type > 0:
        for q in qs:
            qt = q["question_type"]
            if qt in types and _is_abs(q) and abs_seen[qt] < abs_per_type:
                picked.append(q)
                abs_seen[qt] += 1

    return picked


def report(picked: list[dict]) -> None:
    by_type = Counter(q["question_type"] for q in picked)
    abs_ct = sum(1 for q in picked if _is_abs(q))
    sess = [len(q["haystack_sessions"]) for q in picked]
    turns = [sum(len(s) for s in q["haystack_sessions"]) for q in picked]
    ans_types = Counter(type(q["answer"]).__name__ for q in picked)
    print(f"slice: {len(picked)} questions ({abs_ct} abstention)  by_type={dict(by_type)}")
    print(f"haystack sessions/q: min={min(sess)} median={sorted(sess)[len(sess)//2]} max={max(sess)}")
    print(f"haystack turns/q:    min={min(turns)} median={sorted(turns)[len(turns)//2]} max={max(turns)}")
    print(f"answer types: {dict(ans_types)} (judge casts to str)")
    # Per-type haystack size → runtime estimate (single-session is cheap, multi-* is dear).
    print("  per-type median haystack turns (runtime proxy):")
    for t in sorted(by_type):
        tt = [sum(len(s) for s in q["haystack_sessions"]) for q in picked if q["question_type"] == t]
        print(f"    {t:<28} n={len(tt):<3} median_turns={sorted(tt)[len(tt)//2]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-axes", action="store_true",
                    help="cover all 6 LongMemEval axes (default: 2 legacy axes)")
    ap.add_argument("--types", default="",
                    help="comma-separated question_type subset (overrides --all-axes)")
    ap.add_argument("--per-type", type=int, default=10,
                    help="base questions per type (default 10; use 3-4 for a fast first-pass)")
    ap.add_argument("--include-abstention", action="store_true",
                    help="also sample abstention (_abs) questions — tests the reader's abstain path")
    ap.add_argument("--abs-per-type", type=int, default=2,
                    help="abstention questions per type when --include-abstention")
    ap.add_argument("--out", default="",
                    help="output slice path (default: w358 for legacy, 6axis for --all-axes)")
    args = ap.parse_args()

    if args.types:
        types = tuple(t.strip() for t in args.types.split(",") if t.strip())
    elif args.all_axes:
        types = ALL_AXES
    else:
        types = LEGACY_TYPES

    if args.out:
        out_path = pathlib.Path(args.out)
        if not out_path.is_absolute():
            out_path = DATA_DIR.parent / out_path
    else:
        out_path = (DATA_DIR / "longmemeval_slice_6axis.json"
                    if types == ALL_AXES else LEGACY_SLICE)

    # Guard: never silently clobber the validated legacy slice with a different shape.
    if out_path == LEGACY_SLICE and types != LEGACY_TYPES:
        raise SystemExit(
            f"refusing to overwrite {LEGACY_SLICE.name} with a non-legacy slice — "
            f"pass --out data/longmemeval_slice_6axis.json")

    ensure_oracle()
    picked = build_slice(types, args.per_type, args.include_abstention, args.abs_per_type)
    out_path.write_text(json.dumps(picked, indent=2))
    print(f"wrote   {out_path.relative_to(DATA_DIR.parent)}  {out_path.stat().st_size/1e3:.1f}KB")
    report(picked)


if __name__ == "__main__":
    main()
