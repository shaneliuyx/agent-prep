"""Empirical probe: does BGE-M3 sparse head emit NaN on MPS in fp16?

The autoconfig policy `fp16_safe_for_encoder=(device == "cuda")` is based on
upstream NaN reports for BGE-M3's sparse head. This script verifies (or
falsifies) the claim against the current FlagEmbedding + torch + macOS combo.

Method:
  1. Load corpus (jsonl or json array; with/without per-item chunking).
  2. Sample N items, stratify by char-length decile so length-correlated
     failure modes surface separately.
  3. Encode under fp16=False (baseline) and fp16=True (suspect) at the
     autoconfig-recommended batch size for this host.
  4. Per decile: count nan_dense / zero_norm_dense / empty_sparse / nan_sparse.
  5. Report per-decile + total. Decide whether the conservative default
     should flip.

Usage:
  ./.venv/bin/python shared/tests/probe_bge_m3_fp16_mps.py \\
      --corpus-path lab-02-rerank-compress/data/docs.jsonl \\
      --jsonl --n 5000

  ./.venv/bin/python shared/tests/probe_bge_m3_fp16_mps.py \\
      --corpus-path lab-02-5-graphrag/data/corpus.json \\
      --chunk --n 5000

Notes:
  - --jsonl: input is one JSON object per line (W2 docs.jsonl)
  - --chunk: input is a JSON array of articles; chunk each text via
    char_window_chunks(512, 64) before sampling (W2.5 corpus.json)
  - --batch defaults to autoconfig.recommend_encode_batch(probe) so the
    probe matches what production-shape ingest would use.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "shared"))

from rag_hybrid import autoconfig  # noqa: E402
from rag_hybrid.chunking import char_window_chunks  # noqa: E402


def _has_nan(arr) -> bool:
    """Numpy-fast NaN check; falls back to math.isnan for plain lists."""
    try:
        import numpy as np
        return bool(np.isnan(arr).any())
    except Exception:
        return any(math.isnan(float(v)) for v in arr)


def load_corpus(path: Path, jsonl: bool, chunk: bool, text_field: str) -> list[str]:
    """Returns a flat list of text strings."""
    if jsonl:
        items = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        items = json.loads(path.read_text())

    if not chunk:
        return [it[text_field] for it in items]

    out: list[str] = []
    for it in items:
        for c in char_window_chunks(it[text_field], 512, 64):
            out.append(c)
    return out


def encode_and_count(use_fp16: bool, texts: list[str], batch_size: int) -> dict:
    """Load model with given fp16 setting, encode at given batch size, count corruption."""
    from FlagEmbedding import BGEM3FlagModel

    model = BGEM3FlagModel(
        f"{Path.home()}/models/bge-m3",
        use_fp16=use_fp16,
        device="mps",
    )
    out = model.encode(
        texts,
        batch_size=batch_size,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    dense_vecs = out["dense_vecs"]
    sparse_dicts = out["lexical_weights"]
    n = len(texts)

    # Decile bucketing by char length
    lengths = [len(t) for t in texts]
    sorted_idx = sorted(range(n), key=lambda i: lengths[i])
    decile_of: dict[int, int] = {}
    for rank, idx in enumerate(sorted_idx):
        decile_of[idx] = min(9, rank * 10 // n)

    counts = {
        "nan_dense": 0, "zero_norm_dense": 0, "empty_sparse": 0, "nan_sparse": 0,
        "total": n,
        "by_decile": {d: {"n": 0, "nan_dense": 0, "zero_norm_dense": 0,
                          "empty_sparse": 0, "nan_sparse": 0,
                          "min_len": 10**9, "max_len": 0}
                      for d in range(10)},
    }
    for i in range(n):
        d = decile_of[i]
        b = counts["by_decile"][d]
        b["n"] += 1
        b["min_len"] = min(b["min_len"], lengths[i])
        b["max_len"] = max(b["max_len"], lengths[i])

        dv = dense_vecs[i]
        if _has_nan(dv):
            counts["nan_dense"] += 1; b["nan_dense"] += 1
        norm = float((dv * dv).sum()) ** 0.5
        if norm < 1e-6:
            counts["zero_norm_dense"] += 1; b["zero_norm_dense"] += 1

        sd = sparse_dicts[i]
        if not sd or len(sd) == 0:
            counts["empty_sparse"] += 1; b["empty_sparse"] += 1
        elif any(math.isnan(float(v)) for v in sd.values()):
            counts["nan_sparse"] += 1; b["nan_sparse"] += 1

    return counts


def report_deciles(label: str, counts: dict) -> None:
    print(f"\n--- {label} per-decile ---")
    print(f"{'dec':>3} {'n':>5} {'len_range':>13} {'nan_d':>6} {'zN_d':>5} {'empty_s':>7} {'nan_s':>6}")
    for d in range(10):
        b = counts["by_decile"][d]
        if b["n"] == 0:
            continue
        rng = f"{b['min_len']}-{b['max_len']}"
        print(f"{d:>3} {b['n']:>5} {rng:>13} {b['nan_dense']:>6} {b['zero_norm_dense']:>5} {b['empty_sparse']:>7} {b['nan_sparse']:>6}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-path", required=True)
    ap.add_argument("--jsonl", action="store_true", help="treat input as one JSON per line")
    ap.add_argument("--chunk", action="store_true",
                    help="chunk each item via char_window_chunks(512,64)")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=None,
                    help="encode batch size; default = autoconfig.recommend_encode_batch")
    args = ap.parse_args()

    probe = autoconfig.probe_system()
    bs = args.batch or autoconfig.recommend_encode_batch(probe)

    corpus_path = Path(args.corpus_path)
    if not corpus_path.is_absolute():
        corpus_path = REPO / corpus_path
    if not corpus_path.exists():
        print(f"corpus not found: {corpus_path}", file=sys.stderr)
        return 1

    texts = load_corpus(corpus_path, args.jsonl, args.chunk, args.text_field)
    if len(texts) > args.n:
        # Even sampling across length distribution: sort + take every K-th
        idx = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        step = max(1, len(idx) // args.n)
        sampled_idx = idx[::step][: args.n]
        texts = [texts[i] for i in sampled_idx]
    print(f"corpus={corpus_path}  jsonl={args.jsonl}  chunk={args.chunk}")
    print(f"sampled {len(texts)} passages, batch={bs}, device=mps")
    print(f"min_len={min(len(t) for t in texts)} max_len={max(len(t) for t in texts)}")

    print("\n=== run with use_fp16=False (baseline) ===")
    fp32 = encode_and_count(False, texts, bs)
    print(f"totals: nan_dense={fp32['nan_dense']} zero_norm_dense={fp32['zero_norm_dense']} "
          f"empty_sparse={fp32['empty_sparse']} nan_sparse={fp32['nan_sparse']}")
    report_deciles("fp32", fp32)

    print("\n=== run with use_fp16=True (the suspect) ===")
    fp16 = encode_and_count(True, texts, bs)
    print(f"totals: nan_dense={fp16['nan_dense']} zero_norm_dense={fp16['zero_norm_dense']} "
          f"empty_sparse={fp16['empty_sparse']} nan_sparse={fp16['nan_sparse']}")
    report_deciles("fp16", fp16)

    print("\n=== delta (fp16 - fp32; positive = fp16-only corruption) ===")
    fp16_extra = 0
    for k in ("nan_dense", "zero_norm_dense", "empty_sparse", "nan_sparse"):
        delta = fp16[k] - fp32[k]
        fp16_extra += max(0, delta)
        print(f"  {k}: fp16={fp16[k]:>5d}  fp32={fp32[k]:>5d}  delta={delta:+d}")

    if fp16_extra == 0:
        print(f"\nRESULT: 0 fp16-induced corruption across {len(texts)} passages on MPS.")
    else:
        pct = 100.0 * fp16_extra / len(texts)
        print(f"\nRESULT: fp16 introduced {fp16_extra}/{len(texts)} corrupted passages ({pct:.2f}%).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
