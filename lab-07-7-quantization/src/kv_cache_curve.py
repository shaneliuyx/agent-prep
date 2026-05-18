"""Phase 3 — KV-cache memory + per-token latency vs sequence length.

Formula: $M_{\\text{KV}} = 2 \\cdot n_{\\text{layers}} \\cdot n_{\\text{heads}} \\cdot d_{\\text{head}} \\cdot L_{\\text{seq}} \\cdot b_{\\text{KV}}$

Generates 200 tokens at each seq_len in {512, 1k, 2k, 4k, 8k}. Records
peak RSS + tokens/sec. Compare measured vs theoretical KV memory.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import psutil
from mlx_lm import load, generate


def measure_at_seq_len(model_id: str, seq_len: int, n_generate: int = 200) -> dict:
    print(f"\n=== {model_id} @ seq_len={seq_len} ===")
    loaded = load(model_id)
    model, tokenizer = loaded[0], loaded[1]

    # Build a prompt of approximately seq_len tokens by repeating a base sentence
    base = "The model processes this sentence carefully. "
    base_ids = tokenizer.encode(base)
    base_tokens = len(list(base_ids))
    repeats = max(1, seq_len // base_tokens)
    prompt = base * repeats
    actual_seq_len = len(list(tokenizer.encode(prompt)))

    rss_before = psutil.Process(os.getpid()).memory_info().rss / 1e9

    t0 = time.perf_counter()
    _ = generate(model, tokenizer, prompt=prompt, max_tokens=n_generate, verbose=False)
    wall = time.perf_counter() - t0

    rss_after = psutil.Process(os.getpid()).memory_info().rss / 1e9
    tokens_per_sec = n_generate / wall if wall > 0 else 0

    return {
        "model": model_id,
        "target_seq_len": seq_len,
        "actual_seq_len": actual_seq_len,
        "generated_tokens": n_generate,
        "wall_seconds": wall,
        "tokens_per_sec": tokens_per_sec,
        "rss_delta_gb": rss_after - rss_before,
    }


def theoretical_kv_gb(n_layers: int, n_heads: int, d_head: int, seq_len: int,
                     bytes_per_kv: int = 2) -> float:
    """M_KV = 2 * layers * heads * d_head * seq_len * bytes_per_kv"""
    return 2 * n_layers * n_heads * d_head * seq_len * bytes_per_kv / 1e9


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-oss-20b-MXFP4-Q8")
    ap.add_argument("--seq-lens", default="512,1024,2048,4096",
                    help="Comma-separated sequence lengths")
    ap.add_argument("--n-generate", type=int, default=200)
    args = ap.parse_args()

    seq_lens = [int(s) for s in args.seq_lens.split(",")]
    results = [measure_at_seq_len(args.model, n, args.n_generate) for n in seq_lens]

    # gpt-oss-20b: n_layers=32, n_heads=32, d_head=64
    for r in results:
        r["theoretical_kv_gb"] = theoretical_kv_gb(32, 32, 64, r["actual_seq_len"])

    out_path = Path(__file__).resolve().parent.parent / "results" / "phase3_kv_cache.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")
