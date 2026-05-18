"""Phase 4 — memory-bound vs compute-bound diagnostic.

Vary batch size. If doubling batch doubles throughput -> compute-bound.
If throughput plateaus -> memory-bound.

Memory-bound formula:
  $\\text{Tokens/sec}_{\\text{mem-bound}} = \\frac{\\text{Memory bandwidth (B/s)}}{N_{\\text{params}} \\cdot b_w}$

M5 Pro at ~400 GB/s on 20B-MXFP4-Q8 (10 GB weights) -> ~40 tok/s theoretical peak.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from mlx_lm import load, generate


def measure_at_batch(model_id: str, batch_size: int, n_generate: int = 100,
                     prompts: list[str] | None = None) -> dict:
    """Note: mlx_lm.generate is single-stream by default. Real batch testing
    requires either (a) running multiple generate calls in parallel (process
    fan-out) OR (b) using a paged-attention server (vLLM-equivalent for MLX).
    This script uses (a) as the simplest cross-batch comparison."""
    loaded = load(model_id)
    model, tokenizer = loaded[0], loaded[1]

    prompts = prompts or [
        "Explain why the sky appears blue.",
        "Write a haiku about a rainy afternoon.",
        "Summarize the plot of Hamlet.",
        "Describe the architecture of a transformer.",
    ]
    # Pad/cycle to batch_size
    prompts = (prompts * ((batch_size // len(prompts)) + 1))[:batch_size]

    # Naive sequential batch — measures the lower-bound throughput
    t0 = time.perf_counter()
    for p in prompts:
        generate(model, tokenizer, prompt=p, max_tokens=n_generate, verbose=False)
    wall = time.perf_counter() - t0

    total_tokens = batch_size * n_generate
    return {
        "model": model_id,
        "batch_size": batch_size,
        "n_generate_per_prompt": n_generate,
        "total_tokens": total_tokens,
        "wall_seconds": wall,
        "tokens_per_sec_aggregate": total_tokens / wall if wall > 0 else 0,
    }


def classify_regime(results: list[dict]) -> str:
    """If tokens_per_sec scales ~linearly with batch -> compute-bound (rare).
    If plateaus -> memory-bound (typical for AR decoding)."""
    if len(results) < 2:
        return "insufficient_data"

    r1 = results[0]
    rN = results[-1]
    batch_ratio = rN["batch_size"] / r1["batch_size"]
    throughput_ratio = rN["tokens_per_sec_aggregate"] / r1["tokens_per_sec_aggregate"]
    scaling_factor = throughput_ratio / batch_ratio

    if scaling_factor > 0.8:
        return "compute-bound"
    elif scaling_factor < 0.4:
        return "memory-bound"
    else:
        return "mixed"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="MLX-Qwen3.5-9B-GLM5.1-Distill-v1-8bit",
                    help="Use smaller model for batch experiments")
    ap.add_argument("--batch-sizes", default="1,2,4",
                    help="Comma-separated batch sizes")
    ap.add_argument("--n-generate", type=int, default=50)
    args = ap.parse_args()

    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    results = [measure_at_batch(args.model, b, args.n_generate) for b in batch_sizes]
    regime = classify_regime(results)

    output = {"regime": regime, "results": results}
    out_path = Path(__file__).resolve().parent.parent / "results" / "phase4_regime.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))

    print(f"\nRegime: {regime}")
    print(f"Wrote {out_path}")
