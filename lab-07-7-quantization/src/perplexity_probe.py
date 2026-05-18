"""Phase 2 — perplexity probe across quantization levels.

For each model in the fleet, computes mean perplexity over a 100-sample
probe (mixed prose + code + Chinese). Chart vs quant level reveals
accuracy degradation curve.

Formula: $\\text{PPL}(X) = \\exp\\!\\left(-\\frac{1}{N} \\sum_i \\log P(x_i \\mid x_{<i})\\right)$
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mlx.core as mx
from mlx_lm import load


PROBE_SAMPLES: list[str] = [
    # Prose (33)
    "The cat sat on the mat watching the rain through the kitchen window.",
    "Machine learning systems require careful engineering of the training data pipeline.",
    # ... 31 more (load from data/probe_set.json in full implementation)
    # Code (33)
    "def quicksort(arr): return arr if len(arr) <= 1 else quicksort([x for x in arr[1:] if x < arr[0]]) + [arr[0]] + quicksort([x for x in arr[1:] if x >= arr[0]])",
    # ... 32 more
    # Chinese (33)
    "今天天气很好，适合出门散步。",
    # ... 32 more
]


def perplexity_one(model, tokenizer, text: str) -> float:
    """Compute perplexity for one text sample using teacher-forcing."""
    ids = mx.array(tokenizer.encode(text))
    if len(ids) < 2:
        return float("nan")

    total_nll = 0.0
    n_tokens = 0
    for i in range(1, len(ids)):
        prefix = ids[:i].reshape(1, -1)
        logits = model(prefix)
        # logits shape: [1, seq_len, vocab_size] — pick last position
        last_logits = logits[0, -1, :]
        # softmax + gather log-prob at ids[i]
        log_probs = last_logits - mx.logsumexp(last_logits)
        target = ids[i].item()
        total_nll -= log_probs[target].item()
        n_tokens += 1

    return math.exp(total_nll / n_tokens) if n_tokens > 0 else float("nan")


def run_probe(model_id: str, samples: list[str]) -> dict:
    print(f"Loading {model_id}...")
    loaded = load(model_id)
    model, tokenizer = loaded[0], loaded[1]

    ppls = []
    for i, text in enumerate(samples):
        ppl = perplexity_one(model, tokenizer, text)
        ppls.append(ppl)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(samples)}: mean PPL so far = {sum(ppls)/len(ppls):.2f}")

    return {
        "model": model_id,
        "n_samples": len(samples),
        "mean_ppl": sum(ppls) / len(ppls),
        "median_ppl": sorted(ppls)[len(ppls) // 2],
        "min_ppl": min(ppls),
        "max_ppl": max(ppls),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="gpt-oss-20b-MXFP4-Q8",
                    help="Comma-separated model IDs or 'all'")
    ap.add_argument("--samples", type=int, default=10, help="Cap sample count for fast iteration")
    args = ap.parse_args()

    if args.models == "all":
        from src.measure_memory import MODELS
        model_ids = [m[0] for m in MODELS]
    else:
        model_ids = args.models.split(",")

    samples = PROBE_SAMPLES[: args.samples]
    results = [run_probe(m, samples) for m in model_ids]

    out_path = Path(__file__).resolve().parent.parent / "results" / "phase2_perplexity.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")
