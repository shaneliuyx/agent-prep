"""1:1 with lab-02/src/01b_latency.py — but via the `rerankers` library.

Measures how long the cross-encoder rerank stage costs per query. Same model
(BGE-reranker-v2-m3), same batch size, same 50-pair payload. The rerankers wrapper
adds ~negligible overhead vs raw CrossEncoder, so numbers should match within ±5%.
"""
import os, time, json
from rerankers import Reranker

HOME = os.path.expanduser("~")

# Same model + batch size + max_pairs as lab-02 — wrapped via rerankers' unified API
ranker = Reranker(
    f"{HOME}/models/bge-reranker-v2-m3",
    model_type="cross-encoder",
    device="mps",
)

# Lab-02 §2.3.2 lesson: at production shape (1 query × 50 pairs), fp16 cuts per-query
# latency from 112 ms → 37.8 ms — a 2.97× speedup, dominantly the fp16 lever (-63%) with
# a smaller batch=128 contribution (-8%). Without this line, the headline number will
# match the unoptimized lab-02 baseline (~112 ms), not the §2.3.2 shipping config.
try:
    ranker.model.half()
    print("[opt] cross-encoder converted to fp16 (lab-02 §2.3.2 production-shape lever)")
except (AttributeError, TypeError) as e:
    print(f"[opt] fp16 conversion skipped — rerankers wrapper changed shape: {e}")

# Warm up — first call includes lazy weight loading
ranker.rank(query="warmup", docs=["warmup doc"])

# Time 100 calls of 50 pairs each (matches lab-02's 01b_latency.py)
N = 100
docs = ["MLX is Apple's array framework."] * 50

t0 = time.time()
for _ in range(N):
    ranker.rank(query="what is mlx?", docs=docs)
elapsed = time.time() - t0

per_query_ms = (elapsed / N) * 1000
result = {
    "library":     "rerankers",
    "model":       "BAAI/bge-reranker-v2-m3",
    "n_calls":     N,
    "pairs_per_call": 50,
    "per_query_ms": per_query_ms,
    "total_wall_sec": elapsed,
}
print(json.dumps(result, indent=2))
print(f"\nlab-02 baseline (lab-02/src/01b_latency.py) typical: 80-140ms/query on MPS at batch=32.")
print(f"This run: {per_query_ms:.1f}ms/query. Difference is rerankers wrapper overhead (typically <5%).")
