"""How long does reranking 50 → 5 actually cost per query?"""
import time, json
from sentence_transformers import CrossEncoder
from model_config import BGE_RERANKER_V2_M3

R = BGE_RERANKER_V2_M3
BATCH = 128   # or read from R.batch_size if you want to use the spec
rr = CrossEncoder(R.path, device="mps")
rr.model.half()
# Warm up
rr.predict([("q", "d")])

# Time 100 calls of `R.max_pairs_per_query` pairs each
pairs = [("what is mlx?", "MLX is Apple's array framework.")] * R.max_pairs_per_query
N = 100
t0 = time.time()
for _ in range(N):
    rr.predict(pairs, batch_size=BATCH)
elapsed = time.time() - t0
per_query_ms = (elapsed / N) * 1000
print(f"reranker: {per_query_ms:.1f} ms per query (50 pairs, batch={BATCH})")