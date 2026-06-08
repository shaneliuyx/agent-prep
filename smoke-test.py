"""Week 0 smoke test — verifies inference, embeddings, vector store, reranker, traces."""
import os, time
from pathlib import Path
from openai import OpenAI
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

HOME = os.path.expanduser("~")

# 1. oMLX chat completion (sonnet tier — Gemma 26B)
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key=os.getenv("OMLX_API_KEY", "not-needed"))
t0 = time.time()
resp = client.chat.completions.create(
    model="gemma-4-26B-A4B-it-heretic-4bit",
    messages=[{"role": "user", "content": "Reply with exactly: smoke-test-ok"}],
    max_tokens=64,
)
choice = resp.choices[0]
text = (choice.message.content or getattr(choice.message, "reasoning_content", None) or "").strip()
if not text:
    raise RuntimeError(
        f"oMLX returned empty content. finish_reason={choice.finish_reason}, "
        f"usage={resp.usage}, raw={choice.message.model_dump()}"
    )
print(f"[1/5] oMLX chat OK in {time.time()-t0:.1f}s → {text}")

# 2. Embedding (BGE-M3 on MPS)
emb = SentenceTransformer(f"{HOME}/models/bge-m3", device="mps")
vec = emb.encode(["hello agent world"])
print(f"[2/5] BGE-M3 embed OK → shape {vec.shape}")

# 3. Qdrant — create collection, upsert a point, search
qd = QdrantClient(url="http://127.0.0.1:6333")
if qd.collection_exists("smoke"):
    qd.delete_collection("smoke")
qd.create_collection("smoke", vectors_config=VectorParams(size=1024, distance=Distance.COSINE))
qd.upsert("smoke", points=[PointStruct(id=1, vector=vec[0].tolist(), payload={"text": "hello agent world"})])
hit = qd.query_points("smoke", query=vec[0].tolist(), limit=1).points[0]
print(f"[3/5] Qdrant upsert+search OK → id={hit.id} score={hit.score:.3f}")

# 4. Reranker — cross-encode a pair
rr = CrossEncoder(f"{HOME}/models/bge-reranker-v2-m3", device="mps")
score = float(rr.predict([("what is mlx?", "MLX is Apple's array framework.")])[0])
print(f"[4/5] BGE reranker OK → score {score:.2f}")

# 5. Phoenix reachable
import urllib.request
with urllib.request.urlopen("http://127.0.0.1:6006") as r:
    print(f"[5/5] Phoenix UI reachable → HTTP {r.status}")

print("\nALL SMOKE TESTS PASSED — ready for Week 1.")