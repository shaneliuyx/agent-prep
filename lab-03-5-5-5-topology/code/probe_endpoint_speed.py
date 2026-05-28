"""Micro-benchmark — compare two MLX inference servers on identical
prompts + identical model weights. Both endpoints serve
`MLX-Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled-4bit`.

  oMLX      → http://localhost:8000/v1   (alias served via OMLX_MODEL)
  rapid-mlx → http://localhost:8001/v1   (alias 'qwen35-35b-reasoning')

Output: per-prompt wall + tokens_in/out + tokens/sec for each endpoint
+ aggregate comparison table.

Run: python code/probe_endpoint_speed.py
"""
import json
import os
import time
import httpx
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

ENDPOINTS = [
    {
        "name": "oMLX",
        "url": "http://localhost:8000/v1/chat/completions",
        "model": "MLX-Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled-4bit",
        "key": os.getenv("OMLX_API_KEY") or os.getenv("OPENAI_API_KEY") or "sk-local",
    },
    {
        "name": "rapid-mlx",
        "url": "http://localhost:8001/v1/chat/completions",
        "model": "qwen35-35b-reasoning",
        "key": "not-needed",
    },
]

# Three prompt shapes spanning short / structured / long
PROMPTS = [
    {
        "label": "short_factual",
        "system": "Answer in one sentence.",
        "user": "What is the capital of France?",
        "max_tokens": 100,
    },
    {
        "label": "json_decomposition",
        "system": 'Decompose the question into EXACTLY 3 sub-questions. Return JSON only: {"sub_questions": ["q1","q2","q3"]}',
        "user": "Compare regulatory frameworks for AI across EU, US, and UK.",
        "max_tokens": 400,
    },
    {
        "label": "long_synthesis",
        "system": "Write a 5-paragraph explanation.",
        "user": "Explain how transformer attention works, including QKV projections, softmax scaling, and multi-head attention.",
        "max_tokens": 1500,
    },
]


def call(ep, prompt):
    body = {
        "model": ep["model"],
        "messages": [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ],
        "temperature": 0.0,
        "max_tokens": prompt["max_tokens"],
    }
    headers = {"Authorization": f"Bearer {ep['key']}"}
    t0 = time.perf_counter()
    r = httpx.post(ep["url"], json=body, headers=headers, timeout=120)
    wall = time.perf_counter() - t0
    r.raise_for_status()
    data = r.json()
    usage = data.get("usage", {})
    msg = data["choices"][0]["message"]
    content = msg.get("content") or msg.get("reasoning_content") or ""
    return {
        "wall_s": round(wall, 2),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "tokens_per_sec": round(usage.get("completion_tokens", 0) / wall, 1) if wall > 0 else 0,
        "content_len_chars": len(content),
    }


def main():
    results = {ep["name"]: {} for ep in ENDPOINTS}

    for prompt in PROMPTS:
        print(f"\n=== {prompt['label']} (max_tokens={prompt['max_tokens']}) ===")
        for ep in ENDPOINTS:
            print(f"  [{ep['name']:<10}] ", end="", flush=True)
            try:
                r = call(ep, prompt)
                results[ep["name"]][prompt["label"]] = r
                print(f"wall={r['wall_s']}s  out={r['completion_tokens']}tok  {r['tokens_per_sec']}tok/s  chars={r['content_len_chars']}")
            except Exception as e:
                print(f"FAILED: {e}")
                results[ep["name"]][prompt["label"]] = {"error": str(e)}

    # Aggregate
    print("\n" + "=" * 70)
    print(f"{'Prompt':<22} {'oMLX wall':>12} {'rapid-mlx wall':>16} {'Δ':>10} {'rapid/oMLX':>12}")
    print("-" * 70)
    for prompt in PROMPTS:
        omlx = results["oMLX"].get(prompt["label"], {}).get("wall_s")
        rapid = results["rapid-mlx"].get(prompt["label"], {}).get("wall_s")
        if omlx and rapid:
            delta = round(omlx - rapid, 2)
            ratio = round(rapid / omlx, 2)
            faster = "rapid-mlx" if ratio < 0.95 else "oMLX" if ratio > 1.05 else "tie"
            print(f"{prompt['label']:<22} {omlx:>12.2f} {rapid:>16.2f} {delta:>+10.2f} {ratio:>11.2f}× ({faster})")
        else:
            print(f"{prompt['label']:<22} (one or both endpoints failed)")

    print()
    # tokens/sec average
    print(f"{'Prompt':<22} {'oMLX tok/s':>12} {'rapid-mlx tok/s':>16}")
    print("-" * 60)
    for prompt in PROMPTS:
        omlx_tps = results["oMLX"].get(prompt["label"], {}).get("tokens_per_sec")
        rapid_tps = results["rapid-mlx"].get(prompt["label"], {}).get("tokens_per_sec")
        if omlx_tps and rapid_tps:
            print(f"{prompt['label']:<22} {omlx_tps:>12.1f} {rapid_tps:>16.1f}")

    # Save raw
    out_path = "results/endpoint_speed_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw results saved → {out_path}")


if __name__ == "__main__":
    main()
