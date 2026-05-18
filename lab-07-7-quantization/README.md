# lab-07-7-quantization — W7.7

Companion lab for [[Week 7.7 - Quantization and Inference Optimization]] in the curriculum vault.

## What this lab measures

Bit-width / accuracy / latency / memory trade-offs ON LOCAL MLX FLEET (M5 Pro 48 GB):

- `gpt-oss-20b-MXFP4-Q8` (20B params, MXFP4 weights + Q8 activations)
- `MLX-Qwen3.5-9B-GLM5.1-Distill-v1-8bit` (9B params, Q8 weights)
- `gemma-4-26B-A4B-it-heretic-4bit` (26B params, Q4 weights)
- `Qwen3.6-35B-A3B-nvfp4` (35B params, NVFP4 weights)

## Setup

```bash
uv sync
# .env:
#   OMLX_BASE_URL=http://127.0.0.1:8000/v1
#   OMLX_API_KEY=<your key>
```

## Phases

| Phase | Script | What it measures |
|---|---|---|
| 1 | `src/measure_memory.py` | Disk size + runtime RSS + theoretical $N \cdot b$ |
| 2 | `src/perplexity_probe.py` | Perplexity across quant levels on 100-sample multi-domain probe |
| 3 | `src/kv_cache_curve.py` | KV-cache footprint + per-token latency vs seq_len |
| 4 | `src/regime_diagnostic.py` | Memory-bound vs compute-bound classification via batch sweep |

## Run

```bash
# Phase 1 — memory measurement (~5 min)
uv run python -m src.measure_memory

# Phase 2 — perplexity probe (~30 min)
uv run python -m src.perplexity_probe --models all --samples 100

# Phase 3 — KV-cache curve (~1 hour)
uv run python -m src.kv_cache_curve --seq-lens 512,1024,2048,4096,8192

# Phase 4 — regime diagnostic (~30 min)
uv run python -m src.regime_diagnostic --batch-sizes 1,2,4,8,16,32

# Tests
uv run pytest tests/ -v
```

Results land in `results/`. RESULTS.md aggregates measured numbers for chapter back-fill.
