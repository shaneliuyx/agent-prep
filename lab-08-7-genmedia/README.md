# lab-08-7-genmedia — W8.7

Companion lab for [[Week 8.7 - Generative Media and Fine-tuning]].

## What this lab ships

LoRA fine-tuning of SDXL on a brand subject + inference-time stack
(LoRA + ControlNet + IP-Adapter) + CLIP/FID evaluation.

## Phases

| Phase | Script | What it does |
|---|---|---|
| 1 | `src/train_lora.py` | LoRA fine-tuning on 10-15 brand reference images (~30 min M5 Pro) |
| 2 | `src/generate.py` | Inference: SDXL + LoRA + ControlNet (Canny) + IP-Adapter |
| 3 | `src/evaluate.py` | CLIP score + FID + human-panel scaffold |
| 4 | `src/generate_video.py` | I2V via Stable Video Diffusion |

## Setup

```bash
uv sync

# Drop 10-15 brand reference photos into dataset/
# Each photo needs a .txt caption with a UNIQUE token (e.g. "sks-brand" or random UUID)
# Caption pattern: "<myUniqueToken> a product photo on white background"
```

## Run

```bash
# Phase 1 — train LoRA (uses HF accelerate)
uv run accelerate launch src/train_lora.py \
    --pretrained-model stabilityai/stable-diffusion-xl-base-1.0 \
    --instance-data-dir dataset/ \
    --output-dir results/myBrand_lora \
    --rank 16 \
    --learning-rate 1e-4 \
    --max-train-steps 1000

# Phase 2 — generate with stacked LoRA + ControlNet + IP-Adapter
uv run python -m src.generate \
    --prompt "<myUniqueToken> a sleek product photo on white background" \
    --lora results/myBrand_lora \
    --control-image data/layout_canny.png \
    --style-image data/brand_style.jpg

# Phase 3 — evaluate
uv run python -m src.evaluate --generated-dir results/generated --reference-dir dataset/

# Phase 4 — video extension
uv run python -m src.generate_video --image results/generated/0.png

uv run pytest tests/ -v
```

## Hardware notes

- LoRA training: ~30 min on M5 Pro 48 GB at rank 16 / 1000 steps
- Inference: ~15-25s per 1024×1024 image (SDXL + 30 DDIM steps)
- SVD I2V: ~60-120s per 25-frame clip
