"""LoRA configuration for SDXL brand-subject fine-tuning.

Rank 16 = sweet spot per W8.7 §Concept 3:
  - rank 4 too restrictive for visual concepts
  - rank 32 overfits with only 10-15 images
  - 16 is the diffusers default + production default

target_modules selects ATTENTION layers only (cross-attention is where
text-conditioning meets image features). LoRA on FFN layers adds 2x
parameter count for marginal gain.

bias='none' — bias updates are usually destructive; freeze them.
"""
from __future__ import annotations


# Standard SDXL LoRA config dict.
# Use as input to peft.LoraConfig at training time:
#   from peft import LoraConfig
#   cfg = LoraConfig(**STANDARD_LORA_CONFIG)
STANDARD_LORA_CONFIG: dict = {
    "r": 16,
    "lora_alpha": 32,                # usually 2x rank
    "target_modules": ["to_q", "to_k", "to_v", "to_out.0"],
    "lora_dropout": 0.05,
    "bias": "none",
}

# More aggressive config for SUBJECT-DRIVEN training (DreamBooth-style)
SUBJECT_LORA_CONFIG: dict = {
    **STANDARD_LORA_CONFIG,
    "r": 8,                          # lower rank reduces catastrophic forgetting
    "lora_alpha": 16,
}
