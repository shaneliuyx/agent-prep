"""Phase 1 — LoRA fine-tuning of SDXL on brand subject.

This file is a SPEC-level outline. Production-grade LoRA training scripts
exist in the diffusers repo at examples/dreambooth/train_dreambooth_lora_sdxl.py
which is ~800 LOC + has all the boilerplate (accelerate, EMA, mixed
precision, dataloader, etc).

This module provides:
  - LORA_CONFIG     — the config dict that downstream uses
  - the FORWARD-PROCESS math: $x_t = \\sqrt{\\bar{\\alpha}_t} x_0 + \\sqrt{1 - \\bar{\\alpha}_t} \\epsilon$
  - a training-step skeleton for pedagogical reference

To actually train: clone diffusers + run their script with `--rank 16`.
This file documents the SHAPE of what their script does.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def get_alpha_bar_schedule(n_timesteps: int = 1000,
                           beta_start: float = 1e-4,
                           beta_end: float = 0.02) -> list[float]:
    """Linear $\\beta_t$ schedule from $\\beta_{\\text{start}}$ to $\\beta_{\\text{end}}$.
    $\\bar{\\alpha}_t = \\prod_{s=1}^{t}(1 - \\beta_s)$"""
    betas = [beta_start + (beta_end - beta_start) * t / (n_timesteps - 1)
             for t in range(n_timesteps)]
    alphas = [1.0 - b for b in betas]
    alpha_bars = []
    running = 1.0
    for a in alphas:
        running *= a
        alpha_bars.append(running)
    return alpha_bars


def forward_step_x_t(x_0, t: int, eps, alpha_bar_t: float):
    """Forward process closed-form:
       $x_t = \\sqrt{\\bar{\\alpha}_t} x_0 + \\sqrt{1 - \\bar{\\alpha}_t} \\epsilon$

    Returns x_t. Caller supplies x_0 + epsilon as tensors of the same shape.
    `t` is the timestep index; passed for caller's logging clarity even though
    the closed-form takes alpha_bar_t directly.
    """
    import math
    _ = t  # parameter intentional for API symmetry with non-closed-form impls
    return math.sqrt(alpha_bar_t) * x_0 + math.sqrt(1.0 - alpha_bar_t) * eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained-model", default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--instance-data-dir", default="dataset/")
    ap.add_argument("--output-dir", default="results/myBrand_lora")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--max-train-steps", type=int, default=1000)
    args = ap.parse_args()

    print(
        f"\nSPEC-level LoRA training outline:\n"
        f"  pretrained: {args.pretrained_model}\n"
        f"  dataset:    {args.instance_data_dir}\n"
        f"  output:     {args.output_dir}\n"
        f"  rank:       {args.rank}\n"
        f"  lr:         {args.learning_rate}\n"
        f"  steps:      {args.max_train_steps}\n"
        f"\nTo actually train, clone diffusers and use:\n"
        f"  accelerate launch examples/dreambooth/train_dreambooth_lora_sdxl.py "
        f"--rank {args.rank} --learning_rate {args.learning_rate} "
        f"--max_train_steps {args.max_train_steps} "
        f"--instance_data_dir {args.instance_data_dir} "
        f"--output_dir {args.output_dir}"
    )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
