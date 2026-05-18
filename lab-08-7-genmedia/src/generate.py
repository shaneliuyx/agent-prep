"""Phase 2 — inference with stacked LoRA + ControlNet + IP-Adapter.

Three orthogonal control levers (W8.7 §Concept 4):
  LoRA          -> identity (the brand subject)
  ControlNet    -> structure (layout from Canny / depth / pose)
  IP-Adapter    -> style (palette + lighting from reference photo)

Combined output meets production brand-consistency bar.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def generate_image(prompt: str,
                  lora_path: str,
                  control_image_path: str,
                  style_image_path: str,
                  output_path: str,
                  num_inference_steps: int = 30,
                  guidance_scale: float = 7.5,
                  controlnet_scale: float = 0.7,
                  ip_adapter_scale: float = 0.6) -> str:
    """Generate one image. Lazy imports keep the script syntax-checkable
    without the diffusers wheel installed."""
    from diffusers import (
        StableDiffusionXLControlNetPipeline, ControlNetModel,
    )
    from diffusers.utils import load_image

    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-canny-sdxl-1.0",
    )
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
    )

    # Stack: LoRA + IP-Adapter on top of SDXL+ControlNet
    pipe.load_lora_weights(lora_path)
    pipe.load_ip_adapter(
        "h94/IP-Adapter", subfolder="sdxl_models",
        weight_name="ip-adapter_sdxl.bin",
    )

    canny = load_image(control_image_path)
    style = load_image(style_image_path)

    result = pipe(
        prompt=prompt,
        image=canny,
        ip_adapter_image=style,
        controlnet_conditioning_scale=controlnet_scale,
        ip_adapter_scale=ip_adapter_scale,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )
    result.images[0].save(output_path)
    return output_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--control-image", required=True)
    ap.add_argument("--style-image", required=True)
    ap.add_argument("--output", default="results/generated/0.png")
    args = ap.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    path = generate_image(args.prompt, args.lora, args.control_image,
                         args.style_image, args.output)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
