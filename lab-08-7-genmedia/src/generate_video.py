"""Phase 4 — image-to-video via Stable Video Diffusion.

Takes a static image, generates 14-25 frames of motion.
For brand consistency over video, the LoRA's identity carries across
the I2V step IF the LoRA-trained image is used as input.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def generate_video(image_path: str,
                  output_path: str = "results/video.mp4",
                  num_frames: int = 25,
                  num_inference_steps: int = 25,
                  fps: int = 8) -> str:
    """SVD I2V. Lazy imports."""
    from diffusers import StableVideoDiffusionPipeline
    from diffusers.utils import load_image, export_to_video

    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
    )
    input_image = load_image(image_path)
    frames = pipe(
        input_image,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
    ).frames[0]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, output_path, fps=fps)
    return output_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--output", default="results/video.mp4")
    ap.add_argument("--num-frames", type=int, default=25)
    ap.add_argument("--fps", type=int, default=8)
    args = ap.parse_args()

    path = generate_video(args.image, args.output, args.num_frames, fps=args.fps)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
