"""Phase 3 — CLIP score + FID + human-panel scaffold.

CLIP score:
  $\\text{CLIPScore}(I, T) = \\max(0,\\ \\cos(\\text{CLIP}_{\\text{img}}(I),\\ \\text{CLIP}_{\\text{txt}}(T)))$

Higher = better text-image alignment. Saturates around 0.32.

FID:
  $\\text{FID}(P, Q) = \\|\\mu_P - \\mu_Q\\|^2 + \\text{tr}(\\Sigma_P + \\Sigma_Q - 2(\\Sigma_P \\Sigma_Q)^{1/2})$

Lower = better distribution match. SDXL on COCO hits 7-10.

CLIP + FID are NECESSARY but NOT SUFFICIENT. Always include human-panel scores.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def clip_score(images: list, prompts: list[str]) -> float:
    """Mean CLIP score across image-prompt pairs.
    SPEC: implementation via openai/clip-vit-large-patch14."""
    from transformers import CLIPProcessor, CLIPModel
    import torch

    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    inputs = processor(text=prompts, images=images, return_tensors="pt",
                       padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        # Diagonal of similarity matrix = per-pair scores
        scores = outputs.logits_per_image.diag()
    return float(scores.mean().item())


def fid_score(generated_dir: str, reference_dir: str) -> float:
    """Compute FID between two image directories via cleanfid.
    SPEC: install cleanfid separately (`pip install clean-fid`)."""
    from cleanfid import fid
    return fid.compute_fid(generated_dir, reference_dir)


def human_panel_scaffold(generated_paths: list[str],
                        output_csv: str) -> None:
    """Write a CSV stub for human-panel rating collection.
    Each rater fills the 'rating' column (1-10) plus optional 'notes'."""
    import csv
    with open(output_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "rater_id", "rating_1_10", "brand_fit_1_10", "notes"])
        for p in generated_paths:
            w.writerow([p, "", "", "", ""])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generated-dir", required=True)
    ap.add_argument("--reference-dir", required=True)
    ap.add_argument("--panel-csv", default="results/human_panel.csv")
    args = ap.parse_args()

    paths = sorted(str(p) for p in Path(args.generated_dir).glob("*.png"))
    print(f"Found {len(paths)} generated images")

    fid = fid_score(args.generated_dir, args.reference_dir)
    print(f"FID = {fid:.2f}  (lower is better; SDXL on COCO ~7-10)")

    Path(args.panel_csv).parent.mkdir(parents=True, exist_ok=True)
    human_panel_scaffold(paths, args.panel_csv)
    print(f"Wrote panel scaffold {args.panel_csv}")


if __name__ == "__main__":
    main()
