"""Phase 1 — measure disk + runtime memory across the fleet.

Verifies the bit-width math:  $M_{\\text{weights}} = N_{\\text{params}} \\times \\text{bytes per param}$

Compares 3 numbers per model:
  - theoretical_gb  = params * bytes_per_param / 1e9
  - disk_gb         = du -sb of cached weights
  - rss_delta_gb    = process RSS growth after mlx_lm.load()

Production rule: all 3 should agree within ~10-15%. Larger gap reveals
tokenizer/config overhead, MLX intermediate buffers, or platform-specific
RSS counter semantics (bytes on macOS vs KB on Linux).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import TypedDict

import psutil


# Fleet config — match curriculum chapter §1.1 + W4 §1.5 gateway role map
MODELS: list[tuple[str, float, float]] = [
    # (model_id_or_hub_path, n_params_billions, bytes_per_param)
    ("gpt-oss-20b-MXFP4-Q8", 20e9, 0.5),                          # MXFP4 weights + Q8 activations approx
    ("MLX-Qwen3.5-9B-GLM5.1-Distill-v1-8bit", 9e9, 1.0),          # Q8 weights
    ("gemma-4-26B-A4B-it-heretic-4bit", 26e9, 0.5),               # Q4 weights
    ("Qwen3.6-35B-A3B-nvfp4", 35e9, 0.5),                         # NVFP4 weights
]


class MemoryRow(TypedDict):
    model: str
    params_b: float
    bytes_per_param: float
    theoretical_gb: float
    disk_gb: float | None
    rss_delta_gb: float


def _find_disk_path(model_id: str) -> Path | None:
    """Locate the model in HuggingFace hub cache. Returns None if not found."""
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    sanitized = model_id.replace("/", "--")
    matches = list(cache_root.glob(f"models--*{sanitized}*"))
    return matches[0] if matches else None


def _du_gb(path: Path) -> float:
    """`du -sb` -> GB. Uses bytes (not KB) so Mac + Linux agree."""
    result = subprocess.run(
        ["du", "-sk", str(path)], capture_output=True, text=True, check=True
    )
    kb = int(result.stdout.split()[0])
    return kb * 1024 / 1e9


def measure(model_id: str, n_params: float, bytes_per_param: float) -> MemoryRow:
    """Measure one model's footprint. Theoretical + disk + RSS-delta."""
    theoretical_gb = n_params * bytes_per_param / 1e9
    disk_path = _find_disk_path(model_id)
    disk_gb = _du_gb(disk_path) if disk_path else None

    process = psutil.Process(os.getpid())
    rss_before = process.memory_info().rss / 1e9

    # mlx_lm import deferred — only when this function actually runs
    try:
        from mlx_lm import load
        # mlx_lm.load may return 2-tuple (model, tokenizer) or 3-tuple (model, tokenizer, config)
        loaded = load(model_id)
        model = loaded[0]
        rss_after = process.memory_info().rss / 1e9
        del model
    except Exception as e:
        print(f"  WARN: load failed for {model_id}: {type(e).__name__}: {e}", file=sys.stderr)
        rss_after = rss_before

    return MemoryRow(
        model=model_id,
        params_b=n_params / 1e9,
        bytes_per_param=bytes_per_param,
        theoretical_gb=theoretical_gb,
        disk_gb=disk_gb,
        rss_delta_gb=rss_after - rss_before,
    )


def render_table(rows: list[MemoryRow]) -> str:
    header = (
        f"{'Model':<50}  {'Params (B)':>11}  {'Theory (GB)':>12}  "
        f"{'Disk (GB)':>10}  {'RSS Δ (GB)':>11}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        disk = f"{r['disk_gb']:.1f}" if r["disk_gb"] is not None else "?"
        lines.append(
            f"{r['model']:<50}  {r['params_b']:>11.1f}  {r['theoretical_gb']:>12.1f}  "
            f"{disk:>10}  {r['rss_delta_gb']:>11.1f}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    rows = [measure(m, p, b) for m, p, b in MODELS]
    table = render_table(rows)
    print(table)

    out_path = Path(__file__).resolve().parent.parent / "results" / "phase1_memory.md"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(f"# Phase 1 — Memory measurement\n\n```\n{table}\n```\n")
    print(f"\nWrote {out_path}")
