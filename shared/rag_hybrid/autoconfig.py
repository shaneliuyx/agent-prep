"""autoconfig — system-aware default config for the rag_hybrid pipeline.

Q2=i scope: probe the host system (device, memory, cpu count) and recommend
a config tuple. No corpus-aware tuning yet — that's a future phase that
needs richer signal (corpus size + entity density + lang).

Why this exists: the W2.5 dense ingest used ``ENCODE_BATCH=64`` with the
comment "smaller because passages are shorter" — wrong reasoning. The real
constraint is host memory: 8 GB system → 32, 16 GB → 64, 32+ GB → 128.
Encoding this once means future labs don't re-derive it.

Provenance is preserved in the returned config's ``notes`` field — every
recommendation comes with a one-line explanation so an operator can audit
what was decided and why.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from .encoder import EncoderConfig
from .ingest import IngestConfig
from .models import EmbedModelSpec, RerankerSpec
from .rerank import RerankerConfig
from .retrieve import RetrieveConfig


@dataclass(frozen=True)
class SystemProbe:
    """What we measured about the host. Each field has a per-field comment
    explaining how it influences the recommendation."""
    device: str                    # "cuda" | "mps" | "cpu"
    fp16_safe_for_encoder: bool    # BGE-M3 sparse head NaN-prone on MPS in fp16; safe on CUDA
    fp16_safe_for_reranker: bool   # cross-encoder is fp16-safe on both MPS and CUDA
    total_memory_gb: float         # used to pick encode_batch tier
    cpu_count: int                 # influences upsert worker count (not yet wired)


@dataclass(frozen=True)
class RecommendedConfig:
    """Bundle of per-component configs the autoconfig produced. ``notes`` is
    the audit trail — feed it to your logger / write it next to results so
    months-later you remember why batch size was 128 on this run."""
    encoder: EncoderConfig
    reranker: RerankerConfig
    ingest: IngestConfig
    retrieve: RetrieveConfig
    probe: SystemProbe
    notes: list[str]


def _detect_device() -> str:
    """Return the highest-priority available device name. ``cpu`` is the
    fallback, never the recommendation when MPS or CUDA is available."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _total_memory_gb() -> float:
    """Total system RAM in GB. Falls back to a conservative 16.0 if psutil
    is unavailable — better to under-tune than crash on OOM."""
    try:
        import psutil

        return psutil.virtual_memory().total / 1e9
    except ImportError:
        return 16.0


def probe_system() -> SystemProbe:
    """Read host facts. Cheap (no model load, no GPU work)."""
    device = _detect_device()
    return SystemProbe(
        device=device,
        # BGE-M3 specifically — its sparse head has been observed to emit NaNs
        # on MPS in fp16. Only flip on CUDA where the issue doesn't reproduce.
        fp16_safe_for_encoder=(device == "cuda"),
        # Cross-encoder rerank is fp16-safe on both MPS and CUDA; only CPU stays fp32.
        fp16_safe_for_reranker=(device in ("cuda", "mps")),
        total_memory_gb=_total_memory_gb(),
        cpu_count=os.cpu_count() or 1,
    )


def recommend_encode_batch(probe: SystemProbe) -> int:
    """Three-tier mapping from memory budget to encode batch size.

    Tier breakpoints come from observed BGE-M3 forward-pass memory at
    sequence length 512: ~50 MB / batch slot. 32 GB host can handle 128
    with headroom for HNSW indexing competing for the same memory.
    """
    if probe.total_memory_gb >= 32:
        return 128
    if probe.total_memory_gb >= 16:
        return 64
    return 32


def encoder_config_for(embed_spec: EmbedModelSpec) -> EncoderConfig:
    """One-line encoder config from a system probe. Use when a script only
    needs an encoder (ingest pipeline) and doesn't care about reranker tuning."""
    probe = probe_system()
    return EncoderConfig(
        spec=embed_spec,
        device=probe.device,
        use_fp16=probe.fp16_safe_for_encoder,
        default_batch_size=recommend_encode_batch(probe),
    )


def ingest_config() -> IngestConfig:
    """One-line ingest config from the memory probe. Pairs with
    ``encoder_config_for`` for ingest-only scripts."""
    return IngestConfig(encode_batch=recommend_encode_batch(probe_system()))


def recommend(
    embed_spec: EmbedModelSpec,
    reranker_spec: RerankerSpec,
    *,
    overrides: dict | None = None,
) -> RecommendedConfig:
    """Probe the system and assemble per-component configs.

    ``overrides`` is currently a placeholder for the corpus-probe phase —
    treat as ``None`` for now. Caller can always replace any returned
    config dataclass field directly via ``dataclasses.replace`` if they
    disagree with a recommendation.
    """
    if overrides:
        # Keeps the API stable when corpus-probe lands; current scope (Q2=i)
        # ignores overrides.
        pass

    probe = probe_system()
    encode_batch = recommend_encode_batch(probe)

    notes = [
        f"device={probe.device}",
        f"memory={probe.total_memory_gb:.1f}GB",
        f"cpu={probe.cpu_count}",
        f"encode_batch={encode_batch} (memory tier)",
        f"encoder.fp16={'on' if probe.fp16_safe_for_encoder else 'off'} "
        f"({'CUDA-safe' if probe.device == 'cuda' else 'BGE-M3 sparse-head NaN risk on ' + probe.device})",
        f"reranker.fp16={'on' if probe.fp16_safe_for_reranker else 'off'} "
        f"(cross-encoder {'fp16-safe' if probe.device != 'cpu' else 'fp32-only on CPU'})",
    ]

    return RecommendedConfig(
        encoder=EncoderConfig(
            spec=embed_spec,
            device=probe.device,
            use_fp16=probe.fp16_safe_for_encoder,
            default_batch_size=encode_batch,
        ),
        reranker=RerankerConfig(
            spec=reranker_spec,
            device=probe.device,
            fp16=probe.fp16_safe_for_reranker,
        ),
        ingest=IngestConfig(encode_batch=encode_batch),
        retrieve=RetrieveConfig(),
        probe=probe,
        notes=notes,
    )
