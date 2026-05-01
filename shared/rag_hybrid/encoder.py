"""BGE-M3 hybrid encoder + SentenceTransformer dense encoder. Lazy load.

Two encoder classes because the existing labs use both:

  - HybridEncoder wraps BGEM3FlagModel — one forward pass produces dense
    (1024-d) AND sparse (token_id → weight) representations. Used for
    hybrid-collection ingest and hybrid retrieval queries.

  - DenseEncoder wraps SentenceTransformer — dense-only, lighter, no
    sparse weights. Used for dense-only collections, reranker pipelines,
    chunk sweeps. Faster module import (no FlagEmbedding pull).

Both lazy-load. Constructing the wrapper costs nothing; the model load
happens on first ``.encode()`` call. Lets the parity harness import these
without a GPU.

fp16 policy: defaults to OFF (matches every existing callsite — both labs
pass ``use_fp16=False`` explicitly because BGE-M3's sparse head occasionally
emits NaNs on MPS in fp16. Cross-encoder rerank is fp16-safe; encoder is
not). Override via ``EncoderConfig(use_fp16=True)`` on CUDA where it's safe.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .models import EmbedModelSpec


@dataclass(frozen=True)
class EncoderConfig:
    """Encoder load + run config. Frozen so it's safe to share across threads."""
    spec: EmbedModelSpec
    device: str = "mps"          # autoconfig.probe_device() will override
    use_fp16: bool = False       # OFF default — matches every existing callsite (NaN risk on MPS)
    default_batch_size: int = 64  # autoconfig.probe_memory() will override


class HybridEncoder:
    """BGE-M3 in hybrid mode — dense + sparse from one forward pass.

    encode() returns BGE-M3's native shape:
        {"dense_vecs": np.ndarray of shape (N, dim),
         "lexical_weights": list[dict[int, float]] | None}

    Caller converts to whatever the storage layer expects. Matches the output
    contract of BGEM3FlagModel.encode() so existing callers can drop in.
    """

    def __init__(self, cfg: EncoderConfig) -> None:
        if not cfg.spec.supports_sparse:
            raise ValueError(
                f"HybridEncoder requires a model with supports_sparse=True; "
                f"got {cfg.spec.name!r}. Use DenseEncoder for dense-only models."
            )
        self.cfg = cfg
        self._model: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        # Local import — keeps `import rag_hybrid.encoder` cheap.
        from FlagEmbedding import BGEM3FlagModel

        self._model = BGEM3FlagModel(
            self.cfg.spec.path,
            use_fp16=self.cfg.use_fp16,
            device=self.cfg.device,
        )

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
        with_sparse: bool = True,
        with_colbert: bool = False,
    ) -> dict:
        """Encode a list of texts. Returns BGE-M3's native output dict.

        ``batch_size=None`` uses ``cfg.default_batch_size``. ``with_sparse``
        controls the sparse head — set False to skip lexical weights when
        callers only need dense (saves ~25% encode time).

        When ``cfg.use_fp16=True`` the output is NaN-guarded: any NaN in
        dense or sparse output raises ``RuntimeError`` so the Ingestor halts
        before writing corrupted vectors to Qdrant. Cheap check (~µs/batch)
        and only runs in fp16 mode where the risk exists.
        """
        self._ensure_loaded()
        out = self._model.encode(
            list(texts),
            batch_size=batch_size if batch_size is not None else self.cfg.default_batch_size,
            return_dense=True,
            return_sparse=with_sparse,
            return_colbert_vecs=with_colbert,
        )
        if self.cfg.use_fp16:
            self._check_no_nan(out, with_sparse)
        return out

    def _check_no_nan(self, out: dict, with_sparse: bool) -> None:
        """Fail-fast on NaN in fp16 output. Without this guard, NaN sparse
        values silently flow to Qdrant where they break that point's index
        entry without raising — discoverable only at retrieval time as
        empty/wrong results."""
        import math

        import numpy as np

        dense = out.get("dense_vecs")
        if dense is not None and np.isnan(dense).any():
            raise RuntimeError(
                f"BGE-M3 dense output contains NaN — fp16 corruption detected on "
                f"device {self.cfg.device!r}. Set use_fp16=False to recover, then "
                f"file an issue with sample text + FlagEmbedding/torch versions."
            )
        if with_sparse:
            sparse = out.get("lexical_weights") or []
            for sd in sparse:
                if sd and any(math.isnan(float(v)) for v in sd.values()):
                    raise RuntimeError(
                        f"BGE-M3 sparse output contains NaN — fp16 corruption "
                        f"detected on device {self.cfg.device!r}. "
                        f"Set use_fp16=False to recover."
                    )


class DenseEncoder:
    """SentenceTransformer dense-only — used when sparse output isn't needed.

    encode() returns numpy ndarray (N, dim). Normalized by default to match
    the cosine-distance schema used by every collection in the labs.
    """

    def __init__(self, cfg: EncoderConfig) -> None:
        self.cfg = cfg
        self._model: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(
            self.cfg.spec.path,
            device=self.cfg.device,
            trust_remote_code=self.cfg.spec.trust_remote_code,
        )

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
        normalize: bool = True,
    ) -> Any:
        """Encode texts to a dense ndarray. ``normalize=True`` matches the
        cosine-distance schema used everywhere in the labs."""
        self._ensure_loaded()
        return self._model.encode(
            list(texts),
            batch_size=batch_size if batch_size is not None else self.cfg.default_batch_size,
            normalize_embeddings=normalize,
        )
