"""Cross-encoder reranker wrapper — lazy load, device-aware, fp16-opt-in.

Two-stage retrieval pattern: bi-encoder (BGE-M3) does fast recall over the
whole corpus, cross-encoder (BGE-reranker-v2-m3) re-scores the top-N candidates
with full query-doc attention. Cross-encoder is ~50× slower per pair but only
runs on N≈50 pairs per query, so end-to-end latency stays under 200ms.

Why lazy load:
  Importing this module shouldn't pull 2GB of weights. Existing callsites that
  load the reranker at module top (02_compress.py, 02b_answer_eval.py) make the
  library untestable without a GPU. Lazy load fixes that.

Why fp16 is opt-in (default fp32):
  Across the existing labs, four callsites run fp32, one (retrieve.py) calls
  ``reranker.model.half()`` for fp16. Defaulting to fp32 preserves frozen
  result-file hashes during migration; retrieve.py opts in via
  ``RerankerConfig(fp16=True)``. fp16 cross-encoder is safe (no NaN risk like
  BGE-M3 sparse head) and saves ~30% latency on MPS.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .models import RerankerSpec


@dataclass(frozen=True)
class RerankerConfig:
    """How to load and run the cross-encoder. Frozen so it's safe to share
    across threads — the underlying CrossEncoder load happens on first use."""
    spec: RerankerSpec
    device: str = "mps"   # autoconfig.probe_device() will override when implemented
    fp16: bool = False    # opt-in; preserves fp32 baseline behavior across most callsites


class CrossEncoderReranker:
    """Lazy-loaded cross-encoder. Construct cheaply, load on first ``rerank`` call.

    Usage:
        rr = CrossEncoderReranker(RerankerConfig(spec=BGE_RERANKER_V2_M3, fp16=True))
        top = rr.rerank(query, candidates, top_k=5)
    """

    def __init__(self, cfg: RerankerConfig) -> None:
        self.cfg = cfg
        self._model: Any = None  # populated on first rerank() call

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        # Local import — keeps `import rag_hybrid.rerank` cheap.
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(self.cfg.spec.path, device=self.cfg.device)
        if self.cfg.fp16:
            # .half() flips weights to fp16 in place. Cross-encoder doesn't have
            # the sparse-head NaN failure mode that BGE-M3 sometimes shows on MPS.
            model.model.half()
        self._model = model

    def rerank(
        self,
        query: str,
        candidates: Sequence[Any],
        top_k: int | None = None,
    ) -> list[tuple[str, str, float]]:
        """Score (query, candidate.payload['text']) pairs, return top_k descending.

        Returns ``(doc_id, text, score)`` triples. ``doc_id`` and ``text`` are
        read from each candidate's ``payload``; this matches the Qdrant
        ``ScoredPoint`` shape used everywhere else in the labs.

        ``top_k=None`` returns all candidates re-sorted (no truncation).
        """
        self._ensure_loaded()
        pairs = [(query, c.payload["text"]) for c in candidates]
        scores = self._model.predict(pairs, batch_size=self.cfg.spec.batch_size)
        order = sorted(zip(candidates, scores), key=lambda x: -x[1])
        if top_k is not None:
            order = order[:top_k]
        return [(c.payload["doc_id"], c.payload["text"], float(s)) for c, s in order]
