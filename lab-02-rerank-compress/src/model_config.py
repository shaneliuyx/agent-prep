"""Compatibility shim — canonical home is shared/rag_hybrid/models.py.

Re-exports all public symbols so existing imports keep working unchanged:

    from model_config import BGE_M3, BGE_M3_HYBRID, ...

New code should import from rag_hybrid.models directly.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SHARED = _REPO_ROOT / "shared"
if str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))

from rag_hybrid.models import (  # noqa: E402, F401
    HOME,
    EmbedModelSpec,
    RerankerSpec,
    HybridCollectionSpec,
    CollectionSpec,
    ChunkSweepCollectionSpec,
    BGE_M3,
    BGE_RERANKER_V2_M3,
    BGE_M3_HYBRID,
    BGE_M3_HNSW,
    FIQA_HYBRID,
    ALL_COLLECTIONS,
    LONG_DOC_PASSAGES,
    CHUNK_SIZES,
    CHUNK_OVERLAPS,
    SWEEP_VARIANTS,
)
