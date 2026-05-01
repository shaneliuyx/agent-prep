"""Phase 2 parity test — model_config.py shim must re-export rag_hybrid.models
identity-equivalent to the canonical module.

Object identity (`is`) — not just equality. Confirms shim doesn't accidentally
deep-copy or rebuild specs (which would be equal-by-value but break any caller
using `id(spec)` for caching or `spec1 is spec2` checks).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main() -> int:
    # Add both paths the way labs do.
    sys.path.insert(0, str(REPO / "shared"))
    sys.path.insert(0, str(REPO / "lab-02-rerank-compress" / "src"))

    import model_config as via_shim
    from rag_hybrid import models as canonical

    public = [
        "HOME",
        "EmbedModelSpec",
        "RerankerSpec",
        "HybridCollectionSpec",
        "CollectionSpec",
        "ChunkSweepCollectionSpec",
        "BGE_M3",
        "BGE_RERANKER_V2_M3",
        "BGE_M3_HYBRID",
        "BGE_M3_HNSW",
        "FIQA_HYBRID",
        "ALL_COLLECTIONS",
        "LONG_DOC_PASSAGES",
        "CHUNK_SIZES",
        "CHUNK_OVERLAPS",
        "SWEEP_VARIANTS",
    ]

    failures = []
    for name in public:
        a = getattr(via_shim, name, None)
        b = getattr(canonical, name, None)
        if a is None:
            failures.append(f"{name}: missing from shim")
            continue
        if b is None:
            failures.append(f"{name}: missing from canonical")
            continue
        if a is not b:
            failures.append(f"{name}: identity drift (shim={id(a)} canonical={id(b)})")

    # Spot-check a few values to catch the rare case where identity matches but
    # field values diverged (e.g. someone edited the canonical without re-importing).
    assert canonical.BGE_M3.dim == 1024, "BGE_M3.dim drifted"
    assert canonical.BGE_M3_HYBRID.dense_vector_name == "dense"
    assert canonical.BGE_M3_HYBRID.sparse_vector_name == "sparse"
    assert canonical.BGE_RERANKER_V2_M3.batch_size == 32
    assert canonical.BGE_RERANKER_V2_M3.max_pairs_per_query == 50
    assert len(canonical.SWEEP_VARIANTS) == 9
    assert canonical.LONG_DOC_PASSAGES == 8

    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print(f"OK — all {len(public)} symbols identity-match between shim and canonical.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
