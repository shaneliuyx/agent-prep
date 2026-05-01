"""Schema integration test — creates throwaway collections in real Qdrant,
verifies the resulting schema, drops them.

Throwaway collections are namespaced ``parity_test_<feature>_<pid>`` so
parallel test runs don't collide and a crash mid-test leaves
recognizable cleanup targets.
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "shared"))

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance

from rag_hybrid.models import (
    BGE_M3,
    CollectionSpec,
    EmbedModelSpec,
    HybridCollectionSpec,
)
from rag_hybrid.schema import detect_collection_mode, ensure_collection


def _qd() -> QdrantClient:
    return QdrantClient(url=os.getenv("QDRANT_URL", "http://127.0.0.1:6333"), timeout=15)


def _unique(stem: str) -> str:
    """Stable per-process namespace; uuid4 segment guards against accidental
    cross-run collision."""
    return f"parity_test_{stem}_{os.getpid()}_{uuid.uuid4().hex[:6]}"


def test_dense_collection_creates_with_correct_schema() -> None:
    qd = _qd()
    name = _unique("dense")
    spec = CollectionSpec(name=name, model=BGE_M3)
    try:
        ensure_collection(qd, spec)
        info = qd.get_collection(name)
        # Single-vector schema → params has a `vectors` field, no sparse.
        sparse = getattr(info.config.params, "sparse_vectors", None)
        assert not sparse, f"dense collection has sparse config: {sparse}"
        assert detect_collection_mode(qd, name) == "dense"
    finally:
        qd.delete_collection(name)


def test_hybrid_collection_creates_with_correct_schema() -> None:
    qd = _qd()
    name = _unique("hybrid")
    spec = HybridCollectionSpec(name=name, model=BGE_M3)
    try:
        ensure_collection(qd, spec)
        info = qd.get_collection(name)
        sparse = getattr(info.config.params, "sparse_vectors", None)
        assert sparse, f"hybrid collection missing sparse config: params={info.config.params}"
        assert detect_collection_mode(qd, name) == "hybrid"
    finally:
        qd.delete_collection(name)


def test_hybrid_rejects_non_sparse_model() -> None:
    qd = _qd()
    fake = EmbedModelSpec(
        name="fake", path="/tmp/none", dim=384, distance=Distance.COSINE,
        supports_sparse=False,
    )
    spec = HybridCollectionSpec(name=_unique("guard"), model=fake)
    try:
        ensure_collection(qd, spec)
    except ValueError as e:
        assert "supports_sparse=False" in str(e), f"unexpected error message: {e}"
        return
    raise AssertionError("ensure_collection should have raised ValueError")


def test_unsupported_spec_type_rejected() -> None:
    qd = _qd()
    try:
        ensure_collection(qd, "not a spec")  # type: ignore[arg-type]
    except TypeError as e:
        assert "unsupported spec type" in str(e)
        return
    raise AssertionError("ensure_collection should have raised TypeError")


def main() -> int:
    failures = []
    tests = [
        ("test_dense_collection_creates_with_correct_schema",
         test_dense_collection_creates_with_correct_schema),
        ("test_hybrid_collection_creates_with_correct_schema",
         test_hybrid_collection_creates_with_correct_schema),
        ("test_hybrid_rejects_non_sparse_model", test_hybrid_rejects_non_sparse_model),
        ("test_unsupported_spec_type_rejected", test_unsupported_spec_type_rejected),
    ]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS: {name}")
        except AssertionError as e:
            failures.append(f"{name}: {e}")
            print(f"FAIL: {name}: {e}")
        except Exception as e:
            failures.append(f"{name}: {type(e).__name__}: {e}")
            print(f"ERROR: {name}: {type(e).__name__}: {e}")
    if failures:
        return 1
    print(f"OK — {len(tests)} schema tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
