"""SummaryIndex — Level-2 RAPTOR layer over the primary tree-index.

Encapsulates the cluster index produced by build_summary_index.py:
  - Validate tree_hash binding at load time (fail fast on stale index)
  - Expose find_cluster_for_query(q) for cluster-first routing  (Task 3)
  - Provide cluster metadata for pre-fetch hint injection

Load discipline: SummaryIndex(idx_path, tree_path) raises immediately if
the index is stale, missing, or malformed. Caller (agentic.py) catches
and falls back to entity-prefetch — never crashes the user query."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from ._hashing import tree_hash


class SummaryIndex:
    """Cluster index over the primary tree.

    Args:
        index_path: path to summary_index.json
        tree_path:  path to the primary tree.json the index was built from.
                    Used to validate tree_hash binding at construction time.

    Raises:
        FileNotFoundError: if index_path does not exist.
        RuntimeError:      if tree_hash mismatches (index is stale).
        ValueError:        if index_path is malformed JSON or missing fields.
    """

    def __init__(self, index_path: Path, tree_path: Path) -> None:
        self.index_path = Path(index_path)
        self.tree_path = Path(tree_path)
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"summary_index.json not found at {self.index_path}. "
                f"Run: python lab-02-7-pageindex/src/build_summary_index.py"
            )
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"summary_index malformed JSON: {e}") from e

        if "build_meta" not in data:
            raise ValueError(
                "summary_index missing required 'build_meta' field. "
                "File is malformed; rebuild via "
                "'python lab-02-7-pageindex/src/build_summary_index.py'."
            )
        meta = data["build_meta"]
        stored_hash = meta.get("tree_hash", "")
        actual_hash = tree_hash(self.tree_path)
        if stored_hash != actual_hash:
            raise RuntimeError(
                f"summary_index stale: tree_hash mismatch "
                f"(index has {stored_hash[:12]}, tree has {actual_hash[:12]}). "
                f"Rebuild via 'python lab-02-7-pageindex/src/build_summary_index.py'."
            )

        clusters = data.get("clusters", [])
        if not isinstance(clusters, list) or not clusters:
            raise ValueError("summary_index has no clusters")
        self.clusters: list[dict[str, Any]] = clusters
        self.build_meta: dict[str, Any] = meta

    def set_embedder(self, fn: Callable[[str], "np.ndarray"]) -> None:
        """Inject the query-time embedder. Default: caller wires BGE-M3.
        Tests inject deterministic mocks."""
        self._embedder = fn
        self._cluster_emb = np.array(
            self.build_meta.get("cluster_embeddings", []), dtype=np.float32
        )
        if self._cluster_emb.size == 0:
            raise ValueError(
                "summary_index missing cluster_embeddings — rebuild with "
                "current build_summary_index.py"
            )
        # Normalize cluster embeddings for cosine = dot product.
        norms = np.linalg.norm(self._cluster_emb, axis=1, keepdims=True)
        self._cluster_emb = self._cluster_emb / np.maximum(norms, 1e-8)

    def find_cluster_for_query(
        self, query: str, threshold: float = 0.5,
    ) -> Optional[dict]:
        """Return top-1 cluster match if cosine >= threshold, else None.

        Returns: {cluster: dict, confidence: float} or None.
        """
        if not hasattr(self, "_embedder"):
            raise RuntimeError(
                "set_embedder() must be called before find_cluster_for_query"
            )
        q_emb = self._embedder(query).astype(np.float32)
        n = float(np.linalg.norm(q_emb))
        if n < 1e-8:
            return None
        q_emb = q_emb / n
        scores = self._cluster_emb @ q_emb
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        if best_score < threshold:
            return None
        return {"cluster": self.clusters[best_idx], "confidence": best_score}
