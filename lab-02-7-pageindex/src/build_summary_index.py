"""Build the Level-2 summary index from data/tree.json.

Pipeline (across multiple tasks — Task 4 is skeleton):
  1. Extract leaf summaries from primary tree                   <- Task 4
  2. Embed via BGE-M3 (or injected embedder)                    <- Task 7
  3. K-means cluster (auto-K via silhouette)                    <- Task 4
  4. LLM-generate cluster title + summary + tags per cluster    <- Task 6
  5. Atomic write to data/summary_index.json with tree_hash     <- Task 5
                       binding

Resume: per-cluster journaling to .partial; on restart skip completed
clusters. Idempotent given fixed random_state."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

_LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_LAB_ROOT.parents[0] / "shared"))

from tree_index._hashing import tree_hash  # noqa: E402


def extract_summary_nodes(tree: dict) -> list[dict]:
    """Walk the tree, return all nodes with non-empty summary (root,
    internals, leaves alike). Used as input to clustering."""
    out: list[dict] = []

    def walk(node: dict) -> None:
        if node.get("summary"):
            out.append({
                "node_id": node["node_id"],
                "title": node.get("title", ""),
                "summary": node["summary"],
                "tags": node.get("tags", []),
                "start_page": node.get("start_page"),
                "end_page": node.get("end_page", node.get("start_page")),
            })
        for c in node.get("nodes", []):
            walk(c)

    walk(tree)
    return out


def kmeans_cluster(
    embeddings: "np.ndarray", k: int, random_state: int = 42,
) -> "np.ndarray":
    """Deterministic k-means with sklearn. Returns cluster labels.

    Raises ValueError if k > len(embeddings) — sklearn's own error
    is opaque in a multi-step pipeline."""
    if k > len(embeddings):
        raise ValueError(
            f"k={k} exceeds number of embeddings ({len(embeddings)}). "
            f"Reduce k or supply more leaf nodes."
        )
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    return km.fit_predict(embeddings)


def write_atomic(out_path: Path, payload: dict) -> None:
    """Write JSON atomically: temp file in same dir, fsync, rename."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=out_path.name + ".",
        suffix=".tmp",
        dir=str(out_path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, out_path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
    # Clean up any stale partial
    partial = out_path.parent / (out_path.name + ".partial")
    if partial.exists():
        partial.unlink()


def _partial_path(out_path: Path) -> Path:
    return out_path.parent / (out_path.name + ".partial")


def journal_partial(out_path: Path, clusters_so_far: list[dict]) -> None:
    """Per-cluster journal — atomic full-file rewrite of .partial."""
    payload = {"clusters_completed": clusters_so_far,
               "journal_ts": time.time()}
    pp = _partial_path(out_path)
    pp.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=pp.name + ".", suffix=".tmp", dir=str(pp.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, pp)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def load_partial(out_path: Path) -> list[dict]:
    """Return list of already-completed clusters from .partial, [] if none."""
    pp = _partial_path(out_path)
    if not pp.exists():
        return []
    try:
        return json.loads(pp.read_text(encoding="utf-8")).get("clusters_completed", [])
    except Exception:
        return []
