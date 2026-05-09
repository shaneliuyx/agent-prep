# Summary Index Tree (RAPTOR Level-2) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Level-2 summary index tree (RAPTOR pattern) on top of the existing primary tree.json — clusters primary nodes by embedding similarity, generates LLM cluster summaries with verbatim entity preservation, and routes cross-section synthesis questions through cluster-first retrieval.

**Architecture:** Build phase: embed primary summaries → k-means cluster → LLM-generate cluster titles+summaries+tags → persist `summary_index.json` keyed to `tree_hash`. Query phase: detect synthesis pattern → embed question → cosine vs cluster summaries → batch-fetch all member node pages → synthesize. Falls back to entity-prefetch if cluster confidence below threshold.

**Tech Stack:** Python 3.11, `numpy`/`scikit-learn` for k-means, BGE-M3 embeddings (reuse from `lab-02-3-bge_m3_hnsw`), OpenAI-compatible client (vMLX), Phoenix tracing.

---

## File Structure

```
shared/tree_index/
  summary_index.py         # NEW — SummaryIndex class
  agentic.py               # MODIFY — _CLUSTER_TOOL, _find_cluster, prefetch
  prompts.py               # MODIFY — AGENTIC_SYSTEM_TEMPLATE_V2 Rule 4
  __init__.py              # MODIFY — export SummaryIndex

lab-02-7-pageindex/
  src/build_summary_index.py   # NEW — build script with retry+resume
  data/summary_index.json      # NEW — build artifact (gitignored or LFS)
  scripts/run_one_variant.py   # MODIFY — wire SummaryIndex into v2 retriever

tests/
  test_summary_index.py        # NEW — unit tests for SummaryIndex
  test_build_summary_index.py  # NEW — build atomicity + resume tests
  test_cluster_prefetch.py     # NEW — integration tests
```

**File responsibilities:**
- `summary_index.py` — load + validate cluster index, expose `find_cluster_for_query`, encapsulate tree_hash check
- `build_summary_index.py` — build artifact via embedding + kmeans + LLM labels, atomic write, resume from `.partial`
- `agentic.py` — query-time integration: tool schema, dispatch, pre-fetch hint injection
- `prompts.py` — V2 prompt rule for cluster-first routing
- Test files — Given/When/Then scenarios from spec §8

---

## Task 1: Tree-Hash Utility + Tests

**Files:**
- Create: `shared/tree_index/_hashing.py`
- Test: `tests/test_summary_index.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_summary_index.py
import json
import pytest
from pathlib import Path
from tree_index._hashing import tree_hash


def test_tree_hash_stable_across_whitespace(tmp_path: Path) -> None:
    """Same content + different formatting → same hash."""
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text('{"node_id":"0001","title":"X"}')
    b.write_text('{\n  "node_id": "0001",\n  "title": "X"\n}')
    assert tree_hash(a) == tree_hash(b)


def test_tree_hash_changes_on_content_change(tmp_path: Path) -> None:
    """Different content → different hash."""
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text('{"node_id":"0001","title":"X"}')
    b.write_text('{"node_id":"0001","title":"Y"}')
    assert tree_hash(a) != tree_hash(b)


def test_tree_hash_independent_of_key_order(tmp_path: Path) -> None:
    """Same content + different key order → same hash."""
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text('{"node_id":"0001","title":"X"}')
    b.write_text('{"title":"X","node_id":"0001"}')
    assert tree_hash(a) == tree_hash(b)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/yuxinliu/code/agent-prep && /Users/yuxinliu/code/agent-prep/.venv/bin/pytest tests/test_summary_index.py -v -k tree_hash`
Expected: FAIL with `ModuleNotFoundError: No module named 'tree_index._hashing'`

- [ ] **Step 3: Implement `_hashing.py`**

```python
# shared/tree_index/_hashing.py
"""Canonical hashing for tree.json — used to bind summary_index.json
to a specific primary tree state."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def tree_hash(tree_path: Path) -> str:
    """Compute sha256 of canonicalized tree.json.

    Canonical form: sort keys + minimal separators. Independent of
    insertion order, whitespace, indentation. Used to bind
    summary_index.json to the primary tree state at build time."""
    tree = json.loads(Path(tree_path).read_text(encoding="utf-8"))
    canonical = json.dumps(tree, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/yuxinliu/code/agent-prep && /Users/yuxinliu/code/agent-prep/.venv/bin/pytest tests/test_summary_index.py -v -k tree_hash`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
cd /Users/yuxinliu/code/agent-prep
git add shared/tree_index/_hashing.py tests/test_summary_index.py
git commit -m "feat(tree_index): add canonical tree_hash() for summary-index versioning"
```

---

## Task 2: SummaryIndex Class — Load + Validate

**Files:**
- Create: `shared/tree_index/summary_index.py`
- Modify: `tests/test_summary_index.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_summary_index.py`:

```python
from tree_index.summary_index import SummaryIndex


def _write_fixture(tmp_path: Path, tree_hash_value: str) -> tuple[Path, Path]:
    """Create a tree.json + summary_index.json pair for testing."""
    tree = tmp_path / "tree.json"
    tree.write_text('{"node_id":"0001","title":"X"}')
    idx = tmp_path / "summary_index.json"
    idx.write_text(json.dumps({
        "build_meta": {
            "tree_hash": tree_hash_value,
            "k": 2, "embedding_model": "BGE-M3",
            "created": "2026-05-09T01:00:00Z",
        },
        "clusters": [
            {"cluster_id": "C1", "title": "T1", "summary": "S1",
             "tags": ["a", "b"], "member_node_ids": ["0001"],
             "primary_pages": [[1, 2]]},
        ],
    }))
    return tree, idx


def test_summary_index_loads_when_hash_matches(tmp_path: Path) -> None:
    tree, idx = _write_fixture(tmp_path, tree_hash(tmp_path / "tree.json"))
    si = SummaryIndex(idx, tree)
    assert len(si.clusters) == 1
    assert si.clusters[0]["cluster_id"] == "C1"


def test_summary_index_raises_on_stale_hash(tmp_path: Path) -> None:
    tree, idx = _write_fixture(tmp_path, "abc123_wrong")
    with pytest.raises(RuntimeError, match="stale"):
        SummaryIndex(idx, tree)


def test_summary_index_missing_file_raises_with_helpful_message(
    tmp_path: Path,
) -> None:
    tree = tmp_path / "tree.json"
    tree.write_text('{"node_id":"0001"}')
    with pytest.raises(FileNotFoundError, match="build_summary_index"):
        SummaryIndex(tmp_path / "missing.json", tree)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/yuxinliu/code/agent-prep && /Users/yuxinliu/code/agent-prep/.venv/bin/pytest tests/test_summary_index.py -v -k SummaryIndex`
Expected: FAIL with `ImportError: cannot import name 'SummaryIndex'`

- [ ] **Step 3: Implement `SummaryIndex` (load + validate only)**

```python
# shared/tree_index/summary_index.py
"""SummaryIndex — Level-2 RAPTOR layer over the primary tree-index.

Encapsulates the cluster index produced by build_summary_index.py:
  - Validate tree_hash binding at load time (fail fast on stale index)
  - Expose find_cluster_for_query(q) for cluster-first routing
  - Provide cluster metadata for pre-fetch hint injection

Load discipline: SummaryIndex(idx_path, tree_path) raises immediately if
the index is stale, missing, or malformed. Caller (agentic.py) catches
and falls back to entity-prefetch — never crashes the user query."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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

        meta = data.get("build_meta", {})
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/yuxinliu/code/agent-prep && /Users/yuxinliu/code/agent-prep/.venv/bin/pytest tests/test_summary_index.py -v`
Expected: PASS (6 tests total — 3 from Task 1 + 3 new)

- [ ] **Step 5: Commit**

```bash
cd /Users/yuxinliu/code/agent-prep
git add shared/tree_index/summary_index.py tests/test_summary_index.py
git commit -m "feat(tree_index): SummaryIndex class with tree_hash validation"
```

---

## Task 3: SummaryIndex.find_cluster_for_query — Cosine Lookup

**Files:**
- Modify: `shared/tree_index/summary_index.py`
- Modify: `tests/test_summary_index.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_summary_index.py`:

```python
import numpy as np


def test_find_cluster_for_query_returns_top_cluster(
    tmp_path: Path, monkeypatch
) -> None:
    """Mock embedder returns deterministic vectors; verify cosine pick."""
    tree = tmp_path / "tree.json"
    tree.write_text('{"node_id":"0001"}')
    idx = tmp_path / "summary_index.json"
    idx.write_text(json.dumps({
        "build_meta": {
            "tree_hash": tree_hash(tree),
            "k": 2, "embedding_model": "BGE-M3",
            "cluster_embeddings": [
                [1.0, 0.0, 0.0],   # C1 — aligned with 'investments'
                [0.0, 1.0, 0.0],   # C2 — aligned with 'cybersecurity'
            ],
        },
        "clusters": [
            {"cluster_id": "C1", "title": "Investments", "summary": "...",
             "tags": [], "member_node_ids": ["0001"],
             "primary_pages": [[1, 2]]},
            {"cluster_id": "C2", "title": "Cybersecurity", "summary": "...",
             "tags": [], "member_node_ids": ["0001"],
             "primary_pages": [[3, 4]]},
        ],
    }))
    si = SummaryIndex(idx, tree)
    # Inject a fake embedder that maps "investments" → C1 vec
    si.set_embedder(lambda text: np.array([1.0, 0.0, 0.0]))
    hit = si.find_cluster_for_query("Buffett's investments")
    assert hit is not None
    assert hit["cluster"]["cluster_id"] == "C1"
    assert hit["confidence"] >= 0.9


def test_find_cluster_returns_none_below_threshold(
    tmp_path: Path,
) -> None:
    tree = tmp_path / "tree.json"
    tree.write_text('{"node_id":"0001"}')
    idx = tmp_path / "summary_index.json"
    idx.write_text(json.dumps({
        "build_meta": {
            "tree_hash": tree_hash(tree),
            "cluster_embeddings": [[1.0, 0.0]],
        },
        "clusters": [{"cluster_id": "C1", "title": "X", "summary": "",
                      "tags": [], "member_node_ids": ["0001"],
                      "primary_pages": [[1, 2]]}],
    }))
    si = SummaryIndex(idx, tree)
    si.set_embedder(lambda text: np.array([0.0, 1.0]))   # orthogonal
    hit = si.find_cluster_for_query("anything", threshold=0.5)
    assert hit is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/yuxinliu/code/agent-prep && /Users/yuxinliu/code/agent-prep/.venv/bin/pytest tests/test_summary_index.py -v -k find_cluster`
Expected: FAIL with `AttributeError: 'SummaryIndex' object has no attribute 'set_embedder'`

- [ ] **Step 3: Add embedder + find_cluster_for_query to SummaryIndex**

Append to `shared/tree_index/summary_index.py`:

```python
import numpy as np
from typing import Callable, Optional


class SummaryIndex:
    # ... existing __init__ ...

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/yuxinliu/code/agent-prep && /Users/yuxinliu/code/agent-prep/.venv/bin/pytest tests/test_summary_index.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
cd /Users/yuxinliu/code/agent-prep
git add shared/tree_index/summary_index.py tests/test_summary_index.py
git commit -m "feat(tree_index): cosine-similarity find_cluster_for_query with threshold"
```

---

## Task 4: Build Script Skeleton — Embed + K-Means

**Files:**
- Create: `lab-02-7-pageindex/src/build_summary_index.py`
- Create: `tests/test_build_summary_index.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_build_summary_index.py
import json
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def fixture_tree(tmp_path: Path) -> Path:
    """Build a 6-node primary tree fixture for clustering."""
    tree = {
        "node_id": "root", "title": "Root", "nodes": [
            {"node_id": f"00{i+1:02d}",
             "title": f"Section {i}",
             "summary": f"Topic {chr(ord('A') + i % 3)}",
             "tags": [f"tag_{i}"],
             "start_page": i * 2 + 1, "end_page": i * 2 + 2}
            for i in range(6)
        ],
    }
    p = tmp_path / "tree.json"
    p.write_text(json.dumps(tree))
    return p


def test_extract_leaves_returns_all_summary_nodes(fixture_tree: Path) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                          / "lab-02-7-pageindex" / "src"))
    from build_summary_index import extract_leaves
    tree = json.loads(fixture_tree.read_text())
    leaves = extract_leaves(tree)
    assert len(leaves) == 6
    assert all("node_id" in n and "summary" in n for n in leaves)


def test_kmeans_groups_summaries_deterministically() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                          / "lab-02-7-pageindex" / "src"))
    from build_summary_index import kmeans_cluster
    # 6 vectors in 3 clear groups
    embeddings = np.array([
        [1.0, 0.0], [1.0, 0.1], [1.0, -0.1],
        [0.0, 1.0], [0.1, 1.0], [-0.1, 1.0],
    ])
    labels_a = kmeans_cluster(embeddings, k=2, random_state=42)
    labels_b = kmeans_cluster(embeddings, k=2, random_state=42)
    assert (labels_a == labels_b).all()
    assert len(set(labels_a)) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/yuxinliu/code/agent-prep && /Users/yuxinliu/code/agent-prep/.venv/bin/pytest tests/test_build_summary_index.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'build_summary_index'`

- [ ] **Step 3: Create skeleton with extract_leaves + kmeans_cluster**

```python
# lab-02-7-pageindex/src/build_summary_index.py
"""Build the Level-2 summary index from data/tree.json.

Pipeline:
  1. Extract leaf summaries from primary tree
  2. Embed via BGE-M3 (or injected embedder)
  3. K-means cluster (auto-K via silhouette)
  4. LLM-generate cluster title + summary + tags per cluster
  5. Atomic write to data/summary_index.json with tree_hash binding

Resume: per-cluster journaling to .partial; on restart skip completed
clusters. Idempotent given fixed random_state."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_LAB_ROOT.parents[0] / "shared"))

from tree_index._hashing import tree_hash  # noqa: E402


def extract_leaves(tree: dict) -> list[dict]:
    """Walk the primary tree, return all nodes with non-empty summary."""
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
    """Deterministic k-means with sklearn. Returns cluster labels."""
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    return km.fit_predict(embeddings)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/yuxinliu/code/agent-prep && /Users/yuxinliu/code/agent-prep/.venv/bin/pytest tests/test_build_summary_index.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
cd /Users/yuxinliu/code/agent-prep
git add lab-02-7-pageindex/src/build_summary_index.py tests/test_build_summary_index.py
git commit -m "feat(lab-02-7): build_summary_index skeleton — extract_leaves + kmeans"
```

---

## Task 5: Build Script — LLM Cluster Labeling + Atomic Write

**Files:**
- Modify: `lab-02-7-pageindex/src/build_summary_index.py`
- Modify: `tests/test_build_summary_index.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_build_summary_index.py`:

```python
def test_atomic_write_creates_partial_then_commits(
    fixture_tree: Path, tmp_path: Path,
) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                          / "lab-02-7-pageindex" / "src"))
    from build_summary_index import write_atomic

    out = tmp_path / "summary_index.json"
    payload = {"clusters": [{"cluster_id": "C1"}]}
    write_atomic(out, payload)

    assert out.exists()
    assert json.loads(out.read_text()) == payload
    # .partial should be cleaned up
    assert not (out.parent / (out.name + ".partial")).exists()


def test_journal_partial_persists_completed_clusters(tmp_path: Path) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                          / "lab-02-7-pageindex" / "src"))
    from build_summary_index import journal_partial, load_partial

    out = tmp_path / "summary_index.json"
    journal_partial(out, [
        {"cluster_id": "C1", "title": "T1", "summary": "S1",
         "tags": [], "member_node_ids": ["0001"], "primary_pages": [[1, 2]]},
    ])
    completed = load_partial(out)
    assert {c["cluster_id"] for c in completed} == {"C1"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/yuxinliu/code/agent-prep && /Users/yuxinliu/code/agent-prep/.venv/bin/pytest tests/test_build_summary_index.py -v -k "atomic or partial"`
Expected: FAIL with `ImportError: cannot import name 'write_atomic'`

- [ ] **Step 3: Add atomic write + journal helpers**

Append to `lab-02-7-pageindex/src/build_summary_index.py`:

```python
import os
import tempfile
import time


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/yuxinliu/code/agent-prep && /Users/yuxinliu/code/agent-prep/.venv/bin/pytest tests/test_build_summary_index.py -v`
Expected: PASS (4 tests total)

- [ ] **Step 5: Commit**

```bash
cd /Users/yuxinliu/code/agent-prep
git add lab-02-7-pageindex/src/build_summary_index.py tests/test_build_summary_index.py
git commit -m "feat(lab-02-7): build_summary_index atomic write + per-cluster journaling"
```

---

## Task 6: Build Script — LLM Cluster Summarizer

**Files:**
- Modify: `lab-02-7-pageindex/src/build_summary_index.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_build_summary_index.py`:

```python
def test_summarize_cluster_returns_required_fields(monkeypatch) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                          / "lab-02-7-pageindex" / "src"))
    from build_summary_index import summarize_cluster

    member_summaries = [
        "Buffett discusses Coca-Cola, American Express ownership.",
        "Berkshire holds 27.8% of Occidental Petroleum.",
    ]

    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    class M: content = json.dumps({
                        "title": "Non-controlled investments",
                        "summary": ("Discusses Coca-Cola, American Express, "
                                   "and 27.8% Occidental Petroleum stakes."),
                        "tags": ["Coca-Cola", "American Express",
                                 "Occidental", "27.8%"],
                    })
                    class C: message = M()
                    class R: choices = [C()]
                    return R()

    out = summarize_cluster(FakeClient(), "test-model", member_summaries)
    assert out["title"]
    assert len(out["summary"]) >= 30
    assert len(out["tags"]) >= 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/yuxinliu/code/agent-prep && /Users/yuxinliu/code/agent-prep/.venv/bin/pytest tests/test_build_summary_index.py::test_summarize_cluster_returns_required_fields -v`
Expected: FAIL with `ImportError: cannot import name 'summarize_cluster'`

- [ ] **Step 3: Implement summarize_cluster**

Append to `lab-02-7-pageindex/src/build_summary_index.py`:

```python
_CLUSTER_SUMMARIZE_SYSTEM = """You receive 2-10 document section summaries
that share a thematic cluster. Generate a SINGLE cluster meta-summary that
preserves verbatim entities + quoted phrases from the inputs.

OUTPUT: strict JSON with exactly these keys:
  {
    "title":   "<3-8 word cluster theme — distinctive phrase from inputs>",
    "summary": "<100-180 word prose summary covering ALL member sections.
                 PRESERVE: every named entity verbatim (Coca-Cola, Itochu,
                 BNSF), every distinctive quoted phrase ('not-so-secret
                 weapon', 'patience pays'), every numeric fact with units
                 ($364.5 billion, 27.8%). Do NOT paraphrase distinctive
                 vocabulary>",
    "tags":    ["<15-30 lookup tokens — entities, aliases, numeric anchors,
                 quoted phrases>"]
  }

Output ONLY this JSON. No prose preamble."""


def summarize_cluster(client, model: str,
                      member_summaries: list[str]) -> dict:
    """One LLM call → cluster {title, summary, tags}.

    Defensive: returns empty-fields dict on JSON parse failure or
    LLM error. Caller decides whether to retry or use fallback."""
    user = "\n\n---\n\n".join(member_summaries)
    try:
        r = client.chat.completions.create(
            model=model, temperature=0.0, max_tokens=500,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _CLUSTER_SUMMARIZE_SYSTEM},
                {"role": "user", "content": user},
            ],
        )
        raw = (r.choices[0].message.content or "{}").strip()
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            return {"title": "", "summary": "", "tags": []}
        for k in ("title", "summary"):
            if not isinstance(parsed.get(k), str):
                parsed[k] = ""
        if not isinstance(parsed.get("tags"), list):
            parsed["tags"] = []
        return parsed
    except Exception:
        return {"title": "", "summary": "", "tags": []}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/yuxinliu/code/agent-prep && /Users/yuxinliu/code/agent-prep/.venv/bin/pytest tests/test_build_summary_index.py::test_summarize_cluster_returns_required_fields -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/yuxinliu/code/agent-prep
git add lab-02-7-pageindex/src/build_summary_index.py tests/test_build_summary_index.py
git commit -m "feat(lab-02-7): build_summary_index LLM cluster labeler with verbatim preservation"
```

---

## Task 7: Build Script — Main Orchestrator with Resume

**Files:**
- Modify: `lab-02-7-pageindex/src/build_summary_index.py`

- [ ] **Step 1: Add main() with resume logic + CLI**

Append to `lab-02-7-pageindex/src/build_summary_index.py`:

```python
import argparse
import os

from dotenv import load_dotenv
from openai import OpenAI


def _embed_summaries(summaries: list[str]) -> "np.ndarray":
    """BGE-M3 embedding — reuses lab-02-3-bge_m3_hnsw infrastructure.

    Lazy import: only loads the BGE-M3 model when build runs."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-m3", device="mps")
    return np.asarray(
        model.encode(summaries, batch_size=8, normalize_embeddings=True,
                     show_progress_bar=False),
        dtype=np.float32,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the Level-2 summary index over data/tree.json"
    )
    parser.add_argument("--force", action="store_true",
                        help="Ignore .partial and rebuild from scratch")
    parser.add_argument("--k", type=int, default=8,
                        help="Number of clusters (default: 8)")
    parser.add_argument("--check", action="store_true",
                        help="Verify tree_hash match without rebuilding; "
                             "exit 0 if fresh, 1 if stale")
    parser.add_argument("--output", type=str,
                        default=str(_LAB_ROOT / "data" / "summary_index.json"),
                        help="Output path")
    args = parser.parse_args()

    out_path = Path(args.output)
    tree_path = _LAB_ROOT / "data" / "tree.json"

    # --check mode
    if args.check:
        if not out_path.exists():
            print(f"NOT FRESH: {out_path} does not exist")
            sys.exit(1)
        try:
            existing = json.loads(out_path.read_text())
            stored = existing.get("build_meta", {}).get("tree_hash", "")
            actual = tree_hash(tree_path)
            if stored != actual:
                print(f"STALE: tree_hash mismatch ({stored[:12]} != {actual[:12]})")
                sys.exit(1)
            print(f"FRESH: tree_hash {actual[:12]} matches")
            sys.exit(0)
        except Exception as e:
            print(f"NOT FRESH: {e}")
            sys.exit(1)

    load_dotenv(_LAB_ROOT / ".env")
    client = OpenAI(base_url=os.getenv("OMLX_BASE_URL"),
                    api_key=os.getenv("OMLX_API_KEY"))
    model = os.getenv("MODEL_BUILD") or os.getenv("MODEL_SONNET") or ""
    if not model:
        print("FATAL: MODEL_BUILD / MODEL_SONNET unset", file=sys.stderr)
        sys.exit(2)

    # Skip-if-fresh
    if not args.force and out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
            if existing.get("build_meta", {}).get("tree_hash", "") == tree_hash(tree_path):
                print(f"summary_index up to date — skip rebuild "
                      f"(tree_hash {tree_hash(tree_path)[:12]})")
                return
        except Exception:
            pass

    # Load tree + leaves
    tree = json.loads(tree_path.read_text(encoding="utf-8"))
    leaves = extract_leaves(tree)
    print(f"[1/4] {len(leaves)} primary leaves loaded", flush=True)

    # Embed
    print(f"[2/4] Embedding leaf summaries via BGE-M3 ...", flush=True)
    embeddings = _embed_summaries([n["summary"] for n in leaves])

    # Cluster
    k = min(args.k, len(leaves) - 1)
    labels = kmeans_cluster(embeddings, k=k, random_state=42)
    print(f"[3/4] K-means k={k} → {len(set(labels))} clusters", flush=True)

    # Compute cluster centroids (used as cluster_embeddings)
    centroids = np.zeros((k, embeddings.shape[1]), dtype=np.float32)
    for i in range(k):
        mask = labels == i
        if mask.any():
            centroids[i] = embeddings[mask].mean(axis=0)

    # Resume from .partial if present (and not --force)
    completed = [] if args.force else load_partial(out_path)
    completed_ids = {c["cluster_id"] for c in completed}
    print(f"[4/4] Generating cluster summaries "
          f"(resume: {len(completed_ids)}/{k} already done)", flush=True)

    clusters: list[dict] = list(completed)
    for ci in range(k):
        cid = f"C{ci+1}"
        if cid in completed_ids:
            continue
        member_idx = np.where(labels == ci)[0]
        member_summaries = [leaves[j]["summary"] for j in member_idx]
        meta = summarize_cluster(client, model, member_summaries)
        cluster_obj = {
            "cluster_id": cid,
            "title": meta["title"] or f"Cluster {cid}",
            "summary": meta["summary"],
            "tags": meta["tags"],
            "member_node_ids": [leaves[j]["node_id"] for j in member_idx],
            "primary_pages": [
                [leaves[j]["start_page"], leaves[j]["end_page"]]
                for j in member_idx
            ],
        }
        clusters.append(cluster_obj)
        journal_partial(out_path, clusters)
        print(f"    cluster {cid} ({len(member_idx)} members): "
              f"{cluster_obj['title']!r}", flush=True)

    # Final atomic write
    payload = {
        "build_meta": {
            "tree_hash": tree_hash(tree_path),
            "k": k,
            "embedding_model": "BGE-M3",
            "cluster_embeddings": centroids.tolist(),
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "clusters": clusters,
    }
    write_atomic(out_path, payload)
    print(f"\nWrote {out_path} — {k} clusters", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the build pipeline**

Run: `cd /Users/yuxinliu/code/agent-prep/lab-02-7-pageindex && /Users/yuxinliu/code/agent-prep/.venv/bin/python src/build_summary_index.py --check`
Expected: `NOT FRESH: data/summary_index.json does not exist` + exit code 1

- [ ] **Step 3: Build for real (this will take ~3-5 min)**

Run: `cd /Users/yuxinliu/code/agent-prep/lab-02-7-pageindex && /Users/yuxinliu/code/agent-prep/.venv/bin/python src/build_summary_index.py 2>&1 | tee /tmp/build_summary_index.log`
Expected: prints `[1/4]`, `[2/4]`, `[3/4]`, `[4/4]`, then `Wrote data/summary_index.json — 8 clusters`

- [ ] **Step 4: Verify output structure**

Run: `cd /Users/yuxinliu/code/agent-prep && /Users/yuxinliu/code/agent-prep/.venv/bin/python -c "
import json
d = json.loads(open('lab-02-7-pageindex/data/summary_index.json').read())
print(f'k={d[\"build_meta\"][\"k\"]}, clusters={len(d[\"clusters\"])}')
print(f'tree_hash={d[\"build_meta\"][\"tree_hash\"][:12]}')
for c in d['clusters'][:3]:
    print(f'  {c[\"cluster_id\"]}: {c[\"title\"]!r} | {len(c[\"tags\"])} tags | {len(c[\"member_node_ids\"])} members')
"`
Expected: 8 clusters with non-empty titles + tags + members

- [ ] **Step 5: Commit**

```bash
cd /Users/yuxinliu/code/agent-prep
git add lab-02-7-pageindex/src/build_summary_index.py lab-02-7-pageindex/data/summary_index.json
git commit -m "feat(lab-02-7): summary_index build orchestrator with resume + CLI"
```

---

## Task 8: New Tool Schema in agentic.py

**Files:**
- Modify: `shared/tree_index/agentic.py`

- [ ] **Step 1: Add _CLUSTER_TOOL schema**

Locate `_V2_TOOLS` list in `agentic.py` (around line 100-130). Add immediately after it:

```python
_CLUSTER_TOOL = {
    "type": "function",
    "function": {
        "name": "find_cluster_for_synthesis",
        "description": (
            "Cluster-first lookup for cross-section synthesis questions "
            "('what did X say/write about Y'). Returns one thematic cluster "
            "with member node_ids + page ranges. Use BEFORE get_page_content "
            "when the question spans multiple sub-sections — one batched "
            "fetch over all member pages is more efficient than sequential "
            "single-node fetches."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "the user's question or topic"},
            },
            "required": ["query"],
        },
    },
}
```

- [ ] **Step 2: Wire into _tools list when summary_index is provided**

Modify `AgenticTreeRetriever.__init__`:

```python
def __init__(
    self, *, ...,
    summary_index=None,    # NEW — Optional[SummaryIndex]
    ...
) -> None:
    # ... existing fields ...
    self.summary_index = summary_index
    self._tools = list(_DEFAULT_TOOLS)
    if tree_index is not None:
        self._tools.append(_V2_TOOLS[0])
    if entity_index is not None:
        self._tools.append(_V2_TOOLS[1])
    if summary_index is not None:
        self._tools.append(_CLUSTER_TOOL)
```

- [ ] **Step 3: Verify it parses + commits without test failures**

Run: `cd /Users/yuxinliu/code/agent-prep && /Users/yuxinliu/code/agent-prep/.venv/bin/python -c "
from tree_index.agentic import AgenticTreeRetriever, _CLUSTER_TOOL
print(_CLUSTER_TOOL['function']['name'])
"`
Expected: `find_cluster_for_synthesis`

- [ ] **Step 4: Commit**

```bash
git add shared/tree_index/agentic.py
git commit -m "feat(tree_index): add _CLUSTER_TOOL schema to v2 agent"
```

---

## Task 9: Dispatch + Pre-fetch in agentic.py

**Files:**
- Modify: `shared/tree_index/agentic.py`

- [ ] **Step 1: Add _find_cluster method**

Insert near other _find/_fetch methods:

```python
def _find_cluster(self, query: str) -> str:
    """CLUSTER-FIRST LOOKUP: returns cluster summary + member pages."""
    if self.summary_index is None:
        return "[ERROR] find_cluster_for_synthesis requires summary_index"
    threshold = float(__import__("os").getenv(
        "SUMMARY_INDEX_THRESHOLD", "0.5"))
    hit = self.summary_index.find_cluster_for_query(query, threshold=threshold)
    if hit is None:
        return f"No cluster matches {query!r} above threshold {threshold:.2f}"
    c = hit["cluster"]
    pages = c.get("primary_pages", [])
    pages_str = ", ".join(f"[{p[0]}-{p[1]}]" for p in pages)
    return (f"Cluster {c['cluster_id']!r}: {c['title']}\n"
            f"  confidence: {hit['confidence']:.2f}\n"
            f"  member_node_ids: {c['member_node_ids']}\n"
            f"  primary_pages: {pages_str}\n"
            f"  summary: {c['summary'][:300]}\n"
            f"  tags: {c.get('tags', [])[:15]}\n"
            f"NEXT: call get_page_content with the page range covering "
            f"member_node_ids, OR fetch each range and synthesize.")
```

- [ ] **Step 2: Add cluster pre-fetch in answer()**

In `answer()`, after the entity-prefetch hint construction, ADD a parallel cluster-prefetch:

```python
# Cluster-prefetch — for synthesis-pattern queries, pre-fire
# find_cluster_for_synthesis BEFORE first LLM call. Routes
# multi-section synthesis through one batched fetch instead of
# sequential per-node fetches that hit max_iter cliff.
cluster_hint = ""
if (self.summary_index is not None and is_synthesis
        and __import__("os").getenv("SUMMARY_INDEX_ENABLED", "1") != "0"):
    body = self._find_cluster(query)
    if not body.startswith("No cluster") and not body.startswith("[ERROR]"):
        cluster_hint = (
            f"\n\nCLUSTER HINT (auto-fired before your first call): "
            f"{body}\n\nUse the page ranges from this cluster directly "
            f"with get_page_content."
        )
```

- [ ] **Step 3: Inject cluster_hint into user message**

Modify the existing user-message construction to include both `entity_hint` and `cluster_hint`:

```python
msgs: list[dict] = [
    {"role": "system", "content": f"{_nonce}\n{self.system_prompt}"},
    {"role": "user", "content": (
        f"{_nonce}\n"
        f"Document tree:\n{tree_str}{entity_hint}{cluster_hint}\n\n"
        f"Question: {query}")},
]
```

- [ ] **Step 4: Add tool dispatch case**

In the tool-execution loop (around `if name == "get_page_content"`), add:

```python
elif name == "find_cluster_for_synthesis":
    q_arg = str(args.get("query", query))
    content = self._find_cluster(q_arg)
    tool_call_log.append({
        "iter": iteration, "tool": "find_cluster_for_synthesis",
        "args": {"query": q_arg},
        "content_chars": len(content),
    })
```

- [ ] **Step 5: Smoke-test that retriever still works**

Run: `cd /Users/yuxinliu/code/agent-prep/lab-02-7-pageindex && /Users/yuxinliu/code/agent-prep/.venv/bin/python -c "
from tree_index.agentic import AgenticTreeRetriever
print('agentic.py parses OK')
"`
Expected: `agentic.py parses OK`

- [ ] **Step 6: Commit**

```bash
git add shared/tree_index/agentic.py
git commit -m "feat(tree_index): cluster-prefetch + tool dispatch in v2 agent loop"
```

---

## Task 10: V2 Prompt Rule + Routing Heuristic

**Files:**
- Modify: `shared/tree_index/prompts.py`

- [ ] **Step 1: Add Rule 4 to V2 routing heuristic**

In `AGENTIC_SYSTEM_TEMPLATE_V2`, locate the `ROUTING HEURISTIC` block. Insert as Rule 0 (highest priority):

```python
# In prompts.py, modify AGENTIC_SYSTEM_TEMPLATE_V2 routing block:
# Insert AT THE TOP of the routing list:

  -1. **CLUSTER-FIRST FOR SYNTHESIS** (highest priority): For "what did
      X say/write about Y" or "how does X describe Y" questions where
      the topic likely spans multiple sub-sections, your FIRST
      (or zeroth) tool call MUST be find_cluster_for_synthesis. The
      cluster's member_node_ids + primary_pages tell you EXACTLY which
      pages to fetch. Call get_page_content on the FULL page range
      covering all member nodes (one batched fetch). Then synthesize.
      Skip this rule only when no cluster matches above threshold (the
      tool will tell you).
```

- [ ] **Step 2: Verify prompt parses**

Run: `cd /Users/yuxinliu/code/agent-prep && /Users/yuxinliu/code/agent-prep/.venv/bin/python -c "
from tree_index.prompts import AGENTIC_SYSTEM_TEMPLATE_V2
assert 'CLUSTER-FIRST FOR SYNTHESIS' in AGENTIC_SYSTEM_TEMPLATE_V2
print('OK')
"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add shared/tree_index/prompts.py
git commit -m "feat(tree_index): V2 prompt — cluster-first routing rule"
```

---

## Task 11: Wire SummaryIndex into run_one_variant.py

**Files:**
- Modify: `lab-02-7-pageindex/scripts/run_one_variant.py`

- [ ] **Step 1: Import + construct SummaryIndex**

Find where `EntityIndex` is constructed for v2/ensemble. Add SummaryIndex alongside:

```python
from tree_index.summary_index import SummaryIndex
```

In the v2 retriever construction block:

```python
# Try to load SummaryIndex; gracefully degrade if missing/stale
try:
    si = SummaryIndex(
        index_path=_LAB_ROOT / "data" / "summary_index.json",
        tree_path=_LAB_ROOT / "data" / "tree.json",
    )
    # Inject the BGE-M3 embedder reused from prompt_dev / lab-02-3
    from sentence_transformers import SentenceTransformer
    _bge = SentenceTransformer("BAAI/bge-m3", device="mps")
    si.set_embedder(lambda t: _bge.encode([t], normalize_embeddings=True)[0])
    print(f"[{variant}] SummaryIndex loaded: {len(si.clusters)} clusters",
          flush=True)
except (FileNotFoundError, RuntimeError, ValueError) as e:
    print(f"[{variant}] SummaryIndex unavailable: {type(e).__name__}: {e}",
          flush=True)
    si = None

retriever = AgenticTreeRetriever(
    tree=tree, page_provider=page_provider,
    model_client=omlx, model_name=model,
    system_prompt=AGENTIC_SYSTEM_TEMPLATE_V2,
    tree_index=ti, entity_index=ei,
    summary_index=si,    # NEW
)
```

- [ ] **Step 2: Smoke-test run_one_variant on Q-FACT only**

Run: `cd /Users/yuxinliu/code/agent-prep/lab-02-7-pageindex && /Users/yuxinliu/code/agent-prep/.venv/bin/python -c "
import json
import os, sys
sys.path.insert(0, 'src')
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / 'shared'))
from tree_index.summary_index import SummaryIndex
si = SummaryIndex(Path('data/summary_index.json'), Path('data/tree.json'))
print(f'Clusters: {len(si.clusters)}')
"`
Expected: `Clusters: 8`

- [ ] **Step 3: Commit**

```bash
git add lab-02-7-pageindex/scripts/run_one_variant.py
git commit -m "feat(lab-02-7): wire SummaryIndex into v2 retriever construction"
```

---

## Task 12: Export SummaryIndex + Run Full Eval

**Files:**
- Modify: `shared/tree_index/__init__.py`
- Run: full 16q eval

- [ ] **Step 1: Add export**

```python
# shared/tree_index/__init__.py — append to existing imports:
from .summary_index import SummaryIndex

# And to __all__:
__all__ = [
    # ... existing exports ...
    "SummaryIndex",
]
```

- [ ] **Step 2: Run 3-run eval per AC-Q1 protocol**

```bash
cd /Users/yuxinliu/code/agent-prep/lab-02-7-pageindex
for i in 1 2 3; do
    /Users/yuxinliu/code/agent-prep/.venv/bin/python scripts/run_one_variant.py v2 \
        > /tmp/cluster_eval_run$i.log 2>&1
    /Users/yuxinliu/code/agent-prep/.venv/bin/python -c "
import json
d = json.loads(open('results/ab_v2.json').read())
print(f'Run $i: judge={d[\"agg_judge\"]:.3f} lat={d[\"agg_lat\"]:.1f}s')
"
    cp results/ab_v2.json results/cluster_eval_run$i.json
    sleep 10
done
```
Expected: 3 runs each ~15-25 min, prints `judge=0.85+`

- [ ] **Step 3: Compute aggregate stats**

```bash
/Users/yuxinliu/code/agent-prep/.venv/bin/python -c "
import json
import statistics
runs = [json.loads(open(f'results/cluster_eval_run{i}.json').read()) for i in (1,2,3)]
judges = [r['agg_judge'] for r in runs]
print(f'mean={statistics.mean(judges):.3f} stdev={statistics.stdev(judges):.3f}')
print(f'min={min(judges):.3f} max={max(judges):.3f}')
"
```
Expected per AC-Q1: `mean >= 0.85 stdev <= 0.05`

- [ ] **Step 4: Verify acceptance criteria**

Manual check vs spec §13:
- AC-B1..B6: build artifact valid (Task 7 verified)
- AC-Q1..Q6: 3-run mean stats above
- AC-G1..G4: per-question check
- AC-O1..O4: env var smoke tests

- [ ] **Step 5: Commit**

```bash
git add shared/tree_index/__init__.py lab-02-7-pageindex/results/cluster_eval_run*.json
git commit -m "feat(tree_index): export SummaryIndex + 3-run eval validation"
```

---

## Task 13: Documentation + RESULTS.md Update

**Files:**
- Modify: `lab-02-7-pageindex/RESULTS.md`
- Modify: `Week 2.7 - Structure-Aware RAG.md` (Obsidian vault)

- [ ] **Step 1: Append cluster-index section to RESULTS.md**

Append a new `## Summary Index Tree (RAPTOR Level-2) — 2026-05-09` section with:
- TL;DR table (pre vs post cluster-prefetch judge / lat)
- Architecture diagram (copy from spec §6 mermaid)
- Per-question recovery table
- Build cost numbers
- New Bad-Case Journal entries 16-17 (any failures from Task 12 runs)

- [ ] **Step 2: Append Phase 7 section to W2.7 runbook**

Per the per-Python-block bundle pattern from CLAUDE.md, add:
- Phase 7 Block 1: cluster index architecture diagram
- Phase 7 Block 2: build_summary_index.py walkthrough + result + insight
- Phase 7 Block 3: query-time integration walkthrough + result + insight
- Phase 7 Bad-Case Journal entries

- [ ] **Step 3: Commit**

```bash
cd /Users/yuxinliu/code/agent-prep
git add lab-02-7-pageindex/RESULTS.md
cd "/Users/yuxinliu/Documents/Obsidian Vault/Agent Development Curriculum"
git add "Week 2.7 - Structure-Aware RAG.md"
git commit -m "docs(w2.7): summary-index-tree (RAPTOR Level-2) architecture + results"
```

---

## Self-Review Checklist

- [x] **Spec coverage:** §1 problem → covered Task 1-13. §3 build pipeline → Tasks 4-7. §4 query routing → Tasks 8-10. §5 components → all listed in File Structure. §6 data flow → covered. §7 errors → graceful fallback in Task 11. §8 tests → embedded in each task. §11 obs/ops → §11.1 Phoenix in Task 9 (env vars in Task 9, more in §11.2). §13 acceptance → Task 12 verifies.
- [x] **Placeholder scan:** No "TBD" / "TODO" / "implement later" found.
- [x] **Type consistency:** `find_cluster_for_query` used consistently across Tasks 3, 9, 11. `tree_hash` signature stable. `SummaryIndex(index_path, tree_path)` constructor signature stable across Task 2/3/11.
- [x] **Test scenarios mapped to spec §8 T-* IDs:** Task 1 → T-U2 (stale hash). Task 3 → T-U1 (find_cluster). Task 5 → T-U3 (deterministic) + T-I3 (resume). Task 12 → T-E1, T-E2, T-E3.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-05-09-summary-index-tree-implementation.md`.**

**Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration. Best for: most tasks are TDD with clear pass/fail criteria.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints. Best for: Tasks 7, 11, 12 require model + GPU access; you may want to watch them.

Which approach?
