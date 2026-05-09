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


def _partial_path(out_path: Path) -> Path:
    return out_path.parent / (out_path.name + ".partial")


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
        try:
            f = os.fdopen(fd, "w", encoding="utf-8")
        except Exception:
            os.close(fd)
            raise
        with f:
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
    # Clean up any stale partial via the centralized helper
    _partial_path(out_path).unlink(missing_ok=True)


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
        try:
            f = os.fdopen(fd, "w", encoding="utf-8")
        except Exception:
            os.close(fd)
            raise
        with f:
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
        data = json.loads(pp.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        # Corrupt journal — log but don't crash. Build restarts from
        # scratch on next run.
        import logging
        logging.warning(
            f"load_partial: ignoring corrupt journal at {pp}: "
            f"{type(e).__name__}: {e}"
        )
        return []
    if not isinstance(data, dict):
        return []
    completed = data.get("clusters_completed", [])
    return completed if isinstance(completed, list) else []


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

    Defensive contract: returns empty-fields dict on ANY failure (LLM error,
    JSON parse fail, type drift). NEVER raises. NEVER retries — Task 7's
    orchestrator handles retry policy.

    Empty-fields signal: caller checks ``if not result["title"]:`` to detect
    failure. Tags list is element-coerced to strings.
    """
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
        # Coerce tags element-by-element — model can return [1, null, "..."]
        # and downstream tag consumers (entity index, search) require strings.
        tags = parsed.get("tags")
        if not isinstance(tags, list):
            parsed["tags"] = []
        else:
            parsed["tags"] = [t for t in tags if isinstance(t, str)]
        return parsed
    except Exception:
        return {"title": "", "summary": "", "tags": []}
