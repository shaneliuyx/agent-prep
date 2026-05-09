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
