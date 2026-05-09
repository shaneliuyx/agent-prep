"""Bootstrap shared/ onto sys.path so tests can `from tree_index import ...`
the same way lab-02-7-pageindex scripts do."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "shared"))
