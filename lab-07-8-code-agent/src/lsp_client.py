"""Phase 2 — LSP integration via multilspy.

Cross-file index queries:
  find_references(symbol)       -> who calls foo?
  get_type(symbol)              -> what type does pyright infer?

Costs ~10-50ms per query vs ~1-5s for an LLM call. Pre-filter
structural questions through LSP; only invoke LLM after.
"""
from __future__ import annotations

from typing import Any


def open_pyright(repo_root: str):
    """Lazy import + initialize SyncLanguageServer.
    Avoid module-level import so the lab can still be syntax-checked
    without multilspy installed."""
    from multilspy import SyncLanguageServer
    from multilspy.multilspy_config import MultilspyConfig
    from multilspy.multilspy_logger import MultilspyLogger

    config = MultilspyConfig.from_dict({"code_language": "python"})
    return SyncLanguageServer.create(config, MultilspyLogger(), repo_root)


def find_callers(lsp, file: str, line: int, col: int) -> list[dict[str, Any]]:
    """Return all references to the symbol at (file, line, col)."""
    with lsp.start_server():
        refs = lsp.request_references(file, line, col)
    return [
        {"file": r.get("uri", ""), "line": r["range"]["start"]["line"]}
        for r in refs if "range" in r
    ]


def get_type(lsp, file: str, line: int, col: int) -> str | None:
    """Return inferred type at (file, line, col) via hover, or None."""
    with lsp.start_server():
        h = lsp.request_hover(file, line, col)
    if not h or not h.get("contents"):
        return None
    contents = h["contents"]
    if isinstance(contents, dict):
        return contents.get("value")
    return str(contents)
