"""shared/gbrain_cli.py — GBrain CLI wrappers + snippet-guarded context assembly.

Introduced by W3.5.96 (ground_truth_ab.py, answer_route_ab.py). GBrain-SPECIFIC — only
chapters that use GBrain as the memory layer need this (W3.5.95 greps local notes instead, so
it imports `llm` but not this). Reused by any GBrain chapter that retrieves slugs and reads
full page bodies for grounding.

Use from a consuming lab:
    import sys
    sys.path.insert(0, "/Users/yuxinliu/code/agent-prep/shared")
    from gbrain_cli import gbrain_get, gbrain_query_slugs, build_context, SnippetRegression
"""
from __future__ import annotations

import os
import re
import subprocess

_BUN = os.path.expanduser("~/.bun/bin")
_GBRAIN = os.getenv("GBRAIN_BIN") or os.path.join(_BUN, "gbrain")
_LINE = re.compile(r"^\[[-\d.]+\]\s+(\S+)\s+--")
_MIN_BODY_CHARS = 80   # a `gbrain get` page body; a `query --json` snippet is a short fragment


def server_env() -> dict[str, str]:
    """Env for shelling `gbrain` — puts ~/.bun/bin on PATH and defaults DB + embed endpoint."""
    env = dict(os.environ)
    env["PATH"] = _BUN + os.pathsep + env.get("PATH", "")
    env.setdefault("GBRAIN_DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/gbrain")
    env.setdefault("OLLAMA_BASE_URL", "http://localhost:8000/v1")
    return env


def _run(args: list[str]) -> str:
    return subprocess.run(args, capture_output=True, text=True, env=server_env()).stdout


def gbrain_query_slugs(q: str, limit: int) -> list[str]:
    """Hybrid retrieval — ranked slugs only. (`query --json` snippets are too thin to ground;
    fetch bodies with gbrain_get.)"""
    slugs: list[str] = []
    for line in _run([_GBRAIN, "query", q, "--json", "--limit", str(limit)]).splitlines():
        m = _LINE.match(line.strip())
        if m:
            slugs.append(m.group(1))
    return slugs


def gbrain_get(slug: str) -> str:
    """Full page body via `gbrain get <slug>` — NOT the truncated `query --json` snippet."""
    body = _run([_GBRAIN, "get", slug])
    return "\n".join(ln for ln in body.splitlines() if not ln.startswith(("Starting", "[gbrain")))


class SnippetRegression(RuntimeError):
    """A reader injected a truncated `query --json` snippet instead of a full `gbrain get` body —
    the failure mode the lab fixed once and must not regress to."""


def build_context(slugs: list[str], max_body_chars: int = 0,
                  min_body_chars: int = _MIN_BODY_CHARS) -> str:
    """Assemble reader context from FULL `gbrain get` bodies (never `query --json` snippets).
    Guards the seam: raises SnippetRegression (fail loud) if any injected body is suspiciously
    short. `max_body_chars > 0` caps each body for a small-context generator (e.g. the local 14B
    chokes on ~70K-token 10-K sections)."""
    bodies = [gbrain_get(s) for s in slugs]
    if max_body_chars:
        bodies = [b[:max_body_chars] for b in bodies]
    short = [(s, len(b.strip())) for s, b in zip(slugs, bodies) if len(b.strip()) < min_body_chars]
    if short:
        raise SnippetRegression(
            f"injected body too short {short} — reader must pull full `gbrain get` bodies, "
            f"not `query --json` snippets")
    return "\n\n".join(bodies)
