"""web_browse — interactive browser automation via the agent-browser CLI.

Targets the agent-browser CLI that runs **one command per invocation against a
persistent browser session** (open → actions → extract → close), with JSON output of
the shape ``{"success": bool, "data": {...}, "error": str|null}``. Each high-level
:class:`BrowseAction` is mapped to one or more CLI calls.

Use when a page needs interaction (click "Load more", fill a search box, scroll an
infinite feed, wait for SPA JS) before its target content is available. For static pages,
use :func:`web_fetch` instead.

Note: the CLI keeps a single shared browser between invocations, so concurrent
``web_browse`` calls in the same environment are not isolated — call them sequentially.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Any, Optional, Sequence

from ._types import BrowseAction, BrowseResult

__all__ = ["web_browse", "BrowseError", "agent_browser_available", "build_action_commands"]

_AGENT_BROWSER = os.getenv("AGENT_BROWSER_BIN", "agent-browser")
_DEFAULT_TIMEOUT = 30


class BrowseError(RuntimeError):
    """Raised when agent-browser is unavailable or a browser action fails."""


def agent_browser_available() -> bool:
    return shutil.which(_AGENT_BROWSER) is not None


def _require(action: BrowseAction, field: str) -> str:
    val = action.get(field)
    if not isinstance(val, str) or not val:
        raise BrowseError(f'action "{action.get("type")}" requires non-empty {field}')
    return val


def build_action_commands(actions: Sequence[BrowseAction]) -> list[list[str]]:
    """Translate high-level browse actions into agent-browser CLI argument lists
    (one or more CLI invocations per action)."""
    cmds: list[list[str]] = []
    for a in actions:
        t = a.get("type")
        if t == "click":
            cmds.append(["click", _require(a, "selector")])
        elif t == "fill":
            cmds.append(["fill", _require(a, "selector"), _require(a, "value")])
        elif t == "type":
            cmds.append(["type", _require(a, "selector"), _require(a, "value")])
        elif t == "press":
            focus_sel = a.get("selector")
            if focus_sel:
                cmds.append(["focus", focus_sel])
            cmds.append(["press", _require(a, "key")])
        elif t == "wait":
            cmds.append(["wait", str(int(a.get("ms", 0)))])
        elif t == "wait_selector":
            cmds.append(["wait", _require(a, "selector")])  # CLI waits for the selector
        elif t == "scroll":
            direction = a.get("direction", "down")
            if direction == "top":
                cmds.append(["eval", "window.scrollTo(0, 0)"])
            elif direction == "bottom":
                cmds.append(["eval", "window.scrollTo(0, document.body.scrollHeight)"])
            else:
                cmds.append(["scroll", direction, str(a.get("amount", 500))])
        else:
            raise BrowseError(f"unsupported browser action: {t}")
    return cmds


def _run(args: list[str], timeout: int) -> dict[str, Any]:
    """Run one agent-browser command with --json; return the parsed envelope.

    Raises BrowseError on a missing binary or unparseable output.
    """
    try:
        proc = subprocess.run([_AGENT_BROWSER, *args, "--json"],
                              capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as err:
        raise BrowseError(
            f"'{_AGENT_BROWSER}' not found. Install with: npm i -g agent-browser"
        ) from err
    except subprocess.TimeoutExpired as err:
        raise BrowseError(f"agent-browser '{' '.join(args)}' timed out after {timeout}s") from err
    out = proc.stdout.strip()
    if not out:
        return {"success": proc.returncode == 0, "data": {}, "error": proc.stderr.strip() or None}
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        # Non-JSON success line (e.g. "✓ Browser closed"); treat exit code as truth.
        return {"success": proc.returncode == 0, "data": {}, "error": None}


def web_browse(url: str, actions: Optional[Sequence[BrowseAction]] = None, *,
               selector: Optional[str] = None, headless: bool = True,
               timeout: int = _DEFAULT_TIMEOUT) -> BrowseResult:
    """Open a browser, run an action chain, then extract content.

    Args:
        url: starting URL.
        actions: ordered :class:`BrowseAction` list (click/fill/type/press/wait/
            wait_selector/scroll). Max ~25 recommended.
        selector: CSS selector to extract text from the final page state. Omit to return
            the accessibility snapshot instead.
        headless: run without a visible window (default True).
        timeout: per-command timeout in seconds.

    Raises:
        BrowseError: agent-browser not installed, or a browser action fails.

    Failure of the open/extract steps is returned as ``BrowseResult(ok=False, error=...)``.
    """
    actions = list(actions or [])
    if not agent_browser_available():
        raise BrowseError(f"'{_AGENT_BROWSER}' not found. Install with: npm i -g agent-browser")

    steps: list[str] = []

    def step(args: list[str]) -> dict[str, Any]:
        steps.append(" ".join(args))
        return _run(args, timeout)

    try:
        open_args = ["open", url]
        if not headless:
            open_args.append("--headed")
        res = step(open_args)
        if not res.get("success"):
            return BrowseResult(url=url, ok=False, steps=steps,
                                error=f"open failed: {res.get('error') or 'unknown'}")

        for cmd in build_action_commands(actions):
            res = step(cmd)
            if not res.get("success"):
                return BrowseResult(url=url, ok=False, steps=steps,
                                    error=f"action failed: {' '.join(cmd)} - {res.get('error') or 'unknown'}")

        content_res = step(["get", "text", selector] if selector else ["snapshot"])
        cdata = content_res.get("data") or {}
        content = cdata.get("text") or cdata.get("snapshot") or (
            cdata if isinstance(cdata, str) else "")
        if isinstance(content, (dict, list)):
            content = json.dumps(content, ensure_ascii=False)

        title = (step(["get", "title"]).get("data") or {}).get("title", "")
        final_url = (step(["get", "url"]).get("data") or {}).get("url", url)

        return BrowseResult(url=final_url or url, ok=True, title=title,
                            content=content or "", steps=steps)
    finally:
        try:
            subprocess.run([_AGENT_BROWSER, "close"], capture_output=True, text=True, timeout=10)
        except Exception:  # noqa: BLE001 - best-effort cleanup
            pass
