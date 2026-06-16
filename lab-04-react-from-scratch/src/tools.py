"""
src/tools.py — four tool implementations + JSON schemas.

Import this module before calling agent_run() to register tools with the loop.
Each tool:
  1. Has a JSON schema that matches the OpenAI function-calling format.
  2. Has a Python function that implements the actual behavior.
  3. Is registered via register_tool() at the bottom of this file.

Tools are deliberately simple. The goal is to exercise the loop's error-handling,
not to build production-grade tool integrations.
"""

from __future__ import annotations

import ast
import os
import resource
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

sys.path.insert(0, "/Users/yuxinliu/code/agent-prep/shared")
from web_toolkit import web_search as _web_search_backend  # introduced W3.7; reused here

from src.react import register_tool

# ---------------------------------------------------------------------------
# Tool 1: web_search
#    Delegates to shared/web_toolkit (first taught in W3.7) instead of calling
#    DuckDuckGo directly. Backend precedence SearXNG -> Tavily -> DuckDuckGo,
#    disk-cached for reproducibility. The tool's contract (query -> formatted
#    string) is unchanged, so the ReAct loop is unaffected. Per the repo's
#    "introduce inline, reuse via import" rule, W4 consumes web search rather
#    than re-implementing it.
# ---------------------------------------------------------------------------
_WEB_SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for current information. "
            "Use this when you need facts you don't know or that may have changed recently. "
            "Returns up to 5 result snippets."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query. Be specific.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


def web_search(query: str) -> str:
    """Search the web; return top-5 result snippets as a numbered list.

    The actual search is delegated to shared/web_toolkit's structured
    `web_search` (returns typed SearchResult objects); we format them into the
    same numbered string the loop already expects. Swapping the backend (e.g.
    pointing SEARXNG_URL elsewhere) needs no change here.
    """
    try:
        results = _web_search_backend(query, results=5)
        if not results:
            return "web_search: no results found for that query."
        lines = [
            f"[{i}] {r.title}\n    {r.snippet}\n    URL: {r.url}"
            for i, r in enumerate(results, 1)
        ]
        return "\n\n".join(lines)
    except Exception as e:
        return f"web_search error: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Tool 2: python_repl
#    Executes arbitrary Python code in a subprocess with a hard timeout.
#
#    SECURITY BOUNDARY — read before reusing this anywhere real:
#    What this DOES enforce: a wall-clock timeout; process-LOCAL isolation
#    (imports/globals/handles don't leak back into the agent process); a
#    stripped child env (no inherited secrets, see _REPL_ENV); and CPU+memory
#    rlimits (see _repl_rlimits). That bounds accidents and resource abuse.
#    What it does NOT do: the code still runs with your full user privileges
#    and full filesystem access (open() works; _static_check's import blocklist
#    is trivially bypassable via open()/__import__/attribute gadgets). This is
#    NOT a sandbox. It is acceptable HERE only because a ReAct teaching-lab
#    agent is non-adversarial. NEVER expose this REPL to untrusted input.
#    Still-missing pieces for a real boundary (built in W11.5 Agent Security):
#    drop privileges, filesystem containment, container / bubblewrap / nsjail,
#    no network namespace.
# ---------------------------------------------------------------------------
_PYTHON_REPL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "python_repl",
        "description": (
            "Execute Python code and return stdout + stderr. "
            "Use for calculations, data transformations, or verifying logic. "
            "Do not use for file I/O — use read_file / write_file instead. "
            "Code runs in a short-lived subprocess with a timeout; a few "
            "imports (os, sys, subprocess, ...) are blocked."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Valid Python 3 code to execute. Print results you want to see.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Maximum seconds to wait. Default 10. Max 30.",
                    "default": 10,
                },
            },
            "required": ["code"],
            "additionalProperties": False,
        },
    },
}

_ALLOWED_BUILTINS = {
    # Aspirational whitelist — NOT currently enforced. The subprocess runs with
    # the full builtins; nothing applies this set yet (it documents intent and
    # is a TODO: pass as a restricted globals dict). Do not mistake it for a
    # guard — neither this set nor _static_check makes the REPL safe.
    "abs", "all", "any", "bin", "bool", "chr", "dict", "dir", "divmod",
    "enumerate", "filter", "float", "format", "frozenset", "getattr",
    "hasattr", "hash", "hex", "int", "isinstance", "issubclass", "iter",
    "len", "list", "map", "max", "min", "next", "oct", "ord", "pow",
    "print", "range", "repr", "reversed", "round", "set", "slice",
    "sorted", "str", "sum", "tuple", "type", "vars", "zip",
}

_BLOCKED_IMPORTS = {"subprocess", "os", "sys", "shutil", "socket", "ctypes", "importlib"}


def _static_check(code: str) -> str | None:
    """
    Light static analysis before subprocess execution.
    Returns an error string if the code looks dangerous, else None.
    This does NOT make the REPL safe — it is a best-effort guard.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"SyntaxError: {e}"

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            for name in names:
                top = name.split(".")[0]
                if top in _BLOCKED_IMPORTS:
                    return f"blocked import: '{name}' is not allowed in the REPL."
    return None


# Minimal env for the child: keep PATH/locale, DROP everything the parent
# inherited (API keys, tokens, secrets). Defense-in-depth so a bypass of the
# import blocklist (open('/proc/self/environ'), __import__) reads nothing useful.
_REPL_ENV = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}
_REPL_MEM_BYTES = 512 * 1024 * 1024   # 512 MB address-space cap


def _repl_rlimits(cpu_seconds: int):
    """Return a preexec_fn that caps CPU time and address space in the child
    before exec. POSIX-only; each limit is best-effort — some platforms don't
    enforce RLIMIT_AS (notably macOS), so we swallow failures rather than crash
    the child. This bounds resource abuse; it is NOT isolation."""
    def _apply() -> None:
        for res, limit in (
            (resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds)),
            (resource.RLIMIT_AS, (_REPL_MEM_BYTES, _REPL_MEM_BYTES)),
        ):
            try:
                resource.setrlimit(res, limit)
            except (ValueError, OSError):
                pass  # limit unsupported on this platform (e.g. RLIMIT_AS on macOS)
    return _apply


def python_repl(code: str, timeout: int = 10) -> str:
    """Execute Python code in a subprocess; return combined stdout + stderr.

    Hardening (defense-in-depth, NOT a sandbox — see the SECURITY BOUNDARY note
    above): runs `sys.executable` with a stripped env (no inherited secrets) and
    CPU/memory rlimits. Still runs as your user with filesystem access; for real
    isolation see W11.5.
    """
    timeout = min(max(1, timeout), 30)   # clamp to [1, 30] seconds

    err = _static_check(code)
    if err:
        return f"python_repl static check failed: {err}"

    # Write code to a temp file to avoid shell-escaping issues
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmpfile = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmpfile],   # absolute interpreter — no PATH dependence
            capture_output=True,
            text=True,
            timeout=timeout,
            env=_REPL_ENV,               # strip inherited secrets from the child
            preexec_fn=_repl_rlimits(timeout),  # CPU + memory caps (POSIX)
        )
        out = result.stdout
        err_out = result.stderr
        combined = ""
        if out:
            combined += out
        if err_out:
            combined += f"\nSTDERR:\n{err_out}"
        return combined.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return (
            f"python_repl: execution timed out after {timeout}s. "
            "Simplify the code or break it into smaller steps."
        )
    except Exception as e:
        return f"python_repl error: {type(e).__name__}: {e}"
    finally:
        try:
            Path(tmpfile).unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Tool 3: read_file
#    Reads a file from the filesystem and returns its contents.
#    Restricted to paths under ~/code/agent-prep/lab-04-react-from-scratch/data/
#    to prevent path-traversal by the agent.
# ---------------------------------------------------------------------------
_DATA_DIR = Path(os.path.expanduser(
    "~/code/agent-prep/lab-04-react-from-scratch/data"
)).resolve()

_READ_FILE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": (
            "Read the contents of a file in the lab data directory. "
            "Provide the filename only (not a full path). "
            "Returns the file contents as a string."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Filename to read (e.g., 'notes.txt'). No path separators.",
                }
            },
            "required": ["filename"],
            "additionalProperties": False,
        },
    },
}


def read_file(filename: str) -> str:
    """Read a file from the lab data directory; enforce path containment."""
    # Strip any path components the model might inject
    safe_name = Path(filename).name
    target = (_DATA_DIR / safe_name).resolve()

    # Path traversal guard: resolved path must stay inside _DATA_DIR
    if not str(target).startswith(str(_DATA_DIR)):
        return f"read_file error: access denied (path traversal attempt: {filename!r})"

    if not target.exists():
        return f"read_file error: file '{safe_name}' not found in data directory."

    try:
        content = target.read_text(encoding="utf-8")
        return content or "(file is empty)"
    except Exception as e:
        return f"read_file error: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Tool 4: write_file
#    Writes (or overwrites) a file in the lab data directory.
#    Same path-containment guard as read_file.
#    This tool is intentionally non-idempotent in content — calling it twice
#    with different content produces different results. The agent must be aware
#    of this when deciding whether to call it again.
# ---------------------------------------------------------------------------
_WRITE_FILE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": (
            "Write text content to a file in the lab data directory. "
            "Creates the file if it does not exist; overwrites if it does. "
            "Provide the filename only (not a full path) and the content string."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Filename to write (e.g., 'output.txt'). No path separators.",
                },
                "content": {
                    "type": "string",
                    "description": "Text content to write to the file.",
                },
            },
            "required": ["filename", "content"],
            "additionalProperties": False,
        },
    },
}


def write_file(filename: str, content: str) -> str:
    """Write content to a file in the lab data directory; enforce path containment."""
    safe_name = Path(filename).name
    target = (_DATA_DIR / safe_name).resolve()

    if not str(target).startswith(str(_DATA_DIR)):
        return f"write_file error: access denied (path traversal attempt: {filename!r})"

    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        target.write_text(content, encoding="utf-8")
        return f"write_file: wrote {len(content)} characters to '{safe_name}'."
    except Exception as e:
        return f"write_file error: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Registration — runs on import
# ---------------------------------------------------------------------------
register_tool("web_search",   web_search,   _WEB_SEARCH_SCHEMA,   max_calls=4)
register_tool("python_repl",  python_repl,  _PYTHON_REPL_SCHEMA,  max_calls=6)
register_tool("read_file",    read_file,    _READ_FILE_SCHEMA,    max_calls=8)
register_tool("write_file",   write_file,   _WRITE_FILE_SCHEMA,   max_calls=4)