"""
src/tools.py — the shared tool set for the pattern zoo.

Three deterministic, offline tools so every pattern runs reproducibly with zero
network dependency (the point of this lab is to compare AGENT PATTERNS, not tool
backends). Each tool is exposed TWO ways, on purpose:

  1. As a plain Python callable (kb_lookup / add / multiply) — CodeAct executes
     model-emitted code in a namespace where these are in scope, so it can
     compose them (loop, arithmetic, chaining) in ONE action.
  2. As an OpenAI function-calling schema (TOOL_SCHEMAS) + a name->callable
     registry (TOOLS) — ReAct / Plan-and-Solve / ReWOO emit structured tool
     calls and the harness dispatches by name.

Same functions, same results, two surfaces — so the only thing that differs
across patterns is the *control flow*, which is exactly what the matrix measures.
"""

from __future__ import annotations

from typing import Any, Callable

# ---------------------------------------------------------------------------
# Fixed knowledge base. Deterministic so the canonical task has one ground
# truth: (Springfield 30000 + Shelbyville 12000) * growth_factor 3 = 126000.
# ---------------------------------------------------------------------------
_KNOWLEDGE_BASE: dict[str, float] = {
    "springfield": 30000,
    "shelbyville": 12000,
    "growth_factor": 3,
    "capital_city": 95000,
    "ogdenville": 8000,
}


# ---------------------------------------------------------------------------
# Tool 1: kb_lookup — fetch a numeric value by key from the knowledge base.
# ---------------------------------------------------------------------------
def kb_lookup(key: str) -> float:
    """Return the numeric value stored under `key` in the knowledge base.

    Raises KeyError on an unknown key so callers (and CodeAct tracebacks) see a
    real failure rather than a silent default.
    """
    norm = str(key).strip().lower()
    if norm not in _KNOWLEDGE_BASE:
        raise KeyError(
            f"kb_lookup: unknown key {key!r}. "
            f"Known keys: {sorted(_KNOWLEDGE_BASE)}"
        )
    return _KNOWLEDGE_BASE[norm]


# ---------------------------------------------------------------------------
# Tool 2: add — sum two numbers.
# ---------------------------------------------------------------------------
def add(a: float, b: float) -> float:
    """Return a + b."""
    return float(a) + float(b)


# ---------------------------------------------------------------------------
# Tool 3: multiply — product of two numbers.
# ---------------------------------------------------------------------------
def multiply(a: float, b: float) -> float:
    """Return a * b."""
    return float(a) * float(b)


# ---------------------------------------------------------------------------
# Surface A — plain-callable registry (CodeAct injects these into its namespace).
# ---------------------------------------------------------------------------
TOOLS: dict[str, Callable[..., Any]] = {
    "kb_lookup": kb_lookup,
    "add": add,
    "multiply": multiply,
}


# ---------------------------------------------------------------------------
# Surface B — OpenAI function-calling schemas (ReAct / Plan-Solve / ReWOO).
# Same shape as lab-04-react-from-scratch/src/tools.py (type/function/parameters,
# additionalProperties:False).
# ---------------------------------------------------------------------------
_KB_LOOKUP_SCHEMA = {
    "type": "function",
    "function": {
        "name": "kb_lookup",
        "description": (
            "Look up a numeric value by key in the knowledge base. "
            "Use this to fetch facts like a city's population or a named factor. "
            "Returns the number stored under that key."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "The lookup key, e.g. 'Springfield' or 'growth_factor'.",
                }
            },
            "required": ["key"],
            "additionalProperties": False,
        },
    },
}

_ADD_SCHEMA = {
    "type": "function",
    "function": {
        "name": "add",
        "description": "Add two numbers and return the sum.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First addend."},
                "b": {"type": "number", "description": "Second addend."},
            },
            "required": ["a", "b"],
            "additionalProperties": False,
        },
    },
}

_MULTIPLY_SCHEMA = {
    "type": "function",
    "function": {
        "name": "multiply",
        "description": "Multiply two numbers and return the product.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First factor."},
                "b": {"type": "number", "description": "Second factor."},
            },
            "required": ["a", "b"],
            "additionalProperties": False,
        },
    },
}

TOOL_SCHEMAS: list[dict] = [_KB_LOOKUP_SCHEMA, _ADD_SCHEMA, _MULTIPLY_SCHEMA]


def dispatch(name: str, args: dict) -> str:
    """Execute a tool by name with kwargs; return a string result (errors too).

    Shared by the structured-tool-call patterns (ReAct, Plan-Solve, ReWOO).
    Never raises — a failed call becomes an observation the model can read.
    """
    fn = TOOLS.get(name)
    if fn is None:
        return f"ERROR: unknown tool {name!r}. Available: {list(TOOLS)}"
    try:
        return str(fn(**args))
    except TypeError as e:
        return f"ERROR: bad arguments for tool {name!r}: {e}"
    except Exception as e:  # KeyError from kb_lookup, etc. — surface as observation
        return f"ERROR: tool {name!r} raised {type(e).__name__}: {e}"


def tools_doc() -> str:
    """One-line-per-tool signature doc, injected into CodeAct/ReWOO prompts so the
    model knows the exact callable names and arity without re-deriving them."""
    return (
        "kb_lookup(key: str) -> float   # numeric value for a key in the knowledge base\n"
        "add(a: float, b: float) -> float   # a + b\n"
        "multiply(a: float, b: float) -> float   # a * b"
    )
