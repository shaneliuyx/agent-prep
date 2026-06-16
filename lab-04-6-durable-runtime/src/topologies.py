"""topologies.py — four 5-node DAG shapes that isolate STRUCTURE.

WHY all four graphs have exactly 5 nodes:
The bench's question is "does topology change throughput?" To answer it honestly,
hold everything constant except the variable under test. Node count is held at 5
and (for the bench) the model is held constant, so the ONLY thing that varies is
the dependency structure — sequential vs fan-out vs hierarchical vs feedback. Any
wall-clock difference is then attributable to structure, not to "this graph just
has more work". This is the W4.5 "hold mode constant" discipline applied to shape.

Each builder returns (dag, concurrency). `concurrency` is the worker count that
*matches* the shape's available parallelism — a fan-out of 4 with concurrency=1
would measure the scheduler, not the topology.

Node payloads:
- tool nodes carry {"sleep_s": float}
- llm  nodes carry {"prompt": str, "model": MODEL}
Pass `llm=True` to emit llm nodes for the live oMLX bench; default tool nodes for
deterministic offline use (tests, demos without a model).
"""
from __future__ import annotations

from typing import Any

# The single constant model for the bench. Held constant across ALL four
# topologies so structure is the only independent variable (see module docstring).
BENCH_MODEL = "Qwen2.5-Coder-7B-Instruct-MLX-4bit"

_PROMPT = "Reply with exactly one word: ok."
_SLEEP_S = 0.2


def _node(llm: bool, model: str) -> dict[str, Any]:
    """One node spec, either an llm node (real oMLX call) or a tool node (sleep)."""
    if llm:
        return {"type": "llm", "payload": {"prompt": _PROMPT, "model": model}}
    return {"type": "tool", "payload": {"sleep_s": _SLEEP_S}}


def sequential(llm: bool = False, model: str = BENCH_MODEL) -> tuple[dict, int]:
    """Linear chain n1→n2→n3→n4→n5. No parallelism available → concurrency=1.
    The throughput floor: total time ≈ sum of node times."""
    nodes = {f"n{i}": _node(llm, model) for i in range(1, 6)}
    edges = [[f"n{i}", f"n{i + 1}"] for i in range(1, 5)]
    return {"nodes": nodes, "edges": edges}, 1


def parallel(llm: bool = False, model: str = BENCH_MODEL) -> tuple[dict, int]:
    """Fan-out: 1 root → 4 independent leaves. Max parallelism → concurrency=4.
    The throughput ceiling: after the root, the 4 leaves overlap."""
    nodes = {f"n{i}": _node(llm, model) for i in range(1, 6)}
    edges = [["n1", "n2"], ["n1", "n3"], ["n1", "n4"], ["n1", "n5"]]
    return {"nodes": nodes, "edges": edges}, 4


def hierarchical(llm: bool = False, model: str = BENCH_MODEL) -> tuple[dict, int]:
    """Manager → 3 workers → gather (5 nodes total). The 3 workers run in
    parallel, then a single gather joins them → concurrency=3. Models the common
    'orchestrator fans to sub-agents, then synthesizes' agent pattern."""
    nodes = {
        "manager": _node(llm, model),
        "worker1": _node(llm, model),
        "worker2": _node(llm, model),
        "worker3": _node(llm, model),
        "gather": _node(llm, model),
    }
    edges = [
        ["manager", "worker1"], ["manager", "worker2"], ["manager", "worker3"],
        ["worker1", "gather"], ["worker2", "gather"], ["worker3", "gather"],
    ]
    return {"nodes": nodes, "edges": edges}, 3


def workflow(llm: bool = False, model: str = BENCH_MODEL) -> tuple[dict, int]:
    """Generate → validate (with a validation-feedback retry edge) → finalize,
    inside 5 nodes. Strictly sequential → concurrency=1.

    The feedback loop is modeled durably, not with an in-memory while-loop: the
    `validate` node may call mark_failed on `gen`, which (because graph_store
    persists the attempt counter) requeues `gen` READY for up to max_attempts
    retries before giving up. Here we lay out the structure; the validate node's
    handler decides whether to force a gen retry. Nodes:
        gen → validate → finalize, plus two helper nodes (precheck, postcheck)
        to keep the count at 5 and give the validate/finalize stage real deps."""
    nodes = {
        "precheck": _node(llm, model),
        "gen": _node(llm, model),
        "validate": _node(llm, model),
        "finalize": _node(llm, model),
        "postcheck": _node(llm, model),
    }
    edges = [
        ["precheck", "gen"],
        ["gen", "validate"],
        ["validate", "finalize"],
        ["finalize", "postcheck"],
    ]
    return {"nodes": nodes, "edges": edges}, 1
