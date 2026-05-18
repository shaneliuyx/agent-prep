"""Phase 1 — AST walk + 4-class testability filter.

Extracts every FunctionDef from a Python source file. For each function,
classify into one of 5 categories based on the body's reference profile:

  pure_stateless    -> auto-generate test
  mock_required     -> I/O calls present
  fixture_required  -> decorator from web/test framework
  escalate          -> getattr / eval / exec — statically un-testable
  property_test     -> threading / asyncio / multiprocessing primitives

Filter is the load-bearing primitive: it decides WHICH functions the
downstream LLM should be asked to generate tests for, and HOW.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass


IO_NAMES = frozenset({
    "open", "requests", "httpx", "subprocess", "os", "socket",
    "psycopg", "sqlite3", "redis", "pymongo", "boto3", "psutil",
})
DYN_NAMES = frozenset({
    "getattr", "setattr", "eval", "exec", "globals", "locals",
})
CC_NAMES = frozenset({
    "Lock", "Queue", "Semaphore", "Event", "ThreadPoolExecutor",
    "ProcessPoolExecutor", "asyncio", "threading", "multiprocessing",
})


@dataclass(frozen=True)
class FunctionInfo:
    name: str
    signature: str
    has_io_calls: bool
    has_decorators: tuple[str, ...]
    has_dynamic_dispatch: bool
    has_concurrency: bool

    @property
    def testability(self) -> str:
        if self.has_dynamic_dispatch:
            return "escalate"
        if self.has_concurrency:
            return "property_test"
        if self.has_decorators:
            return "fixture_required"
        if self.has_io_calls:
            return "mock_required"
        return "pure_stateless"


def _render_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = []
    for a in node.args.args:
        ann = ast.unparse(a.annotation) if a.annotation else "Any"
        args.append(f"{a.arg}: {ann}")
    ret = ast.unparse(node.returns) if node.returns else "Any"
    return f"({', '.join(args)}) -> {ret}"


def _render_decorator(node: ast.expr) -> str:
    return ast.unparse(node)


def extract(source: str) -> list[FunctionInfo]:
    """Walk the AST + classify every top-level FunctionDef.
    Nested functions are skipped (col_offset > 0 filter)."""
    tree = ast.parse(source)
    out: list[FunctionInfo] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.col_offset > 0:
            continue  # skip nested funcs

        names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
        attrs = {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}
        all_refs = names | attrs

        out.append(FunctionInfo(
            name=node.name,
            signature=_render_signature(node),
            has_io_calls=bool(all_refs & IO_NAMES),
            has_decorators=tuple(_render_decorator(d) for d in node.decorator_list),
            has_dynamic_dispatch=bool(all_refs & DYN_NAMES),
            has_concurrency=bool(all_refs & CC_NAMES),
        ))
    return out
