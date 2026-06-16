"""
Offline unit tests for the pattern zoo — NO LLM endpoint required.

Covers the deterministic substrate every pattern relies on:
  - the fixed tool layer + ground-truth arithmetic (126000),
  - the OpenAI schema shape,
  - CodeAct's restricted-namespace guard (no import / open),
  - ReWOO's plan parsing + #E placeholder substitution.

The live-LLM pattern runs are exercised by src/05_compare.py against oMLX; those
are intentionally not unit tests (they need the local server).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.impl_codeact import _execute, _extract_code, _make_namespace
from src.impl_rewoo import _parse_plan, _substitute
from src.schema import CANONICAL_EXPECTED_ANSWER, AgentResult
from src.tools import TOOL_SCHEMAS, dispatch


def test_ground_truth():
    assert dispatch("kb_lookup", {"key": "Springfield"}) == "30000"
    assert dispatch("kb_lookup", {"key": "Shelbyville"}) == "12000"
    assert dispatch("add", {"a": 30000, "b": 12000}) == "42000.0"
    assert dispatch("multiply", {"a": 42000, "b": 3}) == "126000.0"
    assert CANONICAL_EXPECTED_ANSWER == "126000"


def test_dispatch_errors_are_observations_not_raises():
    assert dispatch("kb_lookup", {"key": "atlantis"}).startswith("ERROR")
    assert dispatch("nope", {}).startswith("ERROR")


def test_tool_schema_shape():
    names = {s["function"]["name"] for s in TOOL_SCHEMAS}
    assert names == {"kb_lookup", "add", "multiply"}
    for s in TOOL_SCHEMAS:
        assert s["type"] == "function"
        assert s["function"]["parameters"]["additionalProperties"] is False


def test_codeact_namespace_blocks_import():
    ns = _make_namespace()
    out = _execute("import os\nprint(os.getcwd())", ns)
    assert out.startswith("ERROR")  # __import__ not in restricted builtins


def test_codeact_namespace_runs_tools():
    ns = _make_namespace()
    code = 'print("FINAL:", multiply(add(kb_lookup("Springfield"), kb_lookup("Shelbyville")), kb_lookup("growth_factor")))'
    out = _execute(code, ns)
    assert "126000" in out


def test_codeact_extract_code_block():
    assert _extract_code("blah\n```python\nprint(1)\n```\nend") == "print(1)\n"
    assert _extract_code("no code here") is None


def test_rewoo_plan_parse_and_substitute():
    plan = (
        '#E1 = kb_lookup[{"key": "Springfield"}]\n'
        '#E2 = kb_lookup[{"key": "Shelbyville"}]\n'
        '#E3 = add[{"a": "#E1", "b": "#E2"}]\n'
    )
    steps = _parse_plan(plan)
    assert len(steps) == 3
    assert steps[0] == ("#E1", "kb_lookup", '{"key": "Springfield"}')
    # placeholder substitution: #E1/#E2 -> computed values, valid JSON out
    args = _substitute('{"a": "#E1", "b": "#E2"}', {"#E1": "30000", "#E2": "12000"})
    assert args == {"a": 30000, "b": 12000}


def test_agent_result_total_tokens():
    r = AgentResult("X", "ans", llm_calls=1, prompt_tokens=10, completion_tokens=5, latency_ms=1)
    assert r.total_tokens == 15
