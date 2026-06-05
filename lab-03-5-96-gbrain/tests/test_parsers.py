"""Tests for the gbrain-output line parser shared by the benchmark + A/B scripts.

`gbrain search/query --json` prints `[score] slug -- text` lines; bench_rrf and
ground_truth_ab each parse the slug out with a module-level `_LINE` regex. These
lock that parsing (ranked slug extraction; ignore non-matching lines).

Run: uv run --with pytest python -m pytest tests/test_parsers.py -v
"""
from __future__ import annotations

import bench_rrf
import ground_truth_ab

_SAMPLE = [
    "[0.9164] people/lin-zhao -- # Lin Zhao",
    "[0.5450] companies/quanta-labs -- # Quanta Labs",
    "[1.6779] deals/helix-series-a -- Helix Series A round",
    "Starting GBrain MCP server (stdio)...",   # noise line — must be ignored
    "",                                          # blank — ignored
    "random log without brackets",               # no match
]
_EXPECTED = ["people/lin-zhao", "companies/quanta-labs", "deals/helix-series-a"]


def _slugs(line_re) -> list[str]:
    out = []
    for ln in _SAMPLE:
        m = line_re.match(ln.strip())
        if m:
            out.append(m.group(1))
    return out


def test_bench_rrf_line_parser():
    assert _slugs(bench_rrf._LINE) == _EXPECTED


def test_ground_truth_line_parser():
    assert _slugs(ground_truth_ab._LINE) == _EXPECTED


def test_line_parser_handles_negative_and_int_scores():
    # scores can be >1 (fused) or negative; the regex must still grab the slug
    for re_ in (bench_rrf._LINE, ground_truth_ab._LINE):
        assert re_.match("[-0.12] people/x -- t").group(1) == "people/x"
        assert re_.match("[2] companies/y -- t").group(1) == "companies/y"


def test_line_parser_rejects_nonmatching():
    for re_ in (bench_rrf._LINE, ground_truth_ab._LINE):
        assert re_.match("no brackets here") is None
        assert re_.match("[0.5] missing-separator text") is None  # no ' -- '
