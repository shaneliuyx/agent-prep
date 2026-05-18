"""Phase 3 — branch coverage measurement + LLM-guided edge-case generation.

`coverage.py --branch` measures (from_line, to_line) edges. Missed edges
become explicit LLM prompts: "generate a test that exercises the branch
from line A to line B in function F."

Loop until target coverage met OR LLM stops producing novel tests.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def run_coverage(target: str = "src/", testdir: str = "tests/",
                 covpath: str = "coverage.json") -> dict:
    """Run pytest with branch coverage + emit JSON report.
    `target` scopes the source-tracking directory (used in production via
    `--source=<target>` flag; omitted here so coverage tracks whatever
    pytest imports)."""
    cmd = ["coverage", "run", f"--source={target}", "--branch",
           "-m", "pytest", testdir]
    subprocess.run(cmd, check=False)
    subprocess.run(["coverage", "json", "-o", covpath], check=True)
    return json.loads(Path(covpath).read_text())


def missed_branches(cov: dict, file: str) -> list[tuple[int, int]]:
    """Return list of (from_line, to_line) for each missed branch in `file`."""
    file_cov = cov.get("files", {}).get(file, {})
    return [tuple(b) for b in file_cov.get("missing_branches", [])]


def coverage_pct(cov: dict, file: str) -> tuple[float, float]:
    """Return (line_coverage_pct, branch_coverage_pct) for `file`."""
    file_cov = cov.get("files", {}).get(file, {})
    summary = file_cov.get("summary", {})
    return (
        summary.get("percent_covered", 0.0),
        summary.get("percent_covered_display", 0.0),
    )


if __name__ == "__main__":
    cov_path = sys.argv[1] if len(sys.argv) > 1 else "coverage.json"
    cov = json.loads(Path(cov_path).read_text())
    print(f"\nFiles measured: {len(cov.get('files', {}))}")
    for file in sorted(cov.get("files", {})):
        line_pct, branch_pct = coverage_pct(cov, file)
        missed = missed_branches(cov, file)
        print(f"  {file:50}  line={line_pct:5.1f}%  missed_branches={len(missed)}")
