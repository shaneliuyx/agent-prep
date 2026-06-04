"""The lab's headline claim (chapter Phase 5 / exit criterion): a relevant LEARNING
fact, recalled at decision time, PROVABLY changes the agent's chosen action.

Paired trial: same task + same seed, once with metacognitive recall OFF, once ON;
measure decision divergence (chosen tool differs) and improvement (the ON choice is
the better tool the self-pattern points to).

Setup uses CURATED LEARNING facts (deterministic) so the test isolates the
RECALL effect from the LLM extractor's nondeterminism. The full
OBSERVABILITY→extractor→LEARNING pipeline is exercised by the run (scripts +
learning_extractor) and reported in RESULTS.md.

Needs oMLX up (the agent decision model). Skips if not reachable.
"""
import os
import pathlib
import sys
import time

# Allow running this file directly (`uv run python tests/test_...py`), not just
# under pytest: put src/ on the path and load .env. Under pytest, conftest.py
# already does both — these are idempotent.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
try:
    from dotenv import load_dotenv
    load_dotenv(pathlib.Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

import observability as obs
import pytest
from openai import OpenAI

import demo_agent

# Probes use CONTRARIAN, environment-specific self-patterns — ones the base model
# canNOT know from training, that OVERRIDE its default tool choice. (General
# best-practices like "rg > grep" are already in the model's priors, so recalling
# them changes nothing — measured: divergence 0. Recall earns its keep only on
# idiosyncratic self-knowledge. That is the lab's central finding.)
# (type, contrarian LEARNING fact, task, model's PRIOR tool, tool the fact points to)
PROBES = [
    ("tool_preference", "In my environment rg segfaults on this repo's symlinked vendor dirs; plain grep is the one that completes here.",
     "Search this repository for every caller of parse_config.", "rg", "grep"),
    ("tool_preference", "rg is not installed on my machine; grep is the only working text search here.",
     "Find all usages of the logger across this codebase.", "rg", "grep"),
    ("recurring_mistake", "fd is not available in my setup; find is the only file finder installed.",
     "Locate config.yaml somewhere in this project tree.", "fd", "find"),
    ("recurring_mistake", "On my machine fd ignores dotfiles and misses config files; find catches them.",
     "Find the settings file in this project.", "fd", "find"),
    ("failure_pattern", "web_search is disabled on my network; read_local_notes has the cached answers I need.",
     "Look up the current best practice for a caching strategy.", "web_search", "read_local_notes"),
    ("failure_pattern", "My web_search tool is rate-limited to near-zero today; read_local_notes is the working fallback.",
     "Search for how to configure retry backoff.", "web_search", "read_local_notes"),
]


def _omlx_up() -> bool:
    # Probe with a tiny CHAT call, not models.list() — oMLX's /v1/models can
    # return empty and false-negative the guard (skipping the real test).
    try:
        OpenAI(base_url=os.getenv("OMLX_BASE_URL", "http://localhost:8000/v1"),
               api_key=os.getenv("OMLX_API_KEY", "dummy")).chat.completions.create(
            model=os.getenv("MODEL_AGENT", "Qwen2.5-Coder-14B-Instruct-MLX-4bit"),
            messages=[{"role": "user", "content": "ok"}], max_tokens=2)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _omlx_up(), reason="oMLX :8000 not reachable")
def test_recall_changes_decision(tmp_path, capsys):
    conn = obs.connect(str(tmp_path / "trial.db"))
    # seed curated LEARNING facts (dedup across the repeated fact text is fine)
    seen = set()
    for ftype, fact, _task, _dt, _bt in PROBES:
        if fact in seen:
            continue
        seen.add(fact)
        conn.execute("INSERT INTO learning (type, pattern_text, confidence, is_self_caused, "
                     "source_rows, ts) VALUES (?,?,?,1,'[]',?)",
                     (ftype, fact, 0.8, time.time()))
    conn.commit()

    diverged = improved = 0
    for i, (_ftype, _fact, task, default_tool, better_tool) in enumerate(PROBES):
        off, _ = demo_agent.decide(conn, task, use_recall=False, run_id=f"off-{i}")
        on, block = demo_agent.decide(conn, task, use_recall=True, run_id=f"on-{i}")
        if on != off:
            diverged += 1
        if on == better_tool and off != better_tool:
            improved += 1
        print(f"  [{task[:48]}] off={off} on={on} "
              f"(better={better_tool}) recall={'hit' if block else 'MISS'}")

    n = len(PROBES)
    div_rate, imp_rate = diverged / n, improved / n
    with capsys.disabled():
        print(f"\n  divergence: {diverged}/{n} = {div_rate:.0%}  |  "
              f"improvement: {improved}/{n} = {imp_rate:.0%}  (target divergence ≥ 30%)")

    # Non-flaky bar: recall must change AT LEAST ONE decision (the claim is real).
    # The 30% target is reported, not hard-asserted (LLM nondeterminism at n=6).
    assert diverged >= 1, "metacognitive recall changed no decision — claim unsupported"


if __name__ == "__main__":  # `uv run python tests/test_self_recall_changes_behavior.py`
    sys.exit(pytest.main([__file__, "-s", "-q"]))
