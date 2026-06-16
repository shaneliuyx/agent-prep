"""Phase 5 cost-latency benchmark — 2-tier architecture.

Runs the eval split through 3 routing configs (heavy_always, router2, random),
measures wall + REAL token usage + soft success per row, aggregates to the
Pareto-front input, writes RESULTS_phase5.json.

PERF (why the old version took >1h): on one-hot oMLX, swapping the executor model
cold-loads a heavy model (~10-30s). The old `for row: for config:` loop swapped
models ~90 times. This version loops `for config:` and SORTS each config's rows by
executor model so the heavy model loads once per config — a handful of cold-loads
total. It also caps max_tokens and uses single-call cells (the cost bench measures
the TIER decision, not mode control-flow), and sets a per-call timeout.

Integration + slow: hits the live oMLX fleet. Run: RUN_INTEGRATION=1 uv run pytest -m slow.
"""
import json
import os
import random
import time
from pathlib import Path

import pytest
from openai import OpenAI

from src.fleet_config import FLEET
from src.probes import load_probes, train_eval_split
from src.router2 import classify2

# 2-tier executor map: haiku-class -> cheap fast model; heavy-class -> gemma-26B workhorse
# (the local model that scored best on tier). One key per distinct hot model.
HEAVY_EXEC = "sonnet"  # FLEET["sonnet"] = gemma-4-26B (the 'heavy' executor)


def exec_tier(tier2: str) -> str:
    return "haiku" if tier2 == "haiku" else HEAVY_EXEC


# Public cloud-equivalent rates ($/M tokens) so local cost is in production language.
COST_PER_M_TOKENS = {
    "haiku":  {"input": 1.00, "output": 5.00},
    "sonnet": {"input": 3.00, "output": 15.00},
}


def _cost_usd(exec_t: str, in_tok: int, out_tok: int) -> float:
    rate = COST_PER_M_TOKENS[exec_t]
    return (in_tok * rate["input"] + out_tok * rate["output"]) / 1_000_000


_RNG = random.Random(42)  # seeded ONCE (old code re-seeded per call -> not random)


def _verdict_tier(cfg: str, prompt: str) -> str:
    """Return the 2-tier {haiku, heavy} decision for a config."""
    if cfg == "heavy_always":
        return "heavy"
    if cfg == "router2":
        return classify2(prompt).tier
    return _RNG.choice(["haiku", "heavy"])  # random baseline


_MODE_SYS = {
    "minimal": "Answer directly and concisely.",
    "react": "Reason step by step (Think -> Act -> Observe), then give the final answer.",
    "deliberate": "First outline a brief numbered plan, then execute it fully in this response.",
}
_MODE_MAXTOK = {"minimal": 512, "react": 2048, "deliberate": 2048}


def _run_cell(exec_t: str, prompt: str, mode: str) -> tuple[str, int, int]:
    """Mode-aware single completion at the routed tier. Gives a CORRECT model enough
    budget + the right prompting to actually succeed on a hard task — fixing the
    512-token truncation confound that made every config fail the strict judge.
    Real usage; long timeout for the 2048-token heavy generations."""
    ep = FLEET[exec_t]
    cli = OpenAI(base_url=ep.base_url, api_key=os.getenv("OMLX_API_KEY"), timeout=180.0)
    r = cli.chat.completions.create(
        model=ep.model,
        messages=[
            {"role": "system", "content": _MODE_SYS.get(mode, _MODE_SYS["minimal"])},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=_MODE_MAXTOK.get(mode, 1024),
    )
    resp = (r.choices[0].message.content or "").strip()
    u = r.usage
    return resp, (u.prompt_tokens if u else 0), (u.completion_tokens if u else 0)


JUDGE_BASE = "http://localhost:8317/v1"  # VibeProxy -> cloud Claude
JUDGE_MODEL = "claude-sonnet-4-6"


def _llm_judge_success(prompt: str, response: str) -> bool:
    """Discriminating success grade via an LLM judge (Sonnet through VibeProxy).

    Replaces the saturating soft grader (which passed any non-empty response). An
    under-powered / shallow answer to a hard prompt is a FAIL — this is what lets the
    bench distinguish router2 from random. User-only roles dodge VibeProxy's
    persona-cloak (BCJ-5). Cheap pre-filter avoids a judge call on empties/refusals.
    """
    if not response or len(response) < 20 or "i cannot" in response.lower():
        return False
    cli = OpenAI(base_url=JUDGE_BASE, api_key="vibeproxy", timeout=90.0)
    rubric = (
        "You are grading an LLM ROUTER's output: does RESPONSE acceptably answer PROMPT? "
        "Reason through the steps below, THEN give a verdict. Be strict.\n\n"
        "DEFINITIONS (what each criterion means):\n"
        "- CORRECT = factually accurate. No fabricated facts, invented APIs/flags, or made-up\n"
        "  citations. A confident wrong answer is worse than a hedged one.\n"
        "- ADEQUATE = the answer's DEPTH matches the PROMPT's DIFFICULTY. A hard prompt (system\n"
        "  design, multi-file/deep debugging, multi-step planning, non-trivial reasoning) needs\n"
        "  substantive, specific analysis. A trivial prompt (a fact, arithmetic, a one-liner)\n"
        "  only needs a short correct answer — extra length does NOT make it better.\n"
        "- UNDER-POWERED = a hard prompt answered shallowly: a generic platitude, a restatement\n"
        "  of the question, or a surface gloss where real analysis was required. This is a FAIL\n"
        "  EVEN IF nothing in it is factually wrong — it is the tell that a weak model was\n"
        "  mis-routed onto a task that needed a stronger one. (This is the whole point of the\n"
        "  bench: catch when 'route the easy class to the cheap model' sent a HARD task there.)\n"
        "- FAILURE MODES = auto-FAIL regardless of the above: empty, a refusal, off-topic, or\n"
        "  hallucinated APIs/facts.\n\n"
        "REASONING STEPS (write 1-2 short sentences for each, in order):\n"
        "  1. DIFFICULTY: rate the PROMPT trivial / moderate / hard, and say why.\n"
        "  2. CORRECTNESS: is RESPONSE factually right? Flag any fabrication.\n"
        "  3. ADEQUACY: does RESPONSE's depth match the difficulty from step 1?\n"
        "  4. FAILURE MODES: any empty / refusal / off-topic / hallucination?\n\n"
        "FINAL LINE — output exactly one of (and nothing after it):\n"
        "  VERDICT: PASS   -> only if CORRECT and ADEQUATE and NO failure mode\n"
        "  VERDICT: FAIL   -> otherwise\n\n"
        f"PROMPT:\n{prompt}\n\nRESPONSE:\n{response}"
    )
    out = cli.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": rubric}],
        max_tokens=600,
    ).choices[0].message.content or ""
    # CoT precedes the verdict — parse the LAST line containing "VERDICT". Default FAIL
    # if the judge produced no parseable verdict (fail-closed: don't credit an ambiguous grade).
    verdict_line = next(
        (ln for ln in reversed(out.upper().splitlines()) if "VERDICT" in ln), ""
    )
    return "PASS" in verdict_line and "FAIL" not in verdict_line


CONFIGS = ("heavy_always", "router2", "random")


@pytest.mark.integration
@pytest.mark.slow
def test_four_way_bench_runs_and_writes_results():
    _, eval_ = train_eval_split(load_probes())

    # Mode is held CONSTANT across configs (router2's mode per row), so the bench isolates
    # the TIER decision — only the executor model varies between configs, not the control-flow.
    modes = {r["prompt"]: classify2(r["prompt"]).mode for r in eval_}

    agg: dict = {}
    for cfg in CONFIGS:
        # Decide tier for every row first, then SORT by executor model so the heavy
        # model cold-loads once (not once per row) — the key perf fix.
        planned = [(r, exec_tier(_verdict_tier(cfg, r["prompt"]))) for r in eval_]
        planned.sort(key=lambda x: x[1])  # group haiku rows, then heavy rows

        rows_ = []
        for r, exec_t in planned:
            t0 = time.perf_counter()
            resp, in_tok, out_tok = _run_cell(exec_t, r["prompt"], modes[r["prompt"]])
            wall_ms = (time.perf_counter() - t0) * 1000
            rows_.append({
                "exec_tier": exec_t,
                "wall_ms": wall_ms,
                "in_tokens": in_tok,
                "out_tokens": out_tok,
                "cost_usd": _cost_usd(exec_t, in_tok, out_tok),
                "success": _llm_judge_success(r["prompt"], resp),
            })

        walls = sorted(r["wall_ms"] for r in rows_)
        n = len(rows_)
        agg[cfg] = {
            "n": n,
            "success_rate": sum(r["success"] for r in rows_) / n,
            "p50_wall_ms": walls[n // 2],
            "p95_wall_ms": walls[min(int(n * 0.95), n - 1)],
            "mean_cost_usd": sum(r["cost_usd"] for r in rows_) / n,
            "pct_routed_haiku": sum(r["exec_tier"] == "haiku" for r in rows_) / n,
        }

    Path("RESULTS_phase5.json").write_text(json.dumps(agg, indent=2))

    # Routing must beat heavy-always on cost (it routes the easy class to haiku)...
    assert agg["router2"]["mean_cost_usd"] < agg["heavy_always"]["mean_cost_usd"], (
        "router2 didn't beat heavy_always on cost — routing isn't winning"
    )
    # ...and beat random on success (the taxonomy carries signal).
    assert agg["router2"]["success_rate"] >= agg["random"]["success_rate"], (
        "router2 didn't beat random on success — taxonomy is broken"
    )
