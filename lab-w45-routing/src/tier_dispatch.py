"""Dispatch a RouterVerdict to the executor: pick endpoint by tier, run
the right control-flow by mode. Mode `react` re-uses W4's run_agent();
`minimal` is a one-shot completion; `deliberate` is plan-then-execute.
"""
from __future__ import annotations

import os
from openai import OpenAI

from src.fleet_config import FLEET
from src.router import RouterVerdict


def _client_for(tier: str) -> OpenAI:
    ep = FLEET[tier]
    return OpenAI(base_url=ep.base_url, api_key=os.getenv("OMLX_API_KEY"))


def run_minimal(prompt: str, tier: str) -> str:
    """One-shot completion. ~85-210ms wall."""
    ep = FLEET[tier]
    r = _client_for(tier).chat.completions.create(
        model=ep.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=512,
    )
    return (r.choices[0].message.content or "").strip()


def run_react(prompt: str, tier: str) -> str:
    """ReAct loop. Re-uses W4's run_agent() with the dispatch tier's endpoint."""
    # W4 lab provides run_agent(prompt, base_url, model) — adapter here.
    # For the chapter's purposes, stub it as run_minimal + a "think-act" prefix
    # so the lab works end-to-end without depending on W4's local module.
    from src.fleet_config import FLEET
    ep = FLEET[tier]
    prefix = "Use the Think→Act→Observe pattern. Show your reasoning steps."
    r = _client_for(tier).chat.completions.create(
        model=ep.model,
        messages=[
            {"role": "system", "content": prefix},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=2048,
    )
    return (r.choices[0].message.content or "").strip()


def run_deliberate(prompt: str, tier: str) -> str:
    """Plan-then-execute. Two LLM calls: planner produces an outline, executor fills it."""
    ep = FLEET[tier]
    cli = _client_for(tier)
    plan = cli.chat.completions.create(
        model=ep.model,
        messages=[
            {"role": "system", "content": "Produce a numbered plan (3-6 steps). No execution."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=512,
    ).choices[0].message.content or ""
    exec_ = cli.chat.completions.create(
        model=ep.model,
        messages=[
            {"role": "system", "content": "Execute this plan step by step. Show work."},
            {"role": "user", "content": f"Question:\n{prompt}\n\nPlan:\n{plan}"},
        ],
        temperature=0.0,
        max_tokens=2048,
    ).choices[0].message.content or ""
    return exec_.strip()


def dispatch(verdict: RouterVerdict, prompt: str) -> str:
    """Pick run_* by verdict.mode; pass verdict.tier as the executor endpoint."""
    if verdict.mode == "minimal":
        return run_minimal(prompt, verdict.tier)
    if verdict.mode == "react":
        return run_react(prompt, verdict.tier)
    return run_deliberate(prompt, verdict.tier)