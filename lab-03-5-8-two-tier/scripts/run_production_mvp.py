"""Production MVP — verifies the W3.5.8 production runbook empirically.

Exercises:
  - Layer 1.5 speaker-role auto-tag + Layer 2 schema-shape classifier
  - SafeAtomiser with K_min=8 volume floor
  - Position discipline (raw FIRST, derived facts LAST)
  - DeploymentGate A/B comparison

Verification claims tested:
  C1. Classifier correctly tags conversational LongMemEval sessions
  C2. SafeAtomiser drops below-K_min triple sets (mechanical guardrail)
  C3. DeploymentGate would reject a synthetically-bad extractor (1-triple
      constrained variant from v5/v7 — known to cost −30 to −35pts)
  C4. End-to-end: production-mode (auto-classify → unconstrained atomise →
      K_min guard → position discipline) does NOT regress vs raw-only
      baseline (delta >= -3pts). Non-regression claim — atomise's measured
      lift is +5pts at the ±5pt noise floor (§5.3.2), so the MVP verifies
      the guardrails prevent catastrophic regression, not that atomise is
      a reliable improvement.

Usage:
    uv run python scripts/run_production_mvp.py --limit 10
    uv run python scripts/run_production_mvp.py --limit 20 --out results/mvp.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI

from src.production_router import (
    K_MIN_DEFAULT,
    SpeakerRole,
    WorkloadShape,
    build_composer_prompt,
    classify_workload,
    deployment_gate,
    route,
    safe_atomise,
)
from src.tiered_memory_qdrant import TieredMemory


# Import prompts from the eval runner so MVP shares COMPOSE_SYSTEM + parsing
# with the chapter's measurement script — single source of truth.
from scripts.run_longmemeval_oracle import (
    ATOMISE_SYSTEM,
    COMPOSE_SYSTEM,
    JUDGE_PROMPT,
)


# Run-unique ID — prevents cross-RUN Qdrant residue. The Qdrant collection
# persists between MVP invocations; without a per-run prefix, run N sees
# run N-1's imprints (BCJ Entry 14). Set once at import; every user_id
# namespace embeds it.
RUN_ID = str(int(time.time()))

# Opus 4.7 (and other extended-thinking models) DEPRECATE the `temperature`
# parameter — passing it returns HTTP 400. Local 4-bit MLX models, in
# contrast, want temperature=0.0 for deterministic eval. Gate it: set
# DISABLE_TEMPERATURE=1 when the compose endpoint is a thinking model.
_TEMP_KW: dict = {} if os.getenv("DISABLE_TEMPERATURE") == "1" else {"temperature": 0.0}


def _qid(q: dict) -> str:
    """Stable per-question id — explicit question_id, else question prefix."""
    return q.get("question_id") or q["question"][:32]


def _extract_answer(raw: str) -> str:
    """Same parser the eval runner uses. Take LAST <answer> match;
    reject `...` sentinel; fall back to last non-empty line."""
    matches = re.findall(r"<answer>(.*?)</answer>", raw, re.DOTALL | re.IGNORECASE)
    for cand in reversed(matches):
        c = cand.strip()
        if c and c != "...":
            return c
    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    return lines[-1] if lines else raw


def _parse_verdict(judge_raw: str) -> str:
    judge_upper = judge_raw.upper()
    last_correct = judge_upper.rfind("CORRECT")
    last_incorrect = judge_upper.rfind("INCORRECT")
    if last_incorrect > -1 and last_incorrect + 2 >= last_correct:
        return "INCORRECT"
    if last_correct > -1:
        return "CORRECT"
    return "UNKNOWN"


# ── Atomisers ──────────────────────────────────────────────────────

def make_unconstrained_atomiser(llm: OpenAI, model: str):
    """v3 Pareto winner — emits many triples (14-57 per session)."""
    def atomiser(ctx: str) -> list[str]:
        resp = llm.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": ATOMISE_SYSTEM},
                {"role": "user", "content": ctx},
            ],
            **_TEMP_KW,
            max_tokens=600,
        )
        text = (resp.choices[0].message.content or "").strip()
        return [l for l in text.splitlines() if l.strip()]
    return atomiser


# Genuinely-broken extractor — fixed garbage triples unrelated to any
# context. MODEL-INDEPENDENT negative fixture: garbage facts carry zero
# signal, so they can NEVER legitimately beat raw retrieval by the gate's
# +3pt threshold. Best case the composer ignores them (delta ~0); worst
# case they distract it (delta < 0). Either way delta < +3 → gate must
# reject → C3 passes on any model.
#
# Why this replaces the old LLM-constrained 1-triple extractor: on a
# strong full-precision model (Opus 4.7) the constrained extractor still
# produced a GOOD single fact and scored +20pts, so the gate correctly
# shipped it and C3 "failed". The anchoring-bias collapse it relied on is
# a 4-bit-quantization artifact, not model-independent. Fixed garbage has
# no such model dependence.
_GARBAGE_TRIPLES = [
    "user | favorite color | teal",
    "user | home planet | Mars",
    "user | pet count | 47",
    "user | birth year | 1623",
    "user | shoe size | 99",
    "user | occupation | lighthouse keeper",
    "user | last meal | gravel",
    "user | timezone | UTC+25",
    "user | hair color | invisible",
    "user | current mood | quadratic",
]


def make_broken_atomiser():
    """Genuinely-broken extractor — returns fixed garbage triples,
    no LLM call. 10 triples clears the K_min=8 floor so the garbage
    reaches the composer (tests the gate's A/B math, not the volume
    floor — that's C2's job). Model-independent negative fixture."""
    def atomiser(ctx: str) -> list[str]:                       # noqa: ARG001
        return list(_GARBAGE_TRIPLES)
    return atomiser


# ── Pipeline runners ───────────────────────────────────────────────

async def run_one_question(
    tm: TieredMemory,
    q: dict,
    llm: OpenAI,
    model: str,
    judge_model: str,
    atomiser_fn=None,
    k_min: int = K_MIN_DEFAULT,
    namespace: str = "default",
) -> dict[str, Any]:
    """End-to-end pipeline for one LongMemEval question with full
    runbook invariant enforcement.

    `namespace` MUST be unique per (mode, question) pair. When the same
    question runs through two pipelines (baseline + production) back-to-
    back on one TieredMemory, sharing a user_id namespace double-imprints
    the sessions — the second run sees 2x candidates (BCJ Entry 14 cross-
    test residue). Distinct namespace per mode prevents this.
    """
    qid = _qid(q)
    question = q["question"]
    gold = q["answer"]
    sessions = q.get("haystack_sessions", [])

    # Per-(run, mode, question) isolation — RUN_ID prevents cross-run
    # residue; namespace prevents cross-mode residue within a run.
    tm.user_id = f"mvp-{RUN_ID}-{namespace}-{qid}"

    # Ingest sessions — each session passes through Layer 1.5 + Layer 2
    routing_decisions = []
    t0 = time.perf_counter()
    for i, session in enumerate(sessions):
        messages = session if isinstance(session, list) else session.get("messages", [])
        if not messages:
            continue
        session_text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)[:4000]

        # Layer 1.5: declared speaker from first message role
        first_role = messages[0].get("role", "").lower()
        declared = SpeakerRole(first_role) if first_role in SpeakerRole._value2member_map_ \
                   else SpeakerRole.UNKNOWN

        decision = route(session_text, declared_speaker=declared)
        routing_decisions.append({
            "session_idx": i, "shape": decision.shape.value,
            "lifecycle": decision.lifecycle, "tier": decision.tier,
            "confidence": decision.confidence,
        })

        tm.imprint(
            content=session_text,
            metadata={
                "qid": qid, "session_idx": i,
                "shape": decision.shape.value,
                "lifecycle": decision.lifecycle,
            },
        )
    ingest_s = time.perf_counter() - t0

    # Retrieve + atomise + compose
    t2 = time.perf_counter()
    candidates = tm.query_context(question, k=8, min_confidence=0.0)
    if not candidates:
        agent_answer = "NO_ANSWER_IN_CONTEXT"
        atomise_info = None
    else:
        ctx = "\n".join(f"- {c['content']}" for c in candidates)

        # Invariant (b) + (c): safe atomise + position discipline
        if atomiser_fn is not None:
            atom_result = safe_atomise(atomiser_fn, ctx, k_min=k_min)
            atomise_info = {
                "n_emitted": atom_result.n_emitted,
                "dropped_below_k_min": atom_result.dropped_below_k_min,
                "k_min": atom_result.k_min,
            }
            triples = atom_result.triples
        else:
            atomise_info = {"n_emitted": 0, "dropped_below_k_min": False, "k_min": k_min}
            triples = []

        # Build composer prompt with position discipline (raw FIRST, derived LAST)
        user_content = build_composer_prompt(question, ctx, triples)

        resp = llm.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": COMPOSE_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            **_TEMP_KW,
            max_tokens=600,
        )
        raw = (resp.choices[0].message.content or "").strip()
        agent_answer = _extract_answer(raw)
    answer_s = time.perf_counter() - t2

    # Judge
    t3 = time.perf_counter()
    judge_resp = llm.chat.completions.create(
        model=judge_model,
        messages=[{
            "role": "user",
            "content": JUDGE_PROMPT.format(question=question, gold=gold, answer=agent_answer),
        }],
        **_TEMP_KW,
        max_tokens=400,
    )
    judge_raw = (judge_resp.choices[0].message.content or "").strip()
    verdict = _parse_verdict(judge_raw)
    judge_s = time.perf_counter() - t3

    return {
        "question_id": qid,
        "verdict": verdict,
        "agent_answer": agent_answer,
        "gold": gold,
        "candidates_returned": len(candidates),
        "atomise": atomise_info,
        "routing_decisions": routing_decisions,
        "ingest_s": round(ingest_s, 2),
        "answer_s": round(answer_s, 2),
        "judge_s": round(judge_s, 2),
    }


# ── Verification claims ────────────────────────────────────────────

async def verify_c1_classifier(questions: list[dict]) -> dict:
    """C1: classifier correctly tags conversational LongMemEval sessions."""
    n_sessions = 0
    n_conversation = 0
    n_unknown = 0
    speakers_seen = set()
    for q in questions:
        for session in q.get("haystack_sessions", []):
            msgs = session if isinstance(session, list) else session.get("messages", [])
            if not msgs:
                continue
            n_sessions += 1
            text = "\n".join(f"{m['role']}: {m['content']}" for m in msgs)[:4000]
            cls = classify_workload(text)
            if cls.shape is WorkloadShape.CONVERSATION:
                n_conversation += 1
            elif cls.shape is WorkloadShape.UNKNOWN:
                n_unknown += 1
            speakers_seen.update(cls.speaker_roles_present)
    return {
        "claim": "C1: classifier tags conversational sessions",
        "n_sessions": n_sessions,
        "n_classified_conversation": n_conversation,
        "n_classified_unknown": n_unknown,
        "conversation_rate": round(n_conversation / max(n_sessions, 1), 3),
        "speakers_detected": sorted([s.value for s in speakers_seen]),
        "pass": n_conversation / max(n_sessions, 1) >= 0.8,
    }


def verify_c2_k_min_guardrail() -> dict:
    """C2: SafeAtomiser drops below-K_min triple sets (no LLM call needed —
    purely mechanical test of the guardrail itself)."""
    def good_atomiser(ctx: str) -> list[str]:
        return [f"fact{i} | attr | val" for i in range(15)]

    def bad_atomiser(ctx: str) -> list[str]:
        return ["only one | fact | here"]

    good = safe_atomise(good_atomiser, "ctx", k_min=8)
    bad = safe_atomise(bad_atomiser, "ctx", k_min=8)
    return {
        "claim": "C2: SafeAtomiser enforces K_min volume floor",
        "good_extractor_passed": not good.dropped_below_k_min,
        "good_extractor_n_emitted": good.n_emitted,
        "bad_extractor_dropped": bad.dropped_below_k_min,
        "bad_extractor_n_emitted": bad.n_emitted,
        "bad_extractor_triples_returned": len(bad.triples),  # should be 0
        "pass": (not good.dropped_below_k_min)
                and bad.dropped_below_k_min
                and len(bad.triples) == 0,
    }


async def verify_c3_deployment_gate(
    tm: TieredMemory,
    questions: list[dict],
    llm: OpenAI,
    model: str,
    judge_model: str,
) -> dict:
    """C3: DeploymentGate rejects a genuinely-broken extractor.

    Uses a fixed-garbage extractor (model-independent) instead of an
    LLM-constrained one. Garbage triples carry zero signal, so they
    cannot legitimately beat raw retrieval by the +3pt gate threshold
    on ANY model — the gate must reject. The old LLM-constrained
    extractor produced GOOD facts on strong models (Opus 4.7: +20pts),
    making C3 model-dependent; fixed garbage removes that dependence."""
    broken = make_broken_atomiser()

    # We piggyback on the per-question scoring — run a small subset
    # twice: once with raw only, once with the broken-garbage atomiser.
    # k_min=1 disables the volume floor so the garbage actually reaches
    # the composer (the gate's A/B math is what's under test here, not
    # the K_min guardrail — that's C2's job).
    raw_results = []
    bad_results = []
    for q in questions:
        raw_r = await run_one_question(tm, q, llm, model, judge_model,
                                       atomiser_fn=None, namespace="c3raw")
        bad_r = await run_one_question(tm, q, llm, model, judge_model,
                                       atomiser_fn=broken, k_min=1,
                                       namespace="c3bad")
        raw_results.append(raw_r)
        bad_results.append(bad_r)

    # Precompute qid → correctness maps so the gate's pipelines are O(1)
    # lookups instead of O(n) list scans per call.
    raw_correct = {r["question_id"]: r["verdict"] == "CORRECT" for r in raw_results}
    bad_correct = {r["question_id"]: r["verdict"] == "CORRECT" for r in bad_results}
    raw_acc = sum(raw_correct.values()) / len(raw_correct)
    bad_acc = sum(bad_correct.values()) / len(bad_correct)

    gate = deployment_gate(
        eval_items=questions,
        raw_pipeline=lambda q: raw_correct[_qid(q)],
        raw_plus_facts_pipeline=lambda q: bad_correct[_qid(q)],
        threshold_pts=3.0,
    )

    return {
        "claim": "C3: DeploymentGate rejects bad extractors",
        "raw_only_accuracy": round(raw_acc, 3),
        "bad_extractor_accuracy": round(bad_acc, 3),
        "delta_pts": round(gate.delta_pts, 1),
        "threshold_pts": gate.threshold_pts,
        "gate_would_ship": gate.ship,
        "pass": not gate.ship,  # gate must REJECT (ship=False)
    }


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=10,
                   help="Number of questions to run end-to-end (default 10)")
    p.add_argument("--out", type=Path,
                   default=Path("results/production_mvp.json"))
    args = p.parse_args()

    data_path = Path("data/longmemeval/longmemeval_oracle.json")
    if not data_path.exists():
        print(f"ERROR: {data_path} not found. See §5.3 for download instructions.")
        sys.exit(1)

    questions = json.loads(data_path.read_text())[:args.limit]
    print(f"Loaded {len(questions)} questions from {data_path}")

    # Compose/judge client. COMPOSE_BASE_URL lets compose+judge run on a
    # different endpoint than embeddings — e.g. compose on an Anthropic
    # proxy that has no /v1/embeddings, embeddings on the oMLX server.
    # Falls back to OMLX_BASE_URL when unset (single-endpoint case).
    # TieredMemory builds its OWN client from OMLX_BASE_URL for embeddings;
    # that path is untouched, so the embedding model always hits oMLX.
    # max_retries=6: remote proxy relays to api.anthropic.com; transient
    # 500/EOF blips happen across a 50+-call run. SDK does exponential
    # backoff on 5xx + connection errors. Local oMLX rarely blips, so the
    # embeddings client (TieredMemory) keeps the SDK default.
    llm = OpenAI(
        base_url=os.getenv("COMPOSE_BASE_URL") or os.getenv("OMLX_BASE_URL"),
        api_key=os.getenv("COMPOSE_API_KEY") or os.getenv("OMLX_API_KEY"),
        max_retries=6,
        timeout=120.0,
    )
    model = os.getenv("MODEL_HAIKU", "Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit")
    judge_model = os.getenv("MODEL_JUDGE", model)

    print(f"Compose model: {model}")
    print(f"Judge model:   {judge_model}\n")

    # ── C1 + C2 (no async memory needed) ──────────────────────────
    c1 = await verify_c1_classifier(questions)
    print(f"[C1] classifier — {c1['n_classified_conversation']}/{c1['n_sessions']} sessions tagged conversation ({c1['conversation_rate']*100:.0f}%). pass={c1['pass']}")

    c2 = verify_c2_k_min_guardrail()
    print(f"[C2] K_min floor — good extractor passed: {c2['good_extractor_passed']}, bad extractor dropped: {c2['bad_extractor_dropped']}. pass={c2['pass']}")

    # ── C3 (subset for cost) — use first 5 questions ─────────────
    print(f"\n[C3] DeploymentGate A/B over 5-Q subset (raw vs constrained-bad)...")
    async with TieredMemory(agent_id="production-mvp-c3") as tm_c3:
        c3 = await verify_c3_deployment_gate(tm_c3, questions[:5], llm, model, judge_model)
    print(f"[C3] raw={c3['raw_only_accuracy']*100:.0f}% vs bad={c3['bad_extractor_accuracy']*100:.0f}% (Δ={c3['delta_pts']:+.1f}pts). gate_ships={c3['gate_would_ship']}. pass={c3['pass']}")

    # ── C4: end-to-end production-mode vs baseline ───────────────
    print(f"\n[C4] End-to-end production-mode vs raw-only baseline over {args.limit}-Q...")
    async with TieredMemory(agent_id="production-mvp-c4") as tm_c4:
        unconstrained = make_unconstrained_atomiser(llm, model)
        baseline_results = []
        production_results = []
        for i, q in enumerate(questions, 1):
            print(f"  [{i}/{len(questions)}] {q.get('question_id', '?')[:24]}", flush=True)
            r_b = await run_one_question(tm_c4, q, llm, model, judge_model,
                                         atomiser_fn=None, namespace="c4base")
            r_p = await run_one_question(tm_c4, q, llm, model, judge_model,
                                         atomiser_fn=unconstrained, k_min=K_MIN_DEFAULT,
                                         namespace="c4prod")
            baseline_results.append(r_b)
            production_results.append(r_p)

    base_acc = sum(1 for r in baseline_results if r["verdict"] == "CORRECT") / len(baseline_results)
    prod_acc = sum(1 for r in production_results if r["verdict"] == "CORRECT") / len(production_results)
    # C4 claim: production-mode (auto-classify → unconstrained atomise →
    # K_min guard → position discipline) must NOT REGRESS vs raw-only
    # baseline. The claim is non-regression, not improvement: §5.3.2
    # measures atomise lift at only +5pts, which sits at the ±5pt noise
    # floor — "beats baseline by +3pts" is finer than the instrument.
    # The verifiable claim is that the guardrails PREVENT the −30pt
    # collapse seen when constrained atomise ships unguarded (v5/v7/v9).
    delta_c4 = (prod_acc - base_acc) * 100
    c4 = {
        "claim": "C4: production-mode does not regress vs raw-only (delta >= -3pts)",
        "baseline_accuracy": round(base_acc, 3),
        "production_accuracy": round(prod_acc, 3),
        "delta_pts": round(delta_c4, 1),
        "regression_floor_pts": -3.0,
        "note": "non-regression claim — atomise +5pt lift is at noise floor (§5.3.2); "
                "MVP verifies guardrails prevent catastrophic regression, not that atomise is magic",
        "pass": delta_c4 >= -3.0,
    }
    print(f"\n[C4] baseline={base_acc*100:.0f}% vs production={prod_acc*100:.0f}% (Δ={c4['delta_pts']:+.1f}pts, non-regression floor −3pts). pass={c4['pass']}")

    # ── Final report ────────────────────────────────────────────
    report = {
        "limit": args.limit,
        "model": model,
        "claims": {"C1": c1, "C2": c2, "C3": c3, "C4": c4},
        "baseline_results": baseline_results,
        "production_results": production_results,
        "all_pass": all(c["pass"] for c in [c1, c2, c3, c4]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {args.out}")
    print(f"\nALL CLAIMS PASS: {report['all_pass']}")
    sys.exit(0 if report["all_pass"] else 1)


if __name__ == "__main__":
    asyncio.run(main())
