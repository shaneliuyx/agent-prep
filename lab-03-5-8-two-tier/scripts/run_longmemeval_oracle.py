"""LongMemEval oracle eval runner — two-tier memory architecture.

Replay haystack sessions into the consolidation pipeline, query the
resulting memory, score the answer against the oracle gold via
LLM-as-judge. Aggregate accuracy + per-question pass/fail.

Run: uv run python scripts/run_longmemeval_oracle.py \\
        --limit 20 --campaign longmemeval-first-2026-05-19
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

# Bootstrap — let this script import `src.*` regardless of the cwd.
# scripts/ lives one level below the lab root; prepend the lab root
# so `from src.tiered_memory import ...` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI

from src.tiered_memory_qdrant import TieredMemory


# Run-unique ID — prevents cross-RUN Qdrant residue. The Qdrant collection
# persists between eval invocations; without a per-run prefix every run
# reuses the `longmemeval-{qid}` namespace and accumulates duplicate
# imprints, inflating k=8 retrieval to bloated multi-run context (BCJ
# Entry 14 cross-test residue — diagnosed 2026-05-20 when a 3B-active MoE
# abstained on the bloat while a probe on a fresh namespace answered
# correctly). RUN_ID makes every run's namespace disjoint.
RUN_ID = str(int(time.time()))

# Opus 4.7 (and other extended-thinking models) DEPRECATE the `temperature`
# parameter — passing it returns HTTP 400. Local 4-bit MLX models want
# temperature=0.0 for deterministic eval. Gate it: set DISABLE_TEMPERATURE=1
# when the compose endpoint is a thinking model.
_TEMP_KW: dict = {} if os.getenv("DISABLE_TEMPERATURE") == "1" else {"temperature": 0.0}


JUDGE_PROMPT = """You are an evaluation judge. Decide if the agent's answer
substantively matches the gold answer. Output the verdict as one word:
CORRECT or INCORRECT (optionally preceded by short reasoning).

Question: {question}
Gold answer: {gold}
Agent answer: {answer}

Rules:
- Paraphrase OK; exact wording not required.
- Missing details = INCORRECT.
- Extra correct details OK (do not penalize).
- Hallucinated wrong details = INCORRECT.
- CRITICAL: if the agent says "I don't know" / "the context does not
  contain the answer" / "no information available" / NO_ANSWER_IN_CONTEXT
  / any abstention, the verdict is INCORRECT — UNLESS the gold answer
  itself is an abstention. Honest abstention is NOT a correct answer
  when the gold is concrete.
- CRITICAL: if the agent emits chain-of-thought reasoning instead of
  a direct answer (e.g., "Thinking Process: 1. Analyze..." or
  numbered analysis steps), the verdict is INCORRECT — the agent
  failed to produce an actual answer.
Output: CORRECT or INCORRECT"""


ATOMISE_SYSTEM = """Extract atomic facts from the conversation excerpts below.

Each fact is a triple in the form:  subject | attribute | value

Focus on:
- Named entities (people, places, products, events)
- Dates and times (absolute or relative — keep verbatim)
- Quantities (counts, durations, prices, sizes)
- Actions (purchased, attended, set up, deployed)
- Relations between entities

Output ONE triple per line. No commentary, no headers, no Markdown.
If a session contains no extractable facts, output nothing for that session.

NOTE: emit MANY triples (15+ per session is fine). Volume buffers extraction
error — empirical 2026-05-20: constrained K=5 atomise caused −30pts on Opus
and −35pts on Qwen3.6-27B due to single-triple anchoring bias in the
downstream composer. See W3.5.8 §5.3.4 for the Bayesian framing.

EXAMPLE INPUT:
- session 1: Bought new Samsung Galaxy S22 today (Feb 20), very excited.
- session 2: My Dell XPS 13 finally arrived (Feb 25).

EXAMPLE OUTPUT:
user | bought | Samsung Galaxy S22
purchase of Samsung Galaxy S22 | date | Feb 20
user | received | Dell XPS 13
arrival of Dell XPS 13 | date | Feb 25"""


COMPOSE_SYSTEM = """You are answering questions about a user's past conversations.

OUTPUT FORMAT — STRICT:
You may emit reasoning, but the FINAL ANSWER must be wrapped in <answer>...</answer> tags.
The parser reads ONLY what is inside <answer>...</answer>.

COMMIT-FIRST RULES:
- DEFAULT to answering. The context usually DOES contain the answer.
- Multi-session reasoning IS part of the task: ordering events, comparing
  dates, counting day-gaps, picking which of two things came first. If the
  facts are present in different sessions, COMBINE them and COMMIT.
- Abstain (<answer>NO_ANSWER_IN_CONTEXT</answer>) ONLY when the context
  is completely unrelated to the question's topic. "I see partial facts but
  am not sure" is NOT a valid reason to abstain — pick the best-supported
  answer and commit.
- Answer in 1-2 sentences using ONLY the context below.

TEMPORAL EXAMPLE (multi-session ordering):
Context:
- Session 2024-02-20: Bought new Samsung Galaxy S22 today, very excited.
- Session 2024-02-25: My Dell XPS 13 finally arrived from the courier.

Question: Which device did I get first, the Samsung Galaxy S22 or the Dell XPS 13?

<answer>The Samsung Galaxy S22 (Feb 20) came before the Dell XPS 13 (Feb 25).</answer>

SIMPLE EXAMPLE:
Context:
- Yesterday I deployed terraform v1.5 to prod.
- The deployment took 12 minutes.

Question: How long did yesterday's deployment take?

<answer>Twelve minutes.</answer>"""


def parse_verdict(judge_raw: str) -> str:
    """Parse a judge response into CORRECT / INCORRECT / UNKNOWN.

    Two-tier. TIER 1: prefer an explicit labeled verdict ("Verdict:
    CORRECT", "**Verdict: INCORRECT**"). Opus-distilled models emit the
    verdict FIRST then reasoning prose, and that prose naturally contains
    the word "incorrect" ("no hallucinated or incorrect information") — a
    whole-text rfind scan grabs the prose word and flips the verdict. A
    labeled verdict is unambiguous wherever it sits.

    TIER 2: fallback whole-text rfind for judges that emit a bare token
    with no label. "INCORRECT" contains "CORRECT" at offset +2, so the
    CORRECT-rfind hit at N+2 corresponds to an INCORRECT match at N;
    check INCORRECT first via offset.

    Single source of truth — imported by scripts/test_llm_io.py so the
    quick-iteration harness scores verdicts exactly as the eval runner.
    """
    m = re.search(r"verdict\s*[:\-]?\s*\**\s*(INCORRECT|CORRECT)\b",
                  judge_raw, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    judge_upper = judge_raw.upper()
    last_correct = judge_upper.rfind("CORRECT")
    last_incorrect = judge_upper.rfind("INCORRECT")
    if last_incorrect > -1 and last_incorrect + 2 >= last_correct:
        return "INCORRECT"
    if last_correct > -1:
        return "CORRECT"
    return "UNKNOWN"


async def run_one_question(tm: TieredMemory, q: dict, llm: OpenAI, judge_model: str) -> dict:
    """Process one LongMemEval question end-to-end. Returns scored result."""
    qid = q.get("question_id") or q["question"][:32]
    question = q["question"]
    gold = q["answer"]
    sessions = q.get("haystack_sessions", [])

    # Per-question isolation: mutate the TieredMemory user_id to a
    # per-question namespace. Qdrant query_context filters on user_id;
    # this prevents Q(N+1) from seeing Q(N)'s imprints (BCJ Entry 14
    # cross-test residue pattern applied to eval runs).
    tm.user_id = f"longmemeval-{RUN_ID}-{qid}"

    # (a) DIRECT IMPRINT each session — bypass consolidate() entirely.
    #
    # Why bypass? src/consolidation.py:SUMMARIZE_PROMPT is biased toward
    # TECHNICAL knowledge extraction ("Production deployments use
    # Terraform IaC pattern...") with explicit SKIP rules for "in-progress
    # notes, failed attempts, debug traces". LongMemEval haystacks are
    # CONVERSATIONAL data (user shares preferences, attends events, has
    # vehicle issues) — the summarizer either SKIPs them or produces
    # tech-flavored summaries that destroy the very details LongMemEval
    # tests for. Diagnosed 2026-05-19: candidates_returned > 0 but agent
    # answer = NO_ANSWER_IN_CONTEXT because summarized content didn't
    # preserve the personal/conversational facts.
    #
    # Direct-imprint preserves session content verbatim. One Qdrant point
    # per session; retrieval works against the original message text.
    t0 = time.perf_counter()
    facts_imprinted = 0
    for i, session in enumerate(sessions):
        messages = session if isinstance(session, list) else session.get("messages", [])
        if not messages:
            continue
        # Concatenate all messages in session into one searchable chunk.
        # bge-m3 handles up to 8192 tokens; truncate at ~4000 chars to stay
        # well within budget while keeping the conversation intact.
        session_text = "\\n".join(f"{m['role']}: {m['content']}" for m in messages)[:4000]
        tm.imprint(
            content=session_text,
            metadata={
                "qid": qid,
                "session_idx": i,
                "source": "longmemeval_haystack",
            },
        )
        facts_imprinted += 1
    ingest_s = time.perf_counter() - t0

    # (c) Query + compose answer.
    t2 = time.perf_counter()
    candidates = tm.query_context(question, k=8, min_confidence=0.0)
    atomise_s = 0.0
    triples = ""
    triple_lines: list[str] = []
    compose_truncated = False
    if not candidates:
        agent_answer = "NO_ANSWER_IN_CONTEXT"
    else:
        ctx = "\\n".join(f"- {c['content']}" for c in candidates)

        # Read-time atomisation: opt-in via ATOMISE_AT_READ env flag.
        # Mirrors §3.2.1's WRITE-time atomisation primitive but applied
        # at READ time. Pre-extracts (subject, attribute, value) triples
        # from retrieved candidates so the composer reasons over
        # structured tuples instead of raw multi-turn dialogue.
        # Trade-off: +1 LLM call (~10s) per question; recovers temporal
        # arithmetic + period-bounded counting failures.
        if os.getenv("ATOMISE_AT_READ", "0") == "1":
            ta = time.perf_counter()
            atom_resp = llm.chat.completions.create(
                model=os.getenv("MODEL_HAIKU", "Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit"),
                messages=[
                    {"role": "system", "content": ATOMISE_SYSTEM},
                    {"role": "user", "content": ctx},
                ],
                **_TEMP_KW,
                max_tokens=600,   # unconstrained: many triples per session
            )
            triples = (atom_resp.choices[0].message.content or "").strip()
            triple_lines = [l for l in triples.splitlines() if l.strip()]
            atomise_s = time.perf_counter() - ta

        if triples and triples.lower() != "(no relevant facts)":
            # PRESERVE raw alongside triples (industry-standard invariant —
            # Mem0/Zep/Letta/Governed Memory all keep raw + derived together).
            # Use NEUTRAL framing — measured 2026-05-20: confidence-boosting
            # labels like "top-N, question-conditioned" caused the composer
            # to over-trust wrong triples (v5: 25%). Match v3's neutral
            # "Atomic facts (structured)" framing.
            user_content = (
                f"Atomic facts (structured):\\n{triples}\\n\\n"
                f"Original context (raw, for fallback):\\n{ctx}\\n\\n"
                f"Question: {question}"
            )
        else:
            user_content = f"Context:\\n{ctx}\\n\\nQuestion: {question}"

        # max_tokens=4000: reasoning-distilled models burn 600+ tokens on
        # chain-of-thought before emitting <answer>. Measured 2026-05-20:
        # at 600 a CoT-heavy model truncated mid-reasoning; 1500 → 2/20
        # truncated; 2000 → 1/20 truncated. 4000 gives full CoT headroom
        # so finish_reason=length never silently corrupts a verdict.
        resp = llm.chat.completions.create(
            model=os.getenv("MODEL_HAIKU", "Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit"),
            messages=[
                {"role": "system", "content": COMPOSE_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            **_TEMP_KW,
            max_tokens=4000,
        )
        compose_truncated = resp.choices[0].finish_reason == "length"
        raw = (resp.choices[0].message.content or "").strip()
        # Extract answer from <answer>...</answer> tags. Reasoning models
        # may emit CoT that echoes the prompt's "<answer>...</answer>"
        # template description, so take the LAST match (real answer)
        # rather than the first (template echo). Reject sentinel `...`
        # and empty content; fall back to last non-empty line.
        matches = re.findall(r"<answer>(.*?)</answer>", raw, re.DOTALL | re.IGNORECASE)
        agent_answer = ""
        for cand in reversed(matches):
            cand = cand.strip()
            if cand and cand != "...":
                agent_answer = cand
                break
        if not agent_answer:
            lines = [l.strip() for l in raw.split("\n") if l.strip()]
            agent_answer = lines[-1] if lines else raw
    answer_s = time.perf_counter() - t2

    # (d) Score answer via LLM-as-judge.
    # max_tokens=1000: verbose self-judging models (e.g. an Opus-distilled
    # model judging its own answers) write multi-step "**Analysis:** 1...
    # 2... 3. Verification:" prose and hit a 400-token cap BEFORE emitting
    # the verdict — diagnosed 2026-05-20 (2/20 UNKNOWN verdicts on a clean
    # run, judge_raw truncated mid-reasoning). 1000 covers the verbose tail.
    judge_resp = llm.chat.completions.create(
        model=judge_model,
        messages=[
            {"role": "user",
             "content": JUDGE_PROMPT.format(
                 question=question, gold=gold, answer=agent_answer
             )},
        ],
        **_TEMP_KW,
        max_tokens=1000,
    )
    judge_truncated = judge_resp.choices[0].finish_reason == "length"
    judge_raw = (judge_resp.choices[0].message.content or "").strip()

    verdict = parse_verdict(judge_raw)   # see parse_verdict() for the two-tier logic
    correct = verdict == "CORRECT"
    if verdict == "UNKNOWN":
        print(f"    [judge-unknown] raw: {judge_raw[:200]!r}")

    return {
        "question_id": qid,
        "question": question,
        "gold": gold,
        "agent_answer": agent_answer,
        "verdict": verdict,
        "judge_raw": judge_raw[:500],   # truncate to keep results JSON small
        "correct": correct,
        "compose_truncated": compose_truncated,
        "judge_truncated": judge_truncated,
        "facts_imprinted": facts_imprinted,
        "scrolls_demoted": 0,
        "candidates_returned": len(candidates),
        "ingest_s": round(ingest_s, 2),
        "atomise_s": round(atomise_s, 2),
        "answer_s": round(answer_s, 2),
        "triples_emitted": len(triples.splitlines()) if triples else 0,
    }


async def main(limit: int, campaign: str, out_path: Path) -> None:
    data_path = Path("data/longmemeval/longmemeval_oracle.json")
    questions = json.loads(data_path.read_text())[:limit]

    # timeout + retries: a transient oMLX hiccup (model auto-eviction,
    # server KeyError) otherwise hangs the run forever — the default
    # OpenAI client has no timeout. Diagnosed 2026-05-20 when a 5-model
    # matrix run hung indefinitely on Q1 after oMLX unloaded the model
    # mid-request. 300s covers a cold model load + a long CoT compose;
    # 2 retries ride out a single transient. A genuinely hung request
    # now raises, the per-question try/except logs it as an error, and
    # the run proceeds instead of deadlocking.
    # Compose/judge client. COMPOSE_BASE_URL lets compose+judge run on a
    # different endpoint than embeddings — e.g. an Anthropic proxy that
    # has no /v1/embeddings route. Falls back to OMLX_BASE_URL when unset.
    # TieredMemory builds its OWN embedding client from OMLX_BASE_URL, so
    # the embedding model always stays on oMLX regardless of this split.
    llm = OpenAI(
        base_url=os.getenv("COMPOSE_BASE_URL") or os.getenv("OMLX_BASE_URL"),
        api_key=os.getenv("COMPOSE_API_KEY") or os.getenv("OMLX_API_KEY"),
        timeout=300.0,
        max_retries=10,   # bumped 6→10 (2026-05-21): a remote Anthropic
        # proxy N=100 run hit 23 transient 500/EOF errors — 6 retries
        # could not ride out a sustained upstream wobble. 10 retries with
        # exponential backoff spans a longer bad window.
    )
    judge_model = os.getenv("MODEL_JUDGE", "Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit")

    async with TieredMemory(agent_id=f"longmemeval-{campaign}") as tm:
        results = []
        for i, q in enumerate(questions, 1):
            print(f"[{i}/{len(questions)}] {q.get('question_id', q['question'][:40])}...", flush=True)
            try:
                r = await run_one_question(tm, q, llm, judge_model)
                results.append(r)
                atom_str = f" + atom {r['atomise_s']}s ({r['triples_emitted']} triples)" if r.get('atomise_s', 0) > 0 else ""
                trunc_str = "  [!] compose TRUNCATED (finish_reason=length)" if r.get('compose_truncated') else ""
                print(f"  → {r['verdict']} (ingest {r['ingest_s']}s{atom_str} + ans {r['answer_s']}s){trunc_str}")
            except Exception as e:                                       # noqa: BLE001
                print(f"  → ERROR: {type(e).__name__}: {e}")
                results.append({"question_id": q.get("question_id"), "error": str(e), "correct": False})

    n_correct = sum(1 for r in results if r.get("correct"))
    n_total = len(results)
    n_err = sum(1 for r in results if r.get("error"))
    n_trunc = sum(1 for r in results if r.get("compose_truncated"))
    n_judge_trunc = sum(1 for r in results if r.get("judge_truncated"))
    accuracy = n_correct / n_total if n_total else 0.0

    summary = {
        "campaign": campaign,
        "run_id": RUN_ID,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_questions": n_total,
        "correct": n_correct,
        "errors": n_err,
        "compose_truncations": n_trunc,
        "judge_truncations": n_judge_trunc,
        "accuracy": round(accuracy, 4),
        "evercore_published": 0.83,
        "delta_vs_evercore": round(accuracy - 0.83, 4),
        "per_question": results,
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\\nFinal: {n_correct}/{n_total} = {accuracy:.1%} (errors: {n_err}). EverCore baseline: 83%. Delta: {summary['delta_vs_evercore']:+.1%}.")
    if n_trunc:
        print(f"[!] {n_trunc}/{n_total} compose calls TRUNCATED (finish_reason=length) — accuracy is a LOWER BOUND; raise max_tokens and re-run.")
    if n_judge_trunc:
        print(f"[!] {n_judge_trunc}/{n_total} JUDGE calls TRUNCATED — verdicts unreliable; raise judge max_tokens and re-run.")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--campaign", type=str, default="longmemeval-oracle")
    p.add_argument("--out", type=Path, default=Path("results/longmemeval_oracle.json"))
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(main(args.limit, args.campaign, args.out))
