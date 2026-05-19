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


COMPOSE_SYSTEM = """You are answering questions about a user's past conversations.

OUTPUT FORMAT — STRICT:
You may emit reasoning, but the FINAL ANSWER must be wrapped in <answer>...</answer> tags.
The parser reads ONLY what is inside <answer>...</answer>.

RULES:
- Answer in 1-2 sentences using ONLY the context below.
- If the context does not contain the answer, output: <answer>NO_ANSWER_IN_CONTEXT</answer>

EXAMPLE:
Context:
- Yesterday I deployed terraform v1.5 to prod.
- The deployment took 12 minutes.

Question: How long did yesterday's deployment take?

<answer>Twelve minutes.</answer>"""


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
    tm.user_id = f"longmemeval-{qid}"

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
    if not candidates:
        agent_answer = "NO_ANSWER_IN_CONTEXT"
    else:
        ctx = "\\n".join(f"- {c['content']}" for c in candidates)
        resp = llm.chat.completions.create(
            model=os.getenv("MODEL_HAIKU", "Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit"),
            messages=[
                {"role": "system", "content": COMPOSE_SYSTEM},
                {"role": "user",
                 "content": f"Context:\\n{ctx}\\n\\nQuestion: {question}"},
            ],
            temperature=0.0,
            max_tokens=600,
        )
        raw = (resp.choices[0].message.content or "").strip()
        # Extract answer from <answer>...</answer> tags. Reasoning model
        # may emit CoT before the tags; parser reads only what's tagged.
        # Fallback: if no tags, use the last non-empty line.
        m = re.search(r"<answer>(.*?)</answer>", raw, re.DOTALL | re.IGNORECASE)
        if m:
            agent_answer = m.group(1).strip()
        else:
            lines = [l.strip() for l in raw.split("\\n") if l.strip()]
            agent_answer = lines[-1] if lines else raw
    answer_s = time.perf_counter() - t2

    # (d) Score answer via LLM-as-judge.
    # max_tokens=400 leaves room for reasoning-model chain-of-thought
    # prelude AND the final verdict token.
    judge_resp = llm.chat.completions.create(
        model=judge_model,
        messages=[
            {"role": "user",
             "content": JUDGE_PROMPT.format(
                 question=question, gold=gold, answer=agent_answer
             )},
        ],
        temperature=0.0,
        max_tokens=400,
    )
    judge_raw = (judge_resp.choices[0].message.content or "").strip()
    judge_upper = judge_raw.upper()

    # Reasoning models (gpt-oss-20b, distilled-from-reasoning Qwen) emit
    # chain-of-thought BEFORE the verdict. Scan whole response — prefer
    # the LATER occurrence (verdict usually at end of reasoning).
    # "INCORRECT" contains "CORRECT" as substring at offset +2, so the
    # CORRECT-rfind hit at position N+2 corresponds to an INCORRECT
    # match at position N; check INCORRECT first via offset.
    last_correct = judge_upper.rfind("CORRECT")
    last_incorrect = judge_upper.rfind("INCORRECT")
    if last_incorrect > -1 and last_incorrect + 2 >= last_correct:
        verdict = "INCORRECT"
        correct = False
    elif last_correct > -1:
        verdict = "CORRECT"
        correct = True
    else:
        verdict = "UNKNOWN"
        correct = False
        print(f"    [judge-unknown] raw: {judge_raw[:200]!r}")

    return {
        "question_id": qid,
        "question": question,
        "gold": gold,
        "agent_answer": agent_answer,
        "verdict": verdict,
        "judge_raw": judge_raw[:500],   # truncate to keep results JSON small
        "correct": correct,
        "facts_imprinted": facts_imprinted,
        "scrolls_demoted": 0,
        "candidates_returned": len(candidates),
        "ingest_s": round(ingest_s, 2),
        "answer_s": round(answer_s, 2),
    }


async def main(limit: int, campaign: str, out_path: Path) -> None:
    data_path = Path("data/longmemeval/longmemeval_oracle.json")
    questions = json.loads(data_path.read_text())[:limit]

    llm = OpenAI(base_url=os.getenv("OMLX_BASE_URL"), api_key=os.getenv("OMLX_API_KEY"))
    judge_model = os.getenv("MODEL_JUDGE", "Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit")

    async with TieredMemory(agent_id=f"longmemeval-{campaign}") as tm:
        results = []
        for i, q in enumerate(questions, 1):
            print(f"[{i}/{len(questions)}] {q.get('question_id', q['question'][:40])}...", flush=True)
            try:
                r = await run_one_question(tm, q, llm, judge_model)
                results.append(r)
                print(f"  → {r['verdict']} (ingest {r['ingest_s']}s + ans {r['answer_s']}s)")
            except Exception as e:                                       # noqa: BLE001
                print(f"  → ERROR: {type(e).__name__}: {e}")
                results.append({"question_id": q.get("question_id"), "error": str(e), "correct": False})

    n_correct = sum(1 for r in results if r.get("correct"))
    n_total = len(results)
    n_err = sum(1 for r in results if r.get("error"))
    accuracy = n_correct / n_total if n_total else 0.0

    summary = {
        "campaign": campaign,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_questions": n_total,
        "correct": n_correct,
        "errors": n_err,
        "accuracy": round(accuracy, 4),
        "evercore_published": 0.83,
        "delta_vs_evercore": round(accuracy - 0.83, 4),
        "per_question": results,
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\\nFinal: {n_correct}/{n_total} = {accuracy:.1%} (errors: {n_err}). EverCore baseline: 83%. Delta: {summary['delta_vs_evercore']:+.1%}.")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--campaign", type=str, default="longmemeval-oracle")
    p.add_argument("--out", type=Path, default=Path("results/longmemeval_oracle.json"))
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(main(args.limit, args.campaign, args.out))
