"""Root-cause probe — is the 25% abstention collapse model capability or a bug?

For a set of abstaining qids, this script:
  1. Imprints the haystack sessions (same as the runner)
  2. Retrieves candidates (same k=8)
  3. Checks whether the GOLD answer text actually appears in retrieved
     context — rules out "retrieval failed, model correctly abstained"
  4. Dumps the RAW compose output (not just the <answer> extraction) —
     rules out "model answered in CoT, parser grabbed wrong span"

If retrieval surfaces the gold fact AND the model still emits
NO_ANSWER_IN_CONTEXT in its raw output → model capability, confirmed.

Usage:
    uv run python scripts/probe_abstention.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI

from scripts.run_longmemeval_oracle import COMPOSE_SYSTEM
from src.tiered_memory_qdrant import TieredMemory

# 3 qids that abstained in the 35B-A3B run — picked to span question types:
# direct-fact, multi-hop ordering, temporal-arithmetic.
PROBE_QIDS = ["gpt4_2655b836", "gpt4_2312f94c", "2a1811e2"]


async def probe() -> None:
    oracle = json.loads(Path("data/longmemeval/longmemeval_oracle.json").read_text())
    by_id = {q["question_id"]: q for q in oracle}

    llm = OpenAI(base_url=os.getenv("OMLX_BASE_URL"), api_key=os.getenv("OMLX_API_KEY"))
    model = os.getenv("MODEL_HAIKU", "MLX-Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled-4bit")
    print(f"Probe model: {model}\n")

    async with TieredMemory(agent_id="probe-abstention") as tm:
        for qid in PROBE_QIDS:
            q = by_id[qid]
            question, gold = q["question"], q["answer"]
            sessions = q.get("haystack_sessions", [])

            tm.user_id = f"probe-{qid}"
            for i, session in enumerate(sessions):
                msgs = session if isinstance(session, list) else session.get("messages", [])
                if not msgs:
                    continue
                text = "\n".join(f"{m['role']}: {m['content']}" for m in msgs)[:4000]
                tm.imprint(content=text, metadata={"qid": qid, "session_idx": i})

            candidates = tm.query_context(question, k=8, min_confidence=0.0)
            ctx = "\n".join(f"- {c['content']}" for c in candidates)

            # Retrieval sanity — does the gold answer's distinctive token
            # appear anywhere in retrieved context?
            gold_core = gold.split(".")[0].strip().lower()
            gold_tokens = [t for t in gold_core.replace("'", "").split() if len(t) > 3]
            hits = [t for t in gold_tokens if t in ctx.lower()]

            print("=" * 72)
            print(f"QID:  {qid}")
            print(f"Q:    {question}")
            print(f"GOLD: {gold}")
            print(f"candidates_returned: {len(candidates)}")
            print(f"gold tokens {gold_tokens} → found in ctx: {hits} "
                  f"({len(hits)}/{len(gold_tokens)})")

            resp = llm.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": COMPOSE_SYSTEM},
                    {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {question}"},
                ],
                temperature=0.0,
                max_tokens=600,
            )
            raw = (resp.choices[0].message.content or "").strip()
            finish = resp.choices[0].finish_reason
            print(f"\nRAW COMPOSE OUTPUT (finish_reason={finish}, len={len(raw)}):")
            print(raw)
            print()


if __name__ == "__main__":
    asyncio.run(probe())
