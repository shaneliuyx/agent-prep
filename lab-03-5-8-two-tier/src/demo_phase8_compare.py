"""Phase 8 — Side-by-side retrieval comparison: EverCore vs Qdrant on Bucket-1 data.

Imprints the same 3 multi-turn dialogues into BOTH backends:
  - EverCore via tiered_memory.TieredMemory (Bucket 1, conversation-shape pipeline)
  - Qdrant   via tiered_memory_qdrant.TieredMemory (Bucket 2, raw vector store)

For Qdrant, each user-assistant turn-pair becomes one imprint (no
extraction pipeline, no profile aggregation — just vector + payload).

After imprint, run the same 3 queries through BOTH backends and print
side-by-side results so the reader can SEE what Bucket 1 gives you that
Bucket 2 doesn't:
  - EverCore returns the per-user EPISODE SUMMARY + (if extracted) profile facts
  - Qdrant returns the nearest TURN-PAIR fragments (no synthesis)

This is the chapter's "Phase 8 vs Phase 7 retrieval-quality comparison"
deliverable. Numbers from a real run feed back into Production Considerations.

Run:
  python -m src.demo_phase8_compare
"""
import asyncio
import time
import uuid

from src.demo_conversational_imprint import DIALOGUES, get_episodes, imprint_dialogue
from src.tiered_memory_qdrant import TieredMemory as QdrantTM


QUERIES = [
    ("alice", "What does Alice care about?"),
    ("bob", "What did Bob ask about auth tokens?"),
    ("carol", "What is the Sev1 MTTR target Carol learned?"),
]


def _short(text: str, n: int = 120) -> str:
    return (text or "")[:n].replace("\n", " ")


async def main() -> None:
    suffix = uuid.uuid4().hex[:6]
    print(f">>> Phase 8 side-by-side: EverCore vs Qdrant on Bucket-1 dialogues (suffix={suffix})\n")

    # ── PHASE 8 path: imprint via EverCore (uses Phase 8 demo's imprint_dialogue) ──
    ec_walls = []
    seeded = []
    print(">>> EverCore (Bucket-1, conversation-shape pipeline)")
    for d in DIALOGUES:
        local = dict(d)
        local["user_id"] = f"{d['user_id']}-{suffix}"
        local["session_id"] = f"{d['session_id']}-{suffix}"
        wall, resp, flush_resp = imprint_dialogue(local)
        ec_walls.append(wall)
        seeded.append(local)
        print(f"  {local['user_id']:25s} wall={wall:.1f}s  flush={flush_resp['data'].get('status', '?')}")

    print(f"\n  Mean EverCore imprint wall: {sum(ec_walls)/len(ec_walls):.1f}s")

    # ── PHASE 7 path: imprint same dialogues into Qdrant turn-by-turn ──
    print("\n>>> Qdrant (Bucket-2 style — each turn as one fact, no extraction)")
    qdrant_walls = []
    async with QdrantTM(agent_id=f"phase8-compare-{suffix}") as qtm:
        for d in DIALOGUES:
            local = dict(d)
            local["user_id"] = f"{d['user_id']}-{suffix}"
            t0 = time.perf_counter()
            for i, (role, content) in enumerate(local["turns"]):
                qtm.imprint(
                    content=f"{role}: {content}",
                    metadata={
                        "type": "observation" if role == "user" else "fact",
                        "user_id": local["user_id"],  # override SHARED for fair comparison
                        "subject": local["topic"],
                        "turn_idx": i,
                    },
                )
            wall = time.perf_counter() - t0
            qdrant_walls.append(wall)
            print(f"  {local['user_id']:25s} {len(local['turns'])} turns  wall={wall*1000:.0f}ms")

    print(f"\n  Mean Qdrant imprint wall (all turns): {sum(qdrant_walls)*1000/len(qdrant_walls):.0f}ms")
    speedup = (sum(ec_walls)/len(ec_walls)) / (sum(qdrant_walls)/len(qdrant_walls))
    print(f"  Qdrant per-dialogue imprint is {speedup:.0f}x faster than EverCore")

    print("\n>>> Waiting 60s for EverCore async extraction...\n")
    time.sleep(60)

    # ── Side-by-side retrieval ──
    print(">>> Side-by-side retrieval (same 3 queries, both backends)\n")

    # Need a Qdrant session for queries
    async with QdrantTM(agent_id=f"phase8-compare-q-{suffix}") as qtm:
        for user_short, query in QUERIES:
            uid = f"{user_short}-{suffix}"
            print(f"  Q: {query!r}")

            # EverCore via per-user episode lookup
            eps = get_episodes(uid)
            if eps:
                ep = eps[0]
                print(f"    EverCore (Bucket 1 — episode summary):")
                print(f"      subject: {_short(ep.get('subject'))}")
                print(f"      summary: {_short(ep.get('summary') or ep.get('episode'))}")
            else:
                print(f"    EverCore (Bucket 1): no episode for {uid}")

            # Qdrant via cosine search on shared collection
            # Note: we imprinted under uid as user_id (not 'shared') to make
            # the per-user comparison fair. Have to override the wrapper's
            # default shared user_id by querying directly.
            qhits_all = qtm.query_context(query=query, k=5)
            # Filter to this user_id since the wrapper auto-applied 'shared'
            # by default in __init__ — we need to manually filter post-hoc:
            qhits = [h for h in qhits_all if h.get("user_id") == uid][:3]
            print(f"    Qdrant (Bucket 2 — top-k nearest turns):")
            if not qhits:
                # Wrapper's default user_id='shared' didn't match our 'alice-xxx'.
                # Re-query with explicit override via the raw http path.
                import httpx, os
                from openai import OpenAI
                llm = OpenAI(base_url=os.getenv("OMLX_BASE_URL"), api_key=os.getenv("OMLX_API_KEY"))
                vec = llm.embeddings.create(model=os.getenv("MODEL_EMBED", "bge-m3-mlx-fp16"), input=query).data[0].embedding
                with httpx.Client(base_url="http://localhost:6333", timeout=10.0) as c:
                    r = c.post(
                        "/collections/lab358_memories/points/search",
                        json={
                            "vector": vec,
                            "limit": 3,
                            "filter": {"must": [{"key": "user_id", "match": {"value": uid}}]},
                            "with_payload": True,
                        },
                    )
                    r.raise_for_status()
                    qhits = [
                        {"content": h["payload"]["content"], "score": h["score"], **h["payload"]}
                        for h in r.json().get("result", [])
                    ]
            for h in qhits[:3]:
                print(f"      score={h.get('score'):.3f} type={h.get('type','?'):11s} {_short(h.get('content'))}")
            print()

    print(">>> Summary table")
    print(f"  EverCore mean imprint wall: {sum(ec_walls)/len(ec_walls):.1f}s per dialogue")
    print(f"  Qdrant   mean imprint wall: {sum(qdrant_walls)*1000/len(qdrant_walls):.0f}ms per dialogue ({len(DIALOGUES[0]['turns'])} turns)")
    print(f"  Imprint speedup (Qdrant / EverCore): {speedup:.0f}x")
    print(f"  EverCore retrieval shape: per-user episode + profile (synthesised)")
    print(f"  Qdrant   retrieval shape: top-k nearest turn-pairs (no synthesis)")


if __name__ == "__main__":
    asyncio.run(main())
