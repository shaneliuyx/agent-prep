"""W3.5 15-Q multi-agent recall benchmark — 4-way comparison.

Backends:
  no_memory      — baseline; zero retrieval (~0% expected; chapter
                   prediction ~10% assumes LLM-compose, this harness
                   is retrieval-only per chapter §5.1 shape)
  guild_only     — operational tier only; raw scrolls via
                   list_closed_quests + get_scroll; no semantic search
  semantic_only  — semantic tier only (Qdrant + bge-m3); no guild,
                   no atomic-claim. Substitutes for EverCore per
                   chapter §5.3.1 architectural-equivalence claim
                   (EverCore would take ~12hr at 30-100s per imprint)
  two_tier      — full architecture: guild scrolls + consolidate
                   into Qdrant + query both tiers

Pass criterion: expected keyword (case-insensitive) appears in the
concatenated retrieved memory text.

Run: uv run python scripts/run_four_way_bench.py --backend two_tier
     uv run python scripts/run_four_way_bench.py --backend all
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

# Bootstrap — let this script import src.* from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# .env autoload — walks from cwd up; finds parent ~/code/agent-prep/.env
# even when no lab-local .env exists. Must run BEFORE src.* imports because
# tiered_memory_qdrant reads OMLX_API_KEY at TieredMemory.__init__ time.
try:
    from dotenv import find_dotenv, load_dotenv
    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    pass

from src.consolidation import consolidate
from src.tiered_memory_qdrant import TieredMemory

# 15 single-agent cross-session recall probes lifted from W3.5
# test_recall.py + extended with multi-agent-flavored variants per
# chapter §5.1's "add the remaining 13 with multi-agent variants."
# Each tuple: (seed_content, query, expected_keyword_lowercase)
PROBES: list[tuple[str, str, str]] = [
    # P01-04: basic semantic recall lifted from W3.5
    ("I live in Osaka.", "where do I live?", "osaka"),
    ("I'm vegan.", "any food restrictions?", "vegan"),
    ("I work as a cloud infrastructure engineer.", "what's my profession?", "engineer"),
    ("I ride my bicycle to work every day.", "what's my hobby?", "bicycle"),
    # P05: multi-fact composition
    ("I'm vegan and I live in Taipei.", "where do I live?", "taipei"),
    # P06: allergy / cross-session recall flavor
    ("I'm allergic to peanuts.", "any food I should avoid?", "peanut"),
    # P07: episodic relevance via paraphrase
    ("I asked about LangGraph for customer-support agent.",
     "tell me more about agent frameworks", "langgraph"),
    # P08: synonym query
    ("I currently reside in Seoul.", "my city of residence", "seoul"),
    # P09-15: multi-agent-shape probes (operational scrolls + cross-agent)
    ("I'm building a customer-support chatbot in Python.",
     "what project am I building?", "chatbot"),
    ("I prefer pytest for testing over unittest.",
     "what test framework should I use?", "pytest"),
    ("We deploy via Terraform IaC with apply-on-merge.",
     "how do we deploy?", "terraform"),
    ("The auth refactor must ship by 2026-06-15.",
     "when is the auth refactor due?", "2026-06-15"),
    ("Our API uses FastAPI with Pydantic v2.",
     "what API framework do we use?", "fastapi"),
    ("Last week's outage was caused by stale Redis TTLs.",
     "what caused last week's outage?", "redis"),
    ("Our team uses semantic commits with conventional-commit prefixes.",
     "what commit format do we use?", "semantic"),
]


def _score(retrieved_text: str, expected: str) -> bool:
    return expected.lower() in retrieved_text.lower()


async def run_no_memory(probes: list[tuple[str, str, str]]) -> dict:
    """Baseline — no retrieval at all. Pass-rate is 0% by construction;
    chapter's ~10% prediction assumed an LLM compose step that this
    retrieval-only harness omits."""
    pass_count = 0
    per_q = []
    for _seed, query, expected in probes:
        retrieved = ""  # no memory, no retrieval
        ok = _score(retrieved, expected)
        per_q.append({"query": query, "expected": expected, "pass": ok})
        pass_count += int(ok)
    return {
        "backend": "no_memory",
        "total": len(probes),
        "pass": pass_count,
        "pass_rate": pass_count / len(probes),
        "per_q": per_q,
    }


async def run_guild_only(probes: list[tuple[str, str, str]], campaign: str) -> dict:
    """Operational tier only — raw scrolls via guild quest_list + quest_scroll.
    No Qdrant, no embedding, no consolidate. Keyword match against
    concatenated raw scroll text."""
    pass_count = 0
    per_q = []
    quest_ids: list[str] = []

    async with TieredMemory(
        agent_id=f"bench-guild-only-{campaign[-8:]}",
        user_id=f"bench-user-{campaign[-8:]}",
    ) as tm:
        # Seed phase: one quest per probe, all tagged with campaign
        for i, (seed, _, _) in enumerate(probes):
            qid = await tm.post_task(subject=f"probe_{i:02d}", campaign=campaign)
            await tm.claim_task(qid)
            await tm.complete_task(qid, report=seed)
            quest_ids.append(qid)

        # Retrieval phase: pull all closed scrolls under campaign + keyword match
        list_text = await tm.list_closed_quests(campaign=campaign)
        all_scrolls_text = list_text
        for qid in quest_ids:
            try:
                all_scrolls_text += " " + await tm.get_scroll(qid)
            except Exception as e:
                all_scrolls_text += f" [scroll-fetch-error: {e}]"

        for _, query, expected in probes:
            ok = _score(all_scrolls_text, expected)
            per_q.append({"query": query, "expected": expected, "pass": ok})
            pass_count += int(ok)

    return {
        "backend": "guild_only",
        "campaign": campaign,
        "total": len(probes),
        "pass": pass_count,
        "pass_rate": pass_count / len(probes),
        "per_q": per_q,
    }


async def run_semantic_only(probes: list[tuple[str, str, str]], campaign: str) -> dict:
    """Semantic tier only (Qdrant + bge-m3). No guild, no atomic-claim.
    imprint() each seed directly; query_context() with k=3 for each probe.
    Substitutes for EverCore per chapter §5.3.1; same architectural
    contract ("semantic store without operational tier")."""
    pass_count = 0
    per_q = []

    async with TieredMemory(
        agent_id=f"bench-semantic-only-{campaign[-8:]}",
        user_id=f"bench-user-sem-{campaign[-8:]}",
    ) as tm:
        # Seed phase: direct imprint to Qdrant; no guild scrolls written
        for i, (seed, _, _) in enumerate(probes):
            tm.imprint(content=seed, metadata={"probe_idx": i, "campaign": campaign})

        # Retrieval phase: vector search for each probe
        for _, query, expected in probes:
            candidates = tm.query_context(query=query, k=3)
            retrieved_text = " ".join(c.get("content", "") for c in candidates)
            ok = _score(retrieved_text, expected)
            per_q.append({"query": query, "expected": expected,
                          "n_candidates": len(candidates), "pass": ok})
            pass_count += int(ok)

    return {
        "backend": "semantic_only",
        "campaign": campaign,
        "total": len(probes),
        "pass": pass_count,
        "pass_rate": pass_count / len(probes),
        "per_q": per_q,
    }


async def run_two_tier(probes: list[tuple[str, str, str]], campaign: str) -> dict:
    """Full two-tier architecture. Guild scrolls + consolidate into Qdrant.
    Retrieval probes BOTH tiers: semantic via query_context + raw via scrolls
    (the chapter's union-of-tiers query pattern)."""
    pass_count = 0
    per_q = []
    quest_ids: list[str] = []

    async with TieredMemory(
        agent_id=f"bench-two-tier-{campaign[-8:]}",
        user_id=f"bench-user-2t-{campaign[-8:]}",
    ) as tm:
        # Seed phase: post + claim + complete + consolidate
        for i, (seed, _, _) in enumerate(probes):
            qid = await tm.post_task(subject=f"probe_{i:02d}", campaign=campaign)
            await tm.claim_task(qid)
            await tm.complete_task(qid, report=seed)
            quest_ids.append(qid)

        # Consolidate: pulls closed scrolls → summarizes → imprints to Qdrant
        c_result = await consolidate(tm, campaign=campaign, use_atomisation=False)
        consolidation_summary = {
            "scrolls_seen": c_result.scrolls_seen,
            "scrolls_imprinted": c_result.scrolls_imprinted,
            "scrolls_skipped": c_result.scrolls_skipped,
            "errors": len(c_result.errors) if c_result.errors else 0,
        }

        # Also pre-pull guild scroll text for union retrieval
        list_text = await tm.list_closed_quests(campaign=campaign)
        scroll_corpus = list_text
        for qid in quest_ids:
            try:
                scroll_corpus += " " + await tm.get_scroll(qid)
            except Exception as e:
                scroll_corpus += f" [scroll-fetch-error: {e}]"

        # Retrieval phase: union of (a) Qdrant vector search + (b) raw scroll text
        for _, query, expected in probes:
            candidates = tm.query_context(query=query, k=3)
            semantic_text = " ".join(c.get("content", "") for c in candidates)
            union_text = semantic_text + " " + scroll_corpus
            ok = _score(union_text, expected)
            per_q.append({
                "query": query, "expected": expected,
                "n_semantic": len(candidates), "pass": ok,
            })
            pass_count += int(ok)

    return {
        "backend": "two_tier",
        "campaign": campaign,
        "total": len(probes),
        "pass": pass_count,
        "pass_rate": pass_count / len(probes),
        "consolidation": consolidation_summary,
        "per_q": per_q,
    }


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", required=True,
                   choices=["no_memory", "guild_only", "semantic_only", "two_tier", "all"])
    p.add_argument("--out", default=None, help="JSON output path; default = stdout only")
    args = p.parse_args()

    campaign = f"bench-w358-{int(time.time())}-{uuid.uuid4().hex[:6]}"

    backends = (
        ["no_memory", "guild_only", "semantic_only", "two_tier"]
        if args.backend == "all"
        else [args.backend]
    )

    results = {}
    for b in backends:
        t0 = time.time()
        # Each backend gets its OWN unique campaign so guild/qdrant state
        # doesn't bleed between runs of the bench
        per_backend_campaign = f"{campaign}-{b}"
        if b == "no_memory":
            r = await run_no_memory(PROBES)
        elif b == "guild_only":
            r = await run_guild_only(PROBES, per_backend_campaign)
        elif b == "semantic_only":
            r = await run_semantic_only(PROBES, per_backend_campaign)
        elif b == "two_tier":
            r = await run_two_tier(PROBES, per_backend_campaign)
        else:
            raise ValueError(f"unknown backend: {b}")
        r["wall_s"] = round(time.time() - t0, 2)
        results[b] = r
        print(f"[{b}] {r['pass']}/{r['total']} ({r['pass_rate']:.1%}) in {r['wall_s']}s")

    print()
    print("=== aggregate ===")
    for b, r in results.items():
        print(f"  {b:<16} {r['pass_rate']:.1%}  (pass {r['pass']}/{r['total']}, wall {r['wall_s']}s)")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
