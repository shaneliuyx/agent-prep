"""Three-act multi-agent handoff via guild quest+journal+scroll."""
from __future__ import annotations

import asyncio

from src.guild_client import GuildClient

CAMPAIGN = "payments-api-3act"


async def act_one_design() -> str:
    print(">>> Act 1 — agent A designs the API spec")
    async with GuildClient(agent_id="agent_a") as gc:
        quest_id = await gc.quest_post(
            subject="design-api-spec: design the new payments API",
            spec="Define REST endpoints, auth model, retry semantics, idempotency strategy.",
            campaign=CAMPAIGN,
        )
        claim = await gc.quest_accept(quest_id)
        print(f"  claim ({quest_id}): {claim[:80]}")
        await gc.quest_journal(
            quest_id,
            "API spec finalized: REST + JSON, Idempotency-Key header, JWT 30m expiry.",
        )
        await gc.quest_fulfill(
            quest_id,
            report="commit design-001; files: docs/api-spec.md; no remaining issues",
        )
        print(f"  journal logged + quest fulfilled ({quest_id})")
    return quest_id


async def act_two_implement(design_quest_id: str) -> str:
    print(">>> Act 2 — agent B implements based on agent A's design context")
    async with GuildClient(agent_id="agent_b") as gc:
        prior = await gc.quest_scroll(design_quest_id)
        print(f"  read design scroll: {prior[:100]}...")
        quest_id = await gc.quest_post(
            subject="implement-api: implement the payments API",
            spec=f"Implement per design in {design_quest_id}. FastAPI + Pydantic.",
            campaign=CAMPAIGN,
        )
        claim = await gc.quest_accept(quest_id)
        print(f"  claim ({quest_id}): {claim[:80]}")
        await gc.quest_journal(
            quest_id,
            "Implemented POST/GET /payments. FastAPI + Pydantic. JWT middleware. Test stubs added.",
        )
        await gc.quest_fulfill(
            quest_id,
            report="commit impl-001; files: src/api/payments.py, src/middleware/jwt.py, "
                   "tests/test_payments.py; remaining: exhaustive tests in Act 3",
        )
    return quest_id


async def act_three_test(design_quest_id: str, impl_quest_id: str) -> str:
    print(">>> Act 3 — agent C writes tests, sees the WHOLE chain")
    async with GuildClient(agent_id="agent_c") as gc:
        # Read both prior scrolls concurrently — independent reads.
        prior_scrolls = await asyncio.gather(
            gc.quest_scroll(design_quest_id),
            gc.quest_scroll(impl_quest_id),
        )
        for q, scroll in zip((design_quest_id, impl_quest_id), prior_scrolls):
            print(f"  read prior scroll [{q}]: {scroll[:80]}...")
        quest_id = await gc.quest_post(
            subject="write-api-tests: exhaustive payments-API test suite",
            campaign=CAMPAIGN,
        )
        claim = await gc.quest_accept(quest_id)
        print(f"  claim ({quest_id}): {claim[:80]}")
        await gc.quest_journal(
            quest_id,
            "Wrote integration tests: happy-path, idempotency-key replay, "
            "JWT expiry, retry-on-503. Coverage 85%. TODO: clock skew on JWT.",
        )
        await gc.quest_fulfill(
            quest_id,
            report="commit test-001; files: tests/test_payments_integration.py; "
                   "coverage 85%; remaining: clock-skew edge case",
        )
    return quest_id


async def main() -> None:
    design_id = await act_one_design()
    print()
    impl_id = await act_two_implement(design_id)
    print()
    await act_three_test(design_id, impl_id)


if __name__ == "__main__":
    asyncio.run(main())
