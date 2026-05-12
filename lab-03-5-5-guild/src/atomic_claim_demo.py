"""Two parallel Python processes race to claim the SAME quest.
Exactly one should win; the other receives a clear rejection.

Two-phase pattern to avoid the find-or-create timing footgun:
  Phase 1 — seed:  create the race quest ONCE, capture its QUEST_ID.
  Phase 2 — race:  spawn alice + bob simultaneously, both attempting
                    to accept the SAME pre-existing QUEST_ID.

Usage:
  python -m src.atomic_claim_demo seed
  python -m src.atomic_claim_demo race QUEST-N alice &
  python -m src.atomic_claim_demo race QUEST-N bob &
  wait
"""
from __future__ import annotations

import asyncio
import sys

from src.guild_client import GuildClient, is_accept_winner

USAGE = "usage: python -m src.atomic_claim_demo {seed|race QUEST-N agent_id}"


async def seed() -> str:
    async with GuildClient(agent_id="seed") as gc:
        quest_id = await gc.quest_post(
            subject="race-the-prize: both agents want this",
            campaign="race-demo",
        )
        print(quest_id)
        return quest_id


async def race(quest_id: str, agent_id: str) -> None:
    async with GuildClient(agent_id=agent_id) as gc:
        print(f"[{agent_id}] attempting claim on {quest_id}...")
        result = await gc.quest_accept(quest_id)
        print(f"[{agent_id}] result: {result[:120]}")

        if is_accept_winner(result):
            print(f"[{agent_id}] WON the claim. Doing the work.")
            await gc.quest_journal(quest_id, f"completed by {agent_id}")
            await gc.quest_fulfill(
                quest_id,
                report=f"commit demo-{agent_id}; files: src/atomic_claim_demo.py; "
                       f"agent {agent_id} won the race",
            )
        else:
            print(f"[{agent_id}] LOST the race. Will pick another quest.")


def main() -> None:
    if len(sys.argv) < 2:
        print(USAGE)
        sys.exit(2)
    mode = sys.argv[1]
    if mode == "seed":
        asyncio.run(seed())
    elif mode == "race" and len(sys.argv) >= 4:
        asyncio.run(race(quest_id=sys.argv[2], agent_id=sys.argv[3]))
    else:
        print(USAGE)
        sys.exit(2)


if __name__ == "__main__":
    main()
