"""Integration test for the wired `_qdrant_supersede` soft-delete path.

Proves the bitemporal contract of Phase 9.6 supersede:
  1. after supersede, the OLD fact is excluded from live recall;
  2. the NEW fact carries a `supersedes` back-pointer;
  3. the OLD fact still EXISTS (soft-deleted), retrievable via
     include_superseded=True, and carries `superseded_by` → new id.

Requires live Qdrant (:6333) + an oMLX embed endpoint; skips cleanly otherwise.
Uses a throwaway user_id and deletes both points on teardown so the shared
collection isn't polluted.
"""
from __future__ import annotations

import os
import pathlib
import uuid

import httpx
import pytest
from dotenv import load_dotenv

load_dotenv(pathlib.Path(__file__).resolve().parent.parent / ".env")

from src.dedup_synthesis import DedupAction, _qdrant_delete, execute_action  # noqa: E402
from src.tiered_memory_qdrant import TieredMemory  # noqa: E402


def _services_up() -> bool:
    try:
        httpx.get("http://localhost:6333/collections", timeout=3).raise_for_status()
    except Exception:
        return False
    embed = os.getenv("EMBED_BASE_URL") or os.getenv("OMLX_BASE_URL")
    if not embed:
        return False
    try:
        httpx.get(embed.rstrip("/") + "/models", timeout=3)
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _services_up(), reason="needs live Qdrant :6333 + oMLX embed endpoint"
)


def test_supersede_soft_deletes_and_filters() -> None:
    tm = TieredMemory(agent_id="test-supersede", user_id=f"test-supersede-{uuid.uuid4().hex[:8]}")
    old_id = tm.imprint("user likes the React framework")
    new_id = None
    try:
        counts = execute_action(
            tm,
            DedupAction(
                action="supersede",
                target_id=old_id,
                supersede_reason="user switched frontend frameworks",
                supersede_category="preference",
            ),
            "user prefers the Vue framework now",
        )
        assert counts["superseded"] == 1 and counts["imprinted"] == 1

        # 1. live recall EXCLUDES the superseded old fact
        live = tm.query_context("frontend framework preference", k=10)
        live_ids = {h["id"] for h in live}
        assert old_id not in live_ids, "superseded fact leaked into live recall"

        # 2. the new fact is present with a `supersedes` back-pointer
        new_hits = [h for h in live if h.get("supersedes") == old_id]
        assert new_hits, "new fact missing or lacks supersedes pointer"
        new_id = new_hits[0]["id"]
        assert new_hits[0]["content"] == "user prefers the Vue framework now"

        # 3. old fact still EXISTS (soft-deleted), reachable for audit,
        #    and points forward to the new fact
        history = tm.query_context("frontend framework preference", k=10, include_superseded=True)
        old_hits = [h for h in history if h["id"] == old_id]
        assert old_hits, "soft-deleted fact was actually removed (hard delete?)"
        assert old_hits[0]["superseded_by"] == new_id
        assert old_hits[0]["content"] == "user likes the React framework"
    finally:
        _qdrant_delete(tm, [old_id] + ([new_id] if new_id else []))


def test_memory_supersede_helper_soft_deletes() -> None:
    """The §9 `memory_supersede` helper (separate path from execute_action) also
    soft-deletes now. Sync verification — the async @pytest.mark.asyncio
    integration test needs pytest-asyncio (not installed); this doesn't."""
    from src.memory_tools import memory_supersede

    uid = f"test-memsup-{uuid.uuid4().hex[:8]}"
    tm = TieredMemory(agent_id="test-memsup", user_id=uid)
    old_id = tm.imprint("user prefers the React framework")
    new_id = None
    try:
        new_id = memory_supersede(
            tm, old_id=old_id, new_content="user prefers the Svelte framework now",
            reason="preference shifted", user_id=uid,
        )
        live_ids = {h["id"] for h in tm.query_context("frontend framework preference", k=10)}
        assert old_id not in live_ids, "memory_supersede left the old fact in live recall"
        history = tm.query_context("frontend framework preference", k=10, include_superseded=True)
        old_hits = [h for h in history if h["id"] == old_id]
        assert old_hits and old_hits[0]["superseded_by"] == new_id
    finally:
        _qdrant_delete(tm, [old_id] + ([new_id] if new_id else []))
