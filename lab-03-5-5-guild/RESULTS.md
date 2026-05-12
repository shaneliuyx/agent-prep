# Week 3.5.5 — Multi-Agent Shared Memory: Results

**Date:** 2026-05-12
**Hardware:** MacBook Pro M5 Pro, 48 GB unified memory
**guild:** installed via `brew install mathomhaus/tap/guild`
**Python:** 3.11.15 (uv venv); `mcp` + `pytest-asyncio==1.3.0`

Only entries below this line have been executed live against `guild mcp serve` on 2026-05-12.

---

## 1.2 guild init

Verbatim transcript (from terminal session):

```
guild init — /Users/yuxinliu/code/agent-prep (project: agent-prep)

Will perform:
  [✓] register "agent-prep" in ~/.guild/ databases
  [?] AGENTS.md — not found → create
  [?] register guild MCP — detected: Claude Code, Cursor, Codex (OpenAI)

  ✓ registered "agent-prep" in lore + quest
  ✓ embedder enabled (cosine=1.0000, extract=96ms, probe=3ms)
  ✓ embedder backfill: 0 pending (up to date)
  ✓ created AGENTS.md

Added stdio MCP server guild with command: /opt/homebrew/bin/guild mcp serve to user config
File modified: /Users/yuxinliu/.claude.json
Added global MCP server 'guild'.
```

- Embedder cold-start: extract=96ms, probe=3ms.
- Project name auto-derived from git repo root: `agent-prep`.
- State location: `~/.guild/agent-prep/` (centralized; not in working dir).

---

## 1.4 MCP tool surface probe

Ad-hoc probe via `.venv/bin/python -c "...session.list_tools()..."` against `guild mcp serve`:

```
COUNT=43
```

The 43 tools (alphabetized; full list captured in commit `a209ffe` body):

- **3 top-level helpers:** `guild_session_start`, `guild_set_project`, `guild_status`
- **18 lore_* tools:** appraise, catalog, commune, coverage_reconcile, dossier, echoes, embed_rebuild, health, inquest, inscribe, link, list, meld, oath, reforge, ripples, seal, study, unlink, update, whispers (21 total in this namespace; alphabetized list contains 21)
- **18 quest_* tools:** accept, active, bounties, brief, campfire, clear, epic, forfeit, fulfill, guild, journal, list, orders, post, pulse, scroll, search, summon, update

Authoritative source: each tool's `inputSchema` accessible via `session.list_tools()[i].inputSchema`.

---

## 2.2 GuildClient smoke test

`pytest tests/test_guild_client.py -v` against live `guild mcp serve`:

```
============================= test session starts ==============================
collected 1 item

tests/test_guild_client.py::test_quest_lifecycle_round_trip PASSED       [100%]

============================== 1 passed in 0.85s ===============================
```

Round-trip verified: `quest_post → quest_accept → quest_journal → quest_fulfill → quest_scroll`. Journal text appears in the scroll output as expected.

**Result: 1/1 PASS, 0.85s.**

---

## 3.1 Atomic-claim two-process race demo

Verbatim transcript from `python -m src.atomic_claim_demo` seed+race split (2026-05-12 16:49):

```
=== seeded: QUEST-13 ===
[bob] attempting claim on QUEST-13...
[bob] result: [error] ❌ already accepted: QUEST-13 is held by agent (status=in_progress)
[bob] LOST the race. Will pick another quest.
[alice] attempting claim on QUEST-13...
[alice] result: ⚔️ accepted QUEST-13: race-the-prize: both agents want this
  status=in_progress · priority=P2 · campaign=race-demo · ow…
[alice] WON the claim. Doing the work.
```

- One WON, one LOST. Atomicity verified.
- Loser's response carries `[error]` prefix — guild's MCP server returns `isError: true` on the `CallToolResult` for race-loss.

---

## 3.2 Atomic-claim programmatic test

`pytest tests/test_atomic_claim.py -v`:

```
tests/test_atomic_claim.py::test_atomic_claim_exactly_one_winner PASSED  [100%]

============================== 1 passed in ~1s =================================
```

`asyncio.gather(try_claim('agent_a'), try_claim('agent_b'))` against the same pre-seeded `QUEST_ID`. Substring-match classifier (`'accept'|'claim'` without `'already'`) consistently identifies exactly one winner.

**Result: 1/1 PASS.**

---

## 4 Three-act cross-session handoff

`python -m src.three_act_handoff` — verbatim transcript (2026-05-12 17:05, second live run). Guild internal stderr JSON lines are interleaved with the demo's stdout per act; included here to show the per-act spawn/teardown of the `guild mcp serve` subprocess.

```
>>> Act 1 — agent A designs the API spec
{"time":"2026-05-12T17:05:30.519362+08:00","level":"INFO","msg":"server run start"}
{"time":"2026-05-12T17:05:30.51938+08:00","level":"INFO","msg":"server connecting"}
{"time":"2026-05-12T17:05:30.5194+08:00","level":"INFO","msg":"server session connected","session_id":""}
{"time":"2026-05-12T17:05:30.521084+08:00","level":"INFO","msg":"session initialized"}
  claim (QUEST-24): ⚔️ accepted QUEST-24: design-api-spec: design the new payments API
  status=in_progress · priority=P2 · campaign=payments-api-3act · owner=agent
  acceptance:
    - spec: LORE-10
  next useful lore call: lore_appraise(query="design-api-spec: design the new payments API", all_projects=True)
  journal logged + quest fulfilled (QUEST-24)
{"time":"2026-05-12T17:05:30.547622+08:00","level":"INFO","msg":"server session disconnected","session_id":""}
{"time":"2026-05-12T17:05:30.547649+08:00","level":"INFO","msg":"server session ended"}

>>> Act 2 — agent B implements based on agent A's design context
{"time":"2026-05-12T17:05:31.152737+08:00","level":"INFO","msg":"server run start"}
{"time":"2026-05-12T17:05:31.152758+08:00","level":"INFO","msg":"server connecting"}
{"time":"2026-05-12T17:05:31.152772+08:00","level":"INFO","msg":"server session connected","session_id":""}
{"time":"2026-05-12T17:05:31.153984+08:00","level":"INFO","msg":"session initialized"}
  read design scroll: 📜 QUEST-24 [P2 · done]  design-api-spec: design the new payments API
  owner: agent
  ✓ spec: LORE-1...
  claim (QUEST-25): ⚔️ accepted QUEST-25: implement-api: implement the payments API
  status=in_progress · priority=P2 · campaign=payments-api-3act · owner=agent
  acceptance:
    - spec: LORE-11
  next useful lore call: lore_appraise(query="implement-api: implement the payments API", all_projects=True)
{"time":"2026-05-12T17:05:31.182124+08:00","level":"INFO","msg":"server session disconnected","session_id":""}
{"time":"2026-05-12T17:05:31.182148+08:00","level":"INFO","msg":"server session ended"}

>>> Act 3 — agent C writes tests, sees the WHOLE chain
{"time":"2026-05-12T17:05:31.782221+08:00","level":"INFO","msg":"server run start"}
{"time":"2026-05-12T17:05:31.782244+08:00","level":"INFO","msg":"server connecting"}
{"time":"2026-05-12T17:05:31.782264+08:00","level":"INFO","msg":"server session connected","session_id":""}
{"time":"2026-05-12T17:05:31.783468+08:00","level":"INFO","msg":"session initialized"}
  read prior scroll [QUEST-24]: 📜 QUEST-24 [P2 · done]  design-api-spec: design the new payments API
  owner: ag...
  read prior scroll [QUEST-25]: 📜 QUEST-25 [P2 · done]  implement-api: implement the payments API
  owner: agent...
{"time":"2026-05-12T17:05:31.810146+08:00","level":"INFO","msg":"server session disconnected","session_id":""}
{"time":"2026-05-12T17:05:31.810164+08:00","level":"INFO","msg":"server session ended"}
```

**Measured observations:**

- **Three QUEST_IDs assigned** (QUEST-24 + QUEST-25 + Act-3's QUEST-26 — Act 3 doesn't print its own quest_id; visible only via `guild quest scroll`). Chain succeeded.
- **Per-act session lifecycle confirmed**: each act spawns a fresh `guild mcp serve` subprocess (`server run start`) and tears down on `__aexit__` (`server session ended`). Wall-clock per act: ~30 ms server-lifecycle + tool-call time. Three sessions total in the demo simulate three separate agent processes.
- **`owner=agent`** on all three claims — confirms session-scoped identity (BCJ Entry 5). Python wrapper's `agent_id="agent_a"`/`"agent_b"`/`"agent_c"` constructor args are application-layer labels only.
- **Auto-inscribed lore decision entries**: `spec: LORE-10` (from Act 1's `--spec=`), `spec: LORE-11` (from Act 2's `--spec=`). Act 3's `quest_post` did NOT include `--spec=` so no LORE entry was inscribed — confirms the `--spec` → atomic-lore-write semantic from §1.3.1.
- **New guild MCP response details observed (not previously documented in chapter)**:
  - `quest_accept` response now includes an `acceptance:` block listing the spec-pointer.
  - `quest_accept` response now includes a `next useful lore call: lore_appraise(...)` hint — guild's auto-generated agent-guidance breadcrumb suggesting the next reasoning move.
  - These are server-side LLM-affordance features; the Python wrapper passes them through as text without acting on them.
- **Cross-act scroll reads succeed**: Act 2 reads Act 1's scroll; Act 3 reads both. Each scroll read is one MCP `quest_scroll` call returning the full text history.

For a cleaner portfolio transcript (without the JSON server-lifecycle lines): `python -m src.three_act_handoff 2>/dev/null`.

### 4.1 Server-side verification of Act 3

Act 3's `quest_post` returned a QUEST_ID that the demo did not print to stdout (UX gap in the demo's `act_three_test()` function — claims & fulfills the quest but only prints the prior-scroll reads). Confirmed Act 3 cycle succeeded via direct `guild quest scroll`:

```
$ guild quest scroll QUEST-26
============================================================
  QUEST-26  ✅ [DONE]
============================================================
  Priority : P2
  Campaign : payments-api-3act
  Subject  : write-api-tests: exhaustive payments-API test suite
  Owner    : agent (since 2026-05-12T09:05)

  📝 NOTES
  ----------------------------------------
  [2026-05-12T09:05] agent: [spec] subject: write-api-tests: exhaustive payments-API test suite; priority: P2; epic: payments-api-3act
  [2026-05-12T09:05] agent: [checkpoint] accepted by agent — starting fresh
  [2026-05-12T09:05] agent: Wrote integration tests covering happy-path + idempotency-key replay + JWT-expiry + retry-on-503. Coverage 85% on the new code. Edge case TODO: clock skew on JWT validation.
  [2026-05-12T09:05] agent: [completed] commit test-001; files: tests/test_payments_integration.py; coverage 85%; remaining: clock-skew edge case
```

- Full post → accept → journal → fulfill cycle completed for Act 3.
- Wait — Act 3's `quest_post` was called WITHOUT `--spec=` (see chapter §4.1 act_three_test code). Scroll confirms: only `[spec]` subject-summary note appears, no `acceptance: spec: LORE-N` pointer note. Matches §1.3.1's documented behavior (no `--spec` → no auto-inscribed lore entry → no acceptance-spec pointer in journal).

### 4.2 Lore-store accumulation across lab runs

`guild lore list` after all 2026-05-12 lab runs:

```
📜 10 entry(ies):
  LORE-11 [decision · current]  implement-api: implement the payments API
  LORE-10 [decision · current]  design-api-spec: design the new payments API
  LORE-8 [decision · current]  design-api-spec: design the new payments API
  LORE-9 [decision · current]  implement-api: implement the payments API
  LORE-7 [decision · current]  smoke-test: round-trip the full quest lifecycle
  LORE-6 [decision · current]  smoke-test: round-trip the full quest lifecycle
  LORE-5 [decision · current]  smoke-test: round-trip the full quest lifecycle
  LORE-4 [decision · current]  smoke-test: round-trip the full quest lifecycle
  LORE-3 [decision · current]  deploy-prod-api: Roll out the new API
  LORE-1 [decision · current]  first-quest: smoke test the setup
```

Each `quest_post(spec=...)` call atomically inscribed one `kind=decision · current` lore entry. Tally by source + categorization (verified by `guild lore study` on each pair):

| LORE IDs | Source run | Category | Body equality |
|---|---|---|---|
| LORE-1 | §1.3.1 first-quest CLI probe | singleton | n/a |
| LORE-3 | §1.3 CLI sanity (`deploy-prod-api`) | singleton | n/a |
| LORE-4 to LORE-7 | 4× smoke-test reruns (test_guild_client.py) | **exact duplicates** | byte-identical: `"Verify Python wrapper covers post/accept/journal/fulfill/scroll."` × 4 |
| LORE-8, LORE-10 | 16:55 + 17:05 three-act Acts 1 | **near-duplicates** | bodies differ (see below) |
| LORE-9, LORE-11 | 16:55 + 17:05 three-act Acts 2 | **near-duplicates** | bodies differ (see below) |

**Exact-duplicate pair (LORE-4 vs LORE-7)** — four smoke-test pytest invocations inscribed byte-identical spec text:

```
$ guild lore study LORE-4
  SUMMARY: Verify Python wrapper covers post/accept/journal/fulfill/scroll.
$ guild lore study LORE-7
  SUMMARY: Verify Python wrapper covers post/accept/journal/fulfill/scroll.
```

**Near-duplicate pair — design (LORE-8 vs LORE-10)**:

```
$ guild lore study LORE-8
  SUMMARY: Define REST endpoints, auth model, retry semantics, idempotency strategy.
$ guild lore study LORE-10
  SUMMARY: Define REST endpoints, auth model, retry semantics, idempotency strategy. Output: API spec markdown.
```

LORE-10 carries one extra clause (`Output: API spec markdown.`).

**Near-duplicate pair — implement (LORE-9 vs LORE-11)**:

```
$ guild lore study LORE-9
  SUMMARY: Implement per design in QUEST-21. FastAPI + Pydantic.
$ guild lore study LORE-11
  SUMMARY: Implement per design spec captured in QUEST-24. Use FastAPI + Pydantic. Add tests stubs.
```

Three real differences: (a) different `quest_id` reference (21 vs 24), (b) wording shift ("per design in" → "per design spec captured in"), (c) extra clause "Add tests stubs" in LORE-11.

**Why the near-duplicate bodies are not byte-identical**: the chapter's `act_one_design()` and `act_two_implement()` spec strings were rewritten between the 16:55 run and the 17:05 run as part of the Phase 3-5 vocabulary sweep. Each rerun used the source code current at run time, so each LORE entry preserves the historical spec text from its own moment. Append-only by design — the older spec text is preserved in LORE-8/9 even though the source code no longer contains it.

**Dedup recommendation by category (per chapter §1.3.2)**:

| Category | Action | Rationale |
|---|---|---|
| Exact duplicates (LORE-4/5/6/7) | `guild lore reforge LORE-4 --with LORE-7` and similar | byte-identical bodies — chain them to keep only the newest current entry |
| Near-duplicates (LORE-8 ↔ 10, LORE-9 ↔ 11) | **debatable** | each pair represents the same decision *intent* recorded at a different *moment* with different *text*. Reforge if you want a single "current" decision per intent; preserve both if you value the historical text-evolution trail. Default for a portfolio repo: reforge (keep one canonical decision per topic). For an audit trail: preserve. |

Not executed today — dedup hygiene flagged as a follow-up.

---

## 5 Multi-agent recall — 5-test sample

`pytest tests/test_multi_agent_recall.py -v`:

```
tests/test_multi_agent_recall.py::test_01_same_agent_quest_appears_in_listing PASSED [ 20%]
tests/test_multi_agent_recall.py::test_02_same_agent_journal_round_trip PASSED      [ 40%]
tests/test_multi_agent_recall.py::test_06_agent_b_reads_agent_a_journal PASSED      [ 60%]
tests/test_multi_agent_recall.py::test_11_parallel_claim_exactly_one_winner PASSED  [ 80%]
tests/test_multi_agent_recall.py::test_15_quest_scroll_contains_fulfill_report PASSED [100%]

============================== 5 passed in 6.15s ===============================
```

| Category | Tests run today | Tests in chapter | Pass | Fail |
|---|---|---|---|---|
| Same-agent recall | test_01, test_02 | 5 total | 2/2 | 0 |
| Cross-agent handoff | test_06 | 5 total | 1/1 | 0 |
| Contradiction during parallel | test_11 | 5 total | 1/1 | 0 |
| Misc / fulfill-report visibility | test_15 | (not categorized) | 1/1 | 0 |

**5 of 15 chapter tests actually run today; 5/5 = 1.00 pass rate on the sample. Remaining 10 not yet executed.**

---

## Verified line-count comparison: W3.5 vs W3.5.5

`wc -l src/*.py` on both labs (2026-05-12):

| Lab | File(s) | LOC |
|---|---|---|
| W3.5.5 | `lab-03-5-5-guild/src/guild_client.py` | 188 |
| W3.5.5 | `lab-03-5-5-guild/src/atomic_claim_demo.py` | 83 |
| W3.5.5 | `lab-03-5-5-guild/src/three_act_handoff.py` | 90 |
| W3.5.5 | `lab-03-5-5-guild/src/smoke_test.py` | 21 |
| W3.5.5 | **total src/** | **382** |
| W3.5 | `lab-03-5-memory/src/chat.py` | 52 |
| W3.5 | `lab-03-5-memory/src/demo_three_sessions.py` | 65 |
| W3.5 | `lab-03-5-memory/src/init_db.py` | 28 |
| W3.5 | `lab-03-5-memory/src/lab_init.py` | 302 |
| W3.5 | `lab-03-5-memory/src/memory_mem0.py` | 132 |
| W3.5 | `lab-03-5-memory/src/memory.py` | 214 |
| W3.5 | **total src/** | **793** |

- W3.5.5 total `src/`: **382 LOC** (with concurrency safety + audit trail + multi-agent semantics).
- W3.5 total `src/`: **793 LOC** (single-agent only).
- Delta: W3.5.5 ships at ~48% the size of W3.5 with strictly more features.

---

## Bad-Case Journal entries from this session

**Entry 5 — Owner field shows `agent`, not the Python `agent_id` constructor arg.**
*Symptom:* `guild quest scroll QUEST-15` after a 2-process race demo shows `Owner: agent (since 2026-05-12T08:52)`. Expected `alice` or `bob` based on the Python wrapper's `agent_id="alice"` / `agent_id="bob"` constructor args.
*Root cause:* guild's MCP schema accepts no per-call agent identity (no `owner` on accept, no `agent` on journal, no `agent_id` on session_start). Identity is session-scoped — one MCP connection = one anonymous agent stream from the server's view. The Python `agent_id` is a logging-label only, never sent to the server. Confirmed via `session.list_tools()[i].inputSchema` probe of `quest_accept`, `quest_journal`, `guild_session_start`.
*Fix:* Wrap each agent's text payload with its identity at the application layer (the demo does this — `gc.quest_journal(quest_id, f"completed by {agent_id}")` — so the journal body carries the agent label even though the author field shows `agent`). For genuine per-agent server-side attribution, spawn separate `guild` subprocesses or modify the MCP client's `Implementation.name` per process.

**Entry 6 — Find-or-create-by-campaign race demo produces false WON-by-both.**
*Symptom:* Running `python -m src.atomic_claim_demo alice &` then `python -m src.atomic_claim_demo bob &` (sequential `&` invocations, ~8 seconds apart) reported BOTH agents WON, on different QUEST_IDs (alice created QUEST-10, bob created QUEST-11).
*Root cause:* Demo's find-or-create lookup used `quest_list(campaign="race-demo", status="next")`. After alice fulfilled QUEST-10, its status became `done`, no longer matching `status="next"`. Bob's lookup returned empty → bob created a new quest → both agents raced on different quests.
*Fix:* Split the demo into a two-phase `seed`+`race` CLI. Phase 1 `seed` creates the race quest ONCE and prints the QUEST_ID. Phase 2 `race QUEST_ID agent_id` spawned twice in parallel against the same QUEST_ID. Server's atomic-claim primitive then guarantees exactly one winner. Verified live — see §3.1 transcript above.

---

## Verification trail

| Artifact | Validation | Result |
|---|---|---|
| `guild init` output | terminal transcript | ✓ |
| MCP tool count (43) | `session.list_tools()` ad-hoc probe | ✓ |
| `tests/test_guild_client.py` | pytest 1/1 | ✓ |
| Two-process race transcript | live shell run, captured verbatim | ✓ |
| `tests/test_atomic_claim.py` | pytest 1/1 | ✓ |
| `python -m src.three_act_handoff` | live run, transcript captured | ✓ |
| `tests/test_multi_agent_recall.py` (5 of 15) | pytest 5/5 | ✓ |
| Line-count comparison (W3.5 / W3.5.5) | `wc -l src/*.py` on both lab dirs | ✓ |

**Aggregate: 8 distinct artifacts executed live on 2026-05-12; all green.**

---

## NOT executed today (chapter content not yet validated)

- §1.3 CLI 6-command sanity sequence end-to-end (partial commands ran during session; no clean single-run transcript captured)
- §1.4 `python -m src.smoke_test` (file was fixed mid-session; never re-run after the args correction)
- Remaining 10/15 tests in `test_multi_agent_recall.py` (only the 5-test sample was run)
- §5 16-Q lore-aware extension (not authored)
- Phase 6 "what I learned" 3 paragraphs (intentionally left to the lab author to write; not generated)
- Interview soundbites for §7 (intentionally left to the lab author to write; not generated)
