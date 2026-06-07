## guild workflow

guild coordinates tasks (quest) and persistent knowledge (lore) across sessions and agents.

**BEFORE ANY OTHER ACTION** — before reading files, editing code, or
responding to the user — call the MCP tool `guild_session_start(project="agent-prep")`.
It returns the full agent contract, active principles (oath), and the
current top bounty. Follow what it returns.

If `guild_session_start` is not visible in your tool list, run your
host's tool-search for `guild` first — some hosts lazy-load MCP tools.
Do NOT fall back to CLI; the MCP server is available.

### Core rules (full contract is returned by session_start)

- **Never use built-in task tools** (TaskCreate / TaskUpdate / TaskList) —
  they're session-scoped. Use `quest_post` / `quest_accept` / `quest_list` instead.
- **Accept before working on a quest** — `quest_accept(quest_id=...)` prevents
  parallel-agent collisions.
- **Appraise before researching** — `lore_appraise(query=..., all_projects=true)`
  first. If current entries exist, use them.
- **Brief before session end** — when wrapping up or compaction is near,
  call `quest_brief("what was done, what's next, gotchas")` without being asked.

MCP namespace: `mcp__guild__*`. CLI fallback: `guild --help` (last resort only).

## shared library — reuse before you reinvent

**BEFORE writing new chapter / lab code, read `shared/` first and reuse what's there.**
`shared/` holds cross-chapter **infrastructure** extracted from earlier labs (LLM client +
model presets + retry + judge in `llm.py`; GBrain connect/read in `gbrain_cli.py` /
`gbrain_engine.ts`). See `shared/README.md` for the provenance map.

Procedure when creating a new lab:
1. **Read `shared/README.md` + the module surfaces.** Identify which utilities the new code
   needs (client/preset resolution, `resilient`, `judge`, `load_pass_criteria`, `bootstrapEngine`,
   `build_context`, …).
2. **Import, don't re-implement** the plumbing:
   - Python — `import sys; sys.path.insert(0, "/Users/yuxinliu/code/agent-prep/shared"); from llm import …`
   - TS/Bun — `import { bootstrapEngine } from "/Users/yuxinliu/code/agent-prep/shared/gbrain_engine.ts"`
3. **Keep teaching primitives in the lab.** The metric / routing / reader / policy logic a chapter
   *introduces* stays inline so the reader sees the mechanism — only plumbing goes through `shared/`.
   Rule: *introduce inline, reuse via import.* Never re-churn a finished chapter to adopt a refactor.
4. **Promote on rule-of-three.** If you find yourself writing infra a 2nd chapter also needs, add it
   to `shared/` (one module, documented in the README provenance table) — not speculatively.

This keeps lab code small and consistent without hiding each chapter's actual lesson.
