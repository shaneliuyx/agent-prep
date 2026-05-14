## guild workflow

guild coordinates tasks (quest) and persistent knowledge (lore) across sessions and agents.

**BEFORE ANY OTHER ACTION** — before reading files, editing code, or
responding to the user — call the MCP tool `guild_session_start(project="lab-03-5-8-two-tier")`.
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
