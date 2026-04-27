"""Week 6 — Claude Code 98.4% Deterministic Infrastructure architecture.

Source: Week 6 - Claude Code Source Dive.md, mermaid block 1.
"""
from pathlib import Path

OUT_SVG = Path("/Users/yuxinliu/code/agent-prep/diagrams/week-6/claude-code-arch.svg")
VAULT_PNG = Path("/Users/yuxinliu/Documents/Obsidian Vault/Agent Development Curriculum/assets/diagrams/week-6/claude-code-arch.png")

BG, TEXT_PRI, TEXT_SEC = "#ffffff", "#111827", "#6b7280"
INFRA_BG, INFRA_S = "#f0f9ff", "#7dd3fc"  # blue tint = deterministic
CORE_BG, CORE_S, CORE_TXT = "#1a1a2e", "#6060cc", "#e0e0ff"  # dark = AI loop core
FLAG_BG, FLAG_S, FLAG_TXT = "#fff7ed", "#fed7aa", "#9a3412"  # warm = feature flagged
SUB_BG, SUB_S = "#ffffff", "#cbd5e1"
CTRL = "#1f2937"
FONT = "'Helvetica Neue', Helvetica, Arial, 'PingFang SC', 'Microsoft YaHei', sans-serif"

W, H = 1280, 900
lines = []
lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">')
lines.append(f'  <style>text {{ font-family: {FONT}; }}</style>')
lines.append('  <defs>')
lines.append(f'    <marker id="arr" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="{CTRL}"/></marker>')
lines.append(f'    <marker id="arr-flag" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="{FLAG_TXT}"/></marker>')
lines.append('    <filter id="ds" x="-10%" y="-10%" width="120%" height="120%"><feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#000000" flood-opacity="0.1"/></filter>')
lines.append('  </defs>')
lines.append(f'  <rect width="{W}" height="{H}" fill="{BG}"/>')

# Title
lines.append(f'  <text x="40" y="40" font-size="22" font-weight="700" fill="{TEXT_PRI}">Claude Code — 98.4% Deterministic Infrastructure</text>')
lines.append(f'  <text x="40" y="62" font-size="13" fill="{TEXT_SEC}">7 deterministic subsystems wrap a single thin AI decision loop. Predictable, debuggable, auditable — the AI is the smallest part of the system.</text>')

# Infrastructure outer container
INFRA_X, INFRA_Y, INFRA_W, INFRA_H = 60, 100, 1160, 660
lines.append(f'  <rect x="{INFRA_X}" y="{INFRA_Y}" width="{INFRA_W}" height="{INFRA_H}" rx="20" ry="20" fill="{INFRA_BG}" stroke="{INFRA_S}" stroke-width="2.5" stroke-dasharray="8,4"/>')
lines.append(f'  <text x="{INFRA_X + 24}" y="{INFRA_Y + 32}" font-size="15" font-weight="700" fill="#0c4a6e">DETERMINISTIC INFRASTRUCTURE  ·  98.4% of system</text>')

def subsys(x, y, num, name, sub):
    w, h = 280, 100
    lines.append(f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" ry="10" fill="{SUB_BG}" stroke="{SUB_S}" stroke-width="1.5" filter="url(#ds)"/>')
    lines.append(f'  <circle cx="{x + 28}" cy="{y + 28}" r="16" fill="{INFRA_S}" stroke="{INFRA_S}" stroke-width="1.5"/>')
    lines.append(f'  <text x="{x + 28}" y="{y + 33}" text-anchor="middle" font-size="13" font-weight="700" fill="#0c4a6e">{num}</text>')
    lines.append(f'  <text x="{x + 56}" y="{y + 32}" font-size="13" font-weight="700" fill="{TEXT_PRI}">{name}</text>')
    # Sub-text wrapped to multiple lines
    for i, line in enumerate(sub.split("\n")):
        lines.append(f'  <text x="{x + 16}" y="{y + 56 + i*15}" font-size="11" fill="{TEXT_SEC}">{line}</text>')

# Top row — 3 upper subsystems
TOP_Y = 165
subsys(105,  TOP_Y, "②", "PERMISSION SYSTEM", "7 modes + ML classifier\ncode-layer gate")
subsys(495,  TOP_Y, "③", "COMPACTION PIPELINE", "5 layers: trim → dedup\n→ summarise → compress\n→ hard-truncate + sentinel")
subsys(885,  TOP_Y, "④", "TOOL ROUTER", "4 tracks: built-in │ MCP\nTask (subagent) │ skills/hooks")

# Center — AI Decision Logic core (the 1.6%)
CORE_X, CORE_Y, CORE_W, CORE_H = 290, 320, 700, 220
lines.append(f'  <rect x="{CORE_X}" y="{CORE_Y}" width="{CORE_W}" height="{CORE_H}" rx="14" ry="14" fill="{CORE_BG}" stroke="{CORE_S}" stroke-width="3" filter="url(#ds)"/>')
lines.append(f'  <text x="{CORE_X + CORE_W/2}" y="{CORE_Y + 28}" text-anchor="middle" font-size="13" font-weight="700" fill="#a5b4fc" letter-spacing="0.5">AI DECISION LOGIC  ·  ONLY 1.6% OF SYSTEM</text>')
lines.append(f'  <text x="{CORE_X + CORE_W/2}" y="{CORE_Y + 58}" text-anchor="middle" font-size="20" font-weight="700" fill="#ffffff">THE AGENT LOOP</text>')
core_code = [
    "while not done:",
    "    response = call_model(messages)",
    "    if tool_calls:",
    "        dispatch → router → permission_gate",
    "    if text:",
    "        emit; check_stop()",
    "    messages = maybe_compact(messages)",
]
for i, code in enumerate(core_code):
    lines.append(f'  <text x="{CORE_X + 60}" y="{CORE_Y + 95 + i*18}" font-family="ui-monospace, Menlo, monospace" font-size="13" fill="#e0e0ff">{code}</text>')

# Bottom row — 3 lower subsystems
BOT_Y = 580
subsys(105,  BOT_Y, "⑤", "SUBAGENT DISPATCH", "Task tool → isolated child\nown context + permission scope")
subsys(495,  BOT_Y, "⑥", "SESSION STORAGE", "append-only event log\nevent sourcing verbatim")
subsys(885,  BOT_Y, "⑦", "HOOK SYSTEM", "PreToolUse · PostToolUse · Stop\nOS-process isolation")

# Feature-flagged proactive (outside infra container, dashed)
FLAG_X, FLAG_Y, FLAG_W, FLAG_H = 460, 790, 360, 80
lines.append(f'  <rect x="{FLAG_X}" y="{FLAG_Y}" width="{FLAG_W}" height="{FLAG_H}" rx="10" ry="10" fill="{FLAG_BG}" stroke="{FLAG_S}" stroke-width="2" stroke-dasharray="6,3"/>')
lines.append(f'  <text x="{FLAG_X + 16}" y="{FLAG_Y + 22}" font-size="11" font-weight="700" fill="{FLAG_TXT}">FEATURE-FLAGGED · OFF BY DEFAULT</text>')
lines.append(f'  <circle cx="{FLAG_X + 28}" cy="{FLAG_Y + 50}" r="14" fill="{FLAG_S}" stroke="{FLAG_S}"/>')
lines.append(f'  <text x="{FLAG_X + 28}" y="{FLAG_Y + 54}" text-anchor="middle" font-size="12" font-weight="700" fill="{FLAG_TXT}">⑧</text>')
lines.append(f'  <text x="{FLAG_X + 50}" y="{FLAG_Y + 50}" font-size="13" font-weight="700" fill="{TEXT_PRI}">PROACTIVE / KAIROS</text>')
lines.append(f'  <text x="{FLAG_X + 50}" y="{FLAG_Y + 66}" font-size="11" fill="{TEXT_SEC}">background monitoring · autonomous trigger + scheduler</text>')

# Arrows from subsystems → core (gate every tool call, compact, dispatch)
def arr(x1, y1, x2, y2, label=None, dashed=False, marker="arr"):
    dash = ' stroke-dasharray="5,3"' if dashed else ''
    color = FLAG_TXT if marker == "arr-flag" else CTRL
    lines.append(f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1.6"{dash} marker-end="url(#{marker})"/>')
    if label:
        lx, ly = (x1+x2)/2, (y1+y2)/2
        tw = len(label)*6.2 + 8
        lines.append(f'  <rect x="{lx - tw/2}" y="{ly - 9}" width="{tw}" height="14" fill="{INFRA_BG}" opacity="0.95"/>')
        lines.append(f'  <text x="{lx}" y="{ly + 2}" text-anchor="middle" font-size="10" font-style="italic" fill="{TEXT_SEC}">{label}</text>')

# Top → Core (3 arrows down)
arr(245, TOP_Y + 100, CORE_X + 80, CORE_Y - 2, "gate")
arr(635, TOP_Y + 100, CORE_X + CORE_W/2, CORE_Y - 2, "compact")
arr(1025, TOP_Y + 100, CORE_X + CORE_W - 80, CORE_Y - 2, "dispatch")

# Core → Bottom (3 arrows down)
arr(CORE_X + 80, CORE_Y + CORE_H, 245, BOT_Y - 2, "spawn Task")
arr(CORE_X + CORE_W/2, CORE_Y + CORE_H, 635, BOT_Y - 2, "append event")
arr(CORE_X + CORE_W - 80, CORE_Y + CORE_H, 1025, BOT_Y - 2, "fire on lifecycle")

# Kairos → Core (dashed, flagged off)
arr(FLAG_X + FLAG_W/2, FLAG_Y, CORE_X + CORE_W/2, CORE_Y + CORE_H + 4, "initiates session (off)", dashed=True, marker="arr-flag")

lines.append('</svg>')
OUT_SVG.parent.mkdir(parents=True, exist_ok=True)
OUT_SVG.write_text("\n".join(lines))
print(f"OK: {OUT_SVG.stat().st_size} bytes")
