"""Week 4 — ReAct Loop visual centerpiece (agent_run()).

Source: Week 4 - ReAct From Scratch.md, mermaid block 1.
"""
from pathlib import Path

OUT_SVG = Path("/Users/yuxinliu/code/agent-prep/diagrams/week-4/react-loop.svg")
VAULT_PNG = Path("/Users/yuxinliu/Documents/Obsidian Vault/Agent Development Curriculum/assets/diagrams/week-4/react-loop.png")

BG = "#ffffff"
TEXT_PRI, TEXT_SEC = "#111827", "#6b7280"
DIVIDER = "#e5e7eb"
CTRL = "#1f2937"
OBJ  = "#9ca3af"

# Subsystem palettes
LOOP_BG, LOOP_S = "#eff6ff", "#bfdbfe"
TOOLS_BG, TOOLS_S = "#f0fdf4", "#bbf7d0"
OBS_BG, OBS_S = "#faf5ff", "#ddd6fe"
DECISION_S = "#1f4068"
TERMINAL_S = "#22c55e"
ERROR_BG = "#fef2f2"; ERROR_S = "#fecaca"; ERROR_TXT = "#7f1d1d"

FONT = "'Helvetica Neue', Helvetica, Arial, 'PingFang SC', 'Microsoft YaHei', sans-serif"

W, H = 1280, 1100

lines = []
lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">')
lines.append(f'  <style>text {{ font-family: {FONT}; }}</style>')
lines.append('  <defs>')
for mid, color in [("ctrl", CTRL), ("obj", OBJ), ("err", ERROR_TXT)]:
    lines.append(f'    <marker id="arr-{mid}" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">')
    lines.append(f'      <polygon points="0 0, 10 3.5, 0 7" fill="{color}"/>')
    lines.append('    </marker>')
lines.append('    <filter id="ds" x="-10%" y="-10%" width="120%" height="120%">')
lines.append('      <feDropShadow dx="0" dy="1" stdDeviation="1.5" flood-color="#000000" flood-opacity="0.06"/>')
lines.append('    </filter>')
lines.append('  </defs>')
lines.append(f'  <rect width="{W}" height="{H}" fill="{BG}"/>')

# Title
lines.append(f'  <text x="40" y="40" font-size="20" font-weight="700" fill="{TEXT_PRI}">Week 4 — ReAct Loop (agent_run)</text>')
lines.append(f'  <text x="40" y="62" font-size="12" fill="{TEXT_SEC}">User → assemble → call LLM → (budget + circular guard) → dispatch tool → record → context-evict → iter-check → loop. Side: tools layer + SQLite observability sidecar.</text>')

# Outer LOOP subgraph
LOOP_X, LOOP_Y, LOOP_W, LOOP_H = 80, 100, 760, 820
lines.append(f'  <rect x="{LOOP_X}" y="{LOOP_Y}" width="{LOOP_W}" height="{LOOP_H}" rx="14" ry="14" fill="{LOOP_BG}" stroke="{LOOP_S}" stroke-width="2"/>')
lines.append(f'  <text x="{LOOP_X + 18}" y="{LOOP_Y + 24}" font-size="13" font-weight="700" fill="{TEXT_PRI}">ReAct Loop — agent_run()</text>')

# Initial node + User → ASSEMBLE (entry)
INIT_X, INIT_Y = 460, 145
lines.append(f'  <circle cx="{INIT_X}" cy="{INIT_Y}" r="8" fill="{CTRL}"/>')
lines.append(f'  <text x="{INIT_X + 14}" y="{INIT_Y + 4}" font-size="11" font-style="italic" fill="{TEXT_SEC}">user message</text>')

def pill(cx, ytop, w, h, label, sub=None, fill="#ffffff", stroke=DIVIDER, txt_color=TEXT_PRI):
    x = cx - w/2
    rx = h/2
    lines.append(f'  <rect x="{x}" y="{ytop}" width="{w}" height="{h}" rx="{rx}" ry="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="1.5" filter="url(#ds)"/>')
    if sub:
        lines.append(f'  <text x="{cx}" y="{ytop + h/2 - 2}" text-anchor="middle" font-size="13" font-weight="600" fill="{txt_color}">{label}</text>')
        lines.append(f'  <text x="{cx}" y="{ytop + h/2 + 14}" text-anchor="middle" font-size="11" fill="{TEXT_SEC}">{sub}</text>')
    else:
        lines.append(f'  <text x="{cx}" y="{ytop + h/2 + 5}" text-anchor="middle" font-size="14" font-weight="600" fill="{txt_color}">{label}</text>')

def diamond(cx, ytop, w, h, label, sub=None):
    cy = ytop + h/2
    pts = f"{cx},{ytop} {cx + w/2},{cy} {cx},{ytop + h} {cx - w/2},{cy}"
    lines.append(f'  <polygon points="{pts}" fill="#ffffff" stroke="{DECISION_S}" stroke-width="1.5" filter="url(#ds)"/>')
    lines.append(f'  <text x="{cx}" y="{cy - 3}" text-anchor="middle" font-size="12" font-weight="600" fill="{TEXT_PRI}">{label}</text>')
    if sub:
        lines.append(f'  <text x="{cx}" y="{cy + 14}" text-anchor="middle" font-size="10" fill="{TEXT_SEC}">{sub}</text>')

def ctrl_arr(x1, y1, x2, y2, label=None, dashed=False):
    dash = ' stroke-dasharray="5,3"' if dashed else ''
    lines.append(f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{CTRL}" stroke-width="1.6"{dash} marker-end="url(#arr-ctrl)"/>')
    if label:
        lx, ly = (x1+x2)/2, (y1+y2)/2
        tw = len(label)*6.2 + 8
        lines.append(f'  <rect x="{lx - tw/2}" y="{ly - 9}" width="{tw}" height="14" fill="{BG}" opacity="0.95"/>')
        lines.append(f'  <text x="{lx}" y="{ly + 2}" text-anchor="middle" font-size="11" font-style="italic" fill="{TEXT_SEC}">{label}</text>')

def err_arr(x1, y1, x2, y2, label=None):
    lines.append(f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{ERROR_TXT}" stroke-width="1.6" marker-end="url(#arr-err)"/>')
    if label:
        lx, ly = (x1+x2)/2, (y1+y2)/2
        tw = len(label)*6.2 + 8
        lines.append(f'  <rect x="{lx - tw/2}" y="{ly - 9}" width="{tw}" height="14" fill="{BG}" opacity="0.95"/>')
        lines.append(f'  <text x="{lx}" y="{ly + 2}" text-anchor="middle" font-size="11" font-style="italic" fill="{ERROR_TXT}">{label}</text>')

# Loop nodes
ASM_X, ASM_Y = 460, 175
pill(ASM_X, ASM_Y, 280, 60, "context_for_llm()", "system + tools + user + scratchpad", fill="#ffffff", stroke=LOOP_S)

LLM_X, LLM_Y = 460, 270
pill(LLM_X, LLM_Y, 280, 60, "call_llm()", "oMLX :8000 · Qwen3.6-35B-A3B-nvfp4", fill="#dbeafe", stroke="#60a5fa")

PARSE_X, PARSE_Y = 460, 365
diamond(PARSE_X, PARSE_Y, 200, 60, "tool_calls?")

ANSWER_X, ANSWER_Y = 200, 372
pill(ANSWER_X, ANSWER_Y, 220, 50, "Return final answer", fill="#dcfce7", stroke=TERMINAL_S, txt_color="#14532d")

BUDGET_X, BUDGET_Y = 460, 460
diamond(BUDGET_X, BUDGET_Y, 240, 70, "budget OK?", "circular args?")

ERR_BUDGET_X, ERR_BUDGET_Y = 200, 470
pill(ERR_BUDGET_X, ERR_BUDGET_Y, 220, 50, "Return error to scratchpad", "circuit breaker", fill=ERROR_BG, stroke=ERROR_S, txt_color=ERROR_TXT)

DISPATCH_X, DISPATCH_Y = 460, 570
pill(DISPATCH_X, DISPATCH_Y, 280, 70, "run_tool()", "resolve name → call · catch all", fill="#ffffff", stroke=LOOP_S)

RESULT_X, RESULT_Y = 460, 680
pill(RESULT_X, RESULT_Y, 280, 60, "Append to Scratchpad", "(append-only event log)", fill="#ffffff", stroke=LOOP_S)

CTX_X, CTX_Y = 460, 770
diamond(CTX_X, CTX_Y, 240, 70, "tokens > LIMIT?")

EVICT_X, EVICT_Y = 200, 780
pill(EVICT_X, EVICT_Y, 220, 50, "Drop oldest entry", "tiered eviction (FIFO)", fill="#fef3c7", stroke="#facc15", txt_color="#854d0e")

ITER_X, ITER_Y = 460, 870
diamond(ITER_X, ITER_Y, 240, 70, "iter ≥ MAX_ITER?")

DLQ_X, DLQ_Y = 200, 880
pill(DLQ_X, DLQ_Y, 220, 50, "Return DLQ message", "AGENT STOPPED", fill=ERROR_BG, stroke=ERROR_S, txt_color=ERROR_TXT)

# TOOLS subgraph (right side)
TOOLS_X, TOOLS_Y, TOOLS_W, TOOLS_H = 880, 380, 360, 290
lines.append(f'  <rect x="{TOOLS_X}" y="{TOOLS_Y}" width="{TOOLS_W}" height="{TOOLS_H}" rx="14" ry="14" fill="{TOOLS_BG}" stroke="{TOOLS_S}" stroke-width="2"/>')
lines.append(f'  <text x="{TOOLS_X + 18}" y="{TOOLS_Y + 24}" font-size="13" font-weight="700" fill="#14532d">Tool Layer — src/tools.py</text>')

tools = [
    ("web_search",   "DDGS · max=4",       TOOLS_X + 90,  TOOLS_Y + 80),
    ("python_repl",  "subprocess · max=6", TOOLS_X + 270, TOOLS_Y + 80),
    ("read_file",    "path-contained · 8", TOOLS_X + 90,  TOOLS_Y + 200),
    ("write_file",   "path-contained · 4", TOOLS_X + 270, TOOLS_Y + 200),
]
for name, sub, cx, cy in tools:
    pill(cx, cy - 30, 160, 60, name, sub, fill="#ffffff", stroke=TOOLS_S)

# OBS subgraph (right-bottom)
OBS_X, OBS_Y, OBS_W, OBS_H = 880, 720, 360, 200
lines.append(f'  <rect x="{OBS_X}" y="{OBS_Y}" width="{OBS_W}" height="{OBS_H}" rx="14" ry="14" fill="{OBS_BG}" stroke="{OBS_S}" stroke-width="2"/>')
lines.append(f'  <text x="{OBS_X + 18}" y="{OBS_Y + 24}" font-size="13" font-weight="700" fill="#5b21b6">Observability Sidecar — src/obs.py</text>')

pill(OBS_X + 100, OBS_Y + 60, 200, 60, "log_event()", "row per iteration", fill="#ffffff", stroke=OBS_S)
# Cylinder for SQLite
sql_x, sql_y = OBS_X + 240, OBS_Y + 130
lines.append(f'  <ellipse cx="{sql_x}" cy="{sql_y}" rx="60" ry="10" fill="#ffffff" stroke=\"{OBS_S}\" stroke-width="1.5"/>')
lines.append(f'  <rect x="{sql_x - 60}" y="{sql_y}" width="120" height="40" fill="#ffffff" stroke="{OBS_S}" stroke-width="1.5"/>')
lines.append(f'  <ellipse cx="{sql_x}" cy="{sql_y + 40}" rx="60" ry="10" fill="#ffffff" stroke=\"{OBS_S}\" stroke-width="1.5"/>')
lines.append(f'  <text x="{sql_x}" y="{sql_y + 22}" text-anchor="middle" font-size="12" font-weight="600" fill="{TEXT_PRI}">SQLite</text>')
lines.append(f'  <text x="{sql_x}" y="{sql_y + 38}" text-anchor="middle" font-size="10" fill="{TEXT_SEC}">agent_events</text>')

# Arrows — main flow
ctrl_arr(INIT_X, INIT_Y + 8, ASM_X, ASM_Y - 2)
ctrl_arr(ASM_X, ASM_Y + 60, LLM_X, LLM_Y - 2)
ctrl_arr(LLM_X, LLM_Y + 60, PARSE_X, PARSE_Y - 2)
# Parse decision branches
ctrl_arr(PARSE_X - 100, PARSE_Y + 30, ANSWER_X + 110, ANSWER_Y + 25, "no")
ctrl_arr(PARSE_X, PARSE_Y + 60, BUDGET_X, BUDGET_Y - 2, "yes")
# Budget decision branches
err_arr(BUDGET_X - 120, BUDGET_Y + 35, ERR_BUDGET_X + 110, ERR_BUDGET_Y + 25, "exceeded")
ctrl_arr(BUDGET_X, BUDGET_Y + 70, DISPATCH_X, DISPATCH_Y - 2, "OK")
# Dispatch → result
ctrl_arr(DISPATCH_X, DISPATCH_Y + 70, RESULT_X, RESULT_Y - 2)
# Err budget joins back to result
err_arr(ERR_BUDGET_X + 110, ERR_BUDGET_Y + 50, RESULT_X - 140, RESULT_Y + 30)
# Result → context guard
ctrl_arr(RESULT_X, RESULT_Y + 60, CTX_X, CTX_Y - 2)
# Ctx guard branches
ctrl_arr(CTX_X - 120, CTX_Y + 35, EVICT_X + 110, EVICT_Y + 25, "yes")
ctrl_arr(CTX_X, CTX_Y + 70, ITER_X, ITER_Y - 2, "no")
# Evict joins back to iter check
ctrl_arr(EVICT_X + 110, EVICT_Y + 50, ITER_X - 130, ITER_Y + 35)
# Iter check
err_arr(ITER_X - 120, ITER_Y + 35, DLQ_X + 110, DLQ_Y + 25, "yes")
# Iter no → back to assemble (loop arrow on the right edge)
lines.append(f'  <path d="M {ITER_X + 120} {ITER_Y + 35} L 800 {ITER_Y + 35} L 800 {ASM_Y + 30} L {ASM_X + 140} {ASM_Y + 30}" stroke="{CTRL}" stroke-width="1.6" fill="none" marker-end="url(#arr-ctrl)"/>')
lines.append(f'  <text x="800" y="{ITER_Y + 30}" text-anchor="middle" font-size="11" font-style="italic" fill="{TEXT_SEC}">no — loop</text>')

# Dispatch → tools (curved into right subgraph)
lines.append(f'  <path d="M {DISPATCH_X + 140} {DISPATCH_Y + 35} C 760 {DISPATCH_Y + 35}, 800 {TOOLS_Y + 100}, {TOOLS_X} {TOOLS_Y + 100}" stroke="{CTRL}" stroke-width="1.6" stroke-dasharray="3,3" fill="none" marker-end="url(#arr-ctrl)"/>')

# Loop → log_event (dashed gray to obs)
lines.append(f'  <path d="M {LOOP_X + LOOP_W} {ITER_Y + 35} C 870 {ITER_Y + 35}, 870 {OBS_Y + 90}, {OBS_X + 100} {OBS_Y + 90}" stroke="{OBJ}" stroke-width="1.4" stroke-dasharray="5,3" fill="none" marker-end="url(#arr-obj)"/>')
lines.append(f'  <text x="845" y="{OBS_Y + 30}" font-size="10" font-style="italic" fill="{TEXT_SEC}" text-anchor="end">log every iter</text>')

# Final node from ANSWER
ANS_FIN_X, ANS_FIN_Y = ANSWER_X, 470
lines.append(f'  <circle cx="{ANS_FIN_X}" cy="{ANS_FIN_Y}" r="14" fill="#ffffff" stroke="{CTRL}" stroke-width="2"/>')
lines.append(f'  <circle cx="{ANS_FIN_X}" cy="{ANS_FIN_Y}" r="7" fill="{CTRL}"/>')
ctrl_arr(ANSWER_X, ANSWER_Y + 50, ANS_FIN_X, ANS_FIN_Y - 14)

# DLQ → final node (error path)
DLQ_FIN_X, DLQ_FIN_Y = DLQ_X, 970
lines.append(f'  <circle cx="{DLQ_FIN_X}" cy="{DLQ_FIN_Y}" r="14" fill="#ffffff" stroke="{ERROR_TXT}" stroke-width="2"/>')
lines.append(f'  <circle cx="{DLQ_FIN_X}" cy="{DLQ_FIN_Y}" r="7" fill="{ERROR_TXT}"/>')
err_arr(DLQ_X, DLQ_Y + 50, DLQ_FIN_X, DLQ_FIN_Y - 14)

# Legend
LX, LY = 40, H - 70
lines.append(f'  <text x="{LX}" y="{LY}" font-size="11" font-weight="600" fill="{TEXT_SEC}">UML NOTATION</text>')
lines.append(f'  <text x="{LX}" y="{LY + 18}" font-size="11" fill="{TEXT_PRI}">●  initial   ◇  decision (with [guard])   ◉  final   solid arrow = control   dashed = log/observability   red arrow = error path</text>')
lines.append(f'  <text x="{LX}" y="{LY + 36}" font-size="11" font-style="italic" fill="{TEXT_SEC}">Blue subgraph = ReAct Loop · Green = Tool Layer · Purple = Observability Sidecar</text>')

lines.append('</svg>')

OUT_SVG.parent.mkdir(parents=True, exist_ok=True)
OUT_SVG.write_text("\n".join(lines))
print(f"SVG written: {OUT_SVG} ({OUT_SVG.stat().st_size} bytes)")
