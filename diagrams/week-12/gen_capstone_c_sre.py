"""Week 12 — Capstone C: SRE LangGraph agent architecture."""
from pathlib import Path
OUT_SVG = Path("/Users/yuxinliu/code/agent-prep/diagrams/week-12/capstone-c-sre.svg")
BG, TEXT_PRI, TEXT_SEC, CTRL = "#ffffff", "#111827", "#6b7280", "#1f2937"
USER_BG, USER_S = "#eff6ff", "#bfdbfe"
LG_BG, LG_S = "#fef3c7", "#facc15"     # LangGraph (yellow)
TOOL_BG, TOOL_S = "#f0fdf4", "#bbf7d0"  # Tools (green)
QDR_BG, QDR_S = "#faf5ff", "#ddd6fe"    # Qdrant (purple)
OBS_BG, OBS_S = "#1a5276", "#1a5276"    # Phoenix (dark blue)
EVAL_BG, EVAL_S = "#1b2631", "#1b2631"  # Eval (dark slate)
OUT_BG, OUT_S = "#dcfce7", "#86efac"
FONT = "'Helvetica Neue', Helvetica, Arial, sans-serif"
W, H = 1280, 1100

lines = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
         f'  <style>text {{ font-family: {FONT}; }}</style>',
         '  <defs>',
         f'    <marker id="a" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="{CTRL}"/></marker>',
         f'    <marker id="ag" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="#9ca3af"/></marker>',
         '    <filter id="ds" x="-10%" y="-10%" width="120%" height="120%"><feDropShadow dx="0" dy="1" stdDeviation="1.5" flood-color="#000000" flood-opacity="0.07"/></filter>',
         '  </defs>',
         f'  <rect width="{W}" height="{H}" fill="{BG}"/>',
         f'  <text x="40" y="40" font-size="20" font-weight="700" fill="{TEXT_PRI}">Capstone C — SRE LangGraph Agent Architecture</text>',
         f'  <text x="40" y="62" font-size="13" fill="{TEXT_SEC}">User → LangGraph orchestrator → 5 deterministic tools → structured response with proposed action. Side: Phoenix observability + 40-question eval suite.</text>']

def box(x, y, w, h, label, sub=None, fill="#fff", stroke="#cbd5e1", txt=TEXT_PRI):
    lines.append(f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" ry="8" fill="{fill}" stroke="{stroke}" stroke-width="1.5" filter="url(#ds)"/>')
    cx = x + w/2
    if sub:
        lines.append(f'  <text x="{cx}" y="{y + h/2 - 2}" text-anchor="middle" font-size="12" font-weight="600" fill="{txt}">{label}</text>')
        for i, sline in enumerate(sub.split("\n")):
            lines.append(f'  <text x="{cx}" y="{y + h/2 + 13 + i*12}" text-anchor="middle" font-size="10" fill="{TEXT_SEC if txt == TEXT_PRI else "#cbd5e1"}">{sline}</text>')
    else:
        lines.append(f'  <text x="{cx}" y="{y + h/2 + 4}" text-anchor="middle" font-size="13" font-weight="600" fill="{txt}">{label}</text>')

def cylinder(x, y, w, h, label, sub=None, fill="#fff", stroke="#cbd5e1"):
    cx = x + w/2
    lines.append(f'  <ellipse cx="{cx}" cy="{y}" rx="{w/2}" ry="10" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    lines.append(f'  <rect x="{x}" y="{y}" width="{w}" height="{h - 20}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    lines.append(f'  <ellipse cx="{cx}" cy="{y + h - 20}" rx="{w/2}" ry="10" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    lines.append(f'  <text x="{cx}" y="{y + h/2 - 4}" text-anchor="middle" font-size="12" font-weight="600" fill="{TEXT_PRI}">{label}</text>')
    if sub:
        for i, sline in enumerate(sub.split("\n")):
            lines.append(f'  <text x="{cx}" y="{y + h/2 + 12 + i*11}" text-anchor="middle" font-size="9" fill="{TEXT_SEC}">{sline}</text>')

def diamond(x, y, w, h, label, sub=None):
    cx, cy = x + w/2, y + h/2
    lines.append(f'  <polygon points="{cx},{y} {x+w},{cy} {cx},{y+h} {x},{cy}" fill="#fff" stroke="{CTRL}" stroke-width="1.5"/>')
    lines.append(f'  <text x="{cx}" y="{cy - 2}" text-anchor="middle" font-size="11" font-weight="600" fill="{TEXT_PRI}">{label}</text>')
    if sub:
        lines.append(f'  <text x="{cx}" y="{cy + 12}" text-anchor="middle" font-size="9" fill="{TEXT_SEC}">{sub}</text>')

def arr(x1, y1, x2, y2, label=None, marker="a"):
    color = "#9ca3af" if marker == "ag" else CTRL
    dash = ' stroke-dasharray="5,3"' if marker == "ag" else ''
    lines.append(f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1.5"{dash} marker-end="url(#{marker})"/>')
    if label:
        lx, ly = (x1+x2)/2, (y1+y2)/2
        tw = len(label)*6 + 8
        lines.append(f'  <rect x="{lx - tw/2}" y="{ly - 8}" width="{tw}" height="14" fill="{BG}" opacity="0.95"/>')
        lines.append(f'  <text x="{lx}" y="{ly + 3}" text-anchor="middle" font-size="10" font-style="italic" fill="{TEXT_SEC}">{label}</text>')

# User (top)
box(60, 100, 380, 110, "👤 SRE / Engineer", "'Why is checkout-service p99 up?'\n'What does terraform apply change?'", fill=USER_BG, stroke=USER_S)

# LangGraph subgraph (top-center/right)
LG_X, LG_Y, LG_W, LG_H = 480, 100, 760, 200
lines.append(f'  <rect x="{LG_X}" y="{LG_Y}" width="{LG_W}" height="{LG_H}" rx="14" ry="14" fill="{LG_BG}" stroke="{LG_S}" stroke-width="2"/>')
lines.append(f'  <text x="{LG_X + 18}" y="{LG_Y + 24}" font-size="14" font-weight="700" fill="#854d0e">LangGraph Agent Loop</text>')
box(LG_X + 30, LG_Y + 50, 240, 130, "Qwen3.6-35B-A3B-nvfp4", "oMLX :8000\n(orchestrator)", stroke=LG_S)
box(LG_X + 290, LG_Y + 50, 220, 130, "Tool Router", "which tool next?", stroke=LG_S)
box(LG_X + 530, LG_Y + 50, 200, 130, "Iteration Guard", "max_iter=12\ncontext budget check", stroke=LG_S)
arr(440, 155, LG_X + 30, LG_Y + 115, "natural language")
arr(LG_X + 270, LG_Y + 115, LG_X + 290, LG_Y + 115)
arr(LG_X + 510, LG_Y + 115, LG_X + 530, LG_Y + 115)

# Tools subgraph (middle)
TX, TY, TW, TH = 60, 350, 1180, 240
lines.append(f'  <rect x="{TX}" y="{TY}" width="{TW}" height="{TH}" rx="14" ry="14" fill="{TOOL_BG}" stroke="{TOOL_S}" stroke-width="2"/>')
lines.append(f'  <text x="{TX + 18}" y="{TY + 24}" font-size="14" font-weight="700" fill="#14532d">Tool Layer — deterministic, no LLM</text>')
tools = [
    ("kubectl_tools",         "get_pods · describe_deploy\nget_events · logs",       TX + 40),
    ("promql_query",          "p99 latency · SLO burn rate",                         TX + 280),
    ("walk_distributed_trace", "find bottleneck span",                                TX + 520),
    ("parse_terraform_plan",  "resource diffs · IAM flags",                          TX + 760),
    ("semantic_runbook_search", "Qdrant retrieval · BGE-M3",                          TX + 1000),
]
for name, sub, x in tools:
    box(x, TY + 60, 220, 140, name, sub, stroke=TOOL_S)
# Connect Iteration Guard → all 5 tools
for _, _, x in tools:
    arr(LG_X + 630, LG_Y + LG_H, x + 110, TY + 60, "iter ok")

# Qdrant subgraph (under tools, right)
QX, QY, QW, QH = 920, 630, 320, 180
lines.append(f'  <rect x="{QX}" y="{QY}" width="{QW}" height="{QH}" rx="14" ry="14" fill="{QDR_BG}" stroke="{QDR_S}" stroke-width="2"/>')
lines.append(f'  <text x="{QX + 18}" y="{QY + 24}" font-size="13" font-weight="700" fill="#5b21b6">Qdrant Vector Store</text>')
cylinder(QX + 30, QY + 50, 120, 110, "Runbooks", "markdown in git\nBGE-M3 embedded", fill="#fff", stroke=QDR_S)
box(QX + 170, QY + 50, 140, 110, "BGE-M3", "embedding model\n(local MLX)", stroke=QDR_S)
arr(tools[4][2] + 110, TY + TH, QX + 90, QY + 50, "embed + search")

# Observability + Eval (bottom row)
OX, OY, OW, OH = 60, 630, 500, 180
lines.append(f'  <rect x="{OX}" y="{OY}" width="{OW}" height="{OH}" rx="14" ry="14" fill="{OBS_BG}" stroke="{OBS_S}" stroke-width="2"/>')
lines.append(f'  <text x="{OX + 18}" y="{OY + 24}" font-size="13" font-weight="700" fill="#a5b4fc">Observability</text>')
box(OX + 40, OY + 50, 420, 110, "Phoenix / Arize", "Trace per agent turn\nSpan per tool call · token counts + latency", fill="#374151", stroke="#6b7280", txt="#fff")
arr(LG_X + 130, LG_Y + LG_H, OX + 250, OY + 50, "trace", marker="ag")

# Eval (bottom-right wide)
EX, EY, EW, EH = 60, 850, 1180, 180
lines.append(f'  <rect x="{EX}" y="{EY}" width="{EW}" height="{EH}" rx="14" ry="14" fill="{EVAL_BG}" stroke="{EVAL_S}" stroke-width="2"/>')
lines.append(f'  <text x="{EX + 18}" y="{EY + 24}" font-size="13" font-weight="700" fill="#a5b4fc">Eval Layer</text>')
box(EX + 40, EY + 60, 320, 100, "40-question test set", "hand-labeled\nlatency / OOM / SLO / terraform", fill="#374151", stroke="#6b7280", txt="#fff")
box(EX + 400, EY + 60, 320, 100, "Deterministic eval", "verify agent claim\nvs live tool output", fill="#374151", stroke="#6b7280", txt="#fff")
box(EX + 760, EY + 60, 380, 100, "results/eval.md", "resolution rate · latency p50/p95\nhallucinated names = 0", fill="#374151", stroke="#6b7280", txt="#fff")
arr(EX + 360, EY + 110, EX + 400, EY + 110)
arr(EX + 720, EY + 110, EX + 760, EY + 110)

# Output (right side, between LangGraph and Eval)
box(640, 280, 320, 60, "Structured Response", "root cause · evidence · action (confirm req'd)", fill=OUT_BG, stroke=OUT_S)
arr(LG_X + 130, LG_Y + LG_H, 800, 280, "done")

lines.append('</svg>')
OUT_SVG.parent.mkdir(parents=True, exist_ok=True)
OUT_SVG.write_text("\n".join(lines))
print(f"OK: {OUT_SVG.stat().st_size} bytes")
