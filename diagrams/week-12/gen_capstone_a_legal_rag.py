"""Week 12 — Capstone A: Legal RAG architecture (simplified linear flow)."""
from pathlib import Path
OUT_SVG = Path("/Users/yuxinliu/code/agent-prep/diagrams/week-12/capstone-a-legal-rag.svg")
BG, TEXT_PRI, TEXT_SEC, CTRL = "#ffffff", "#111827", "#6b7280", "#1f2937"
USER_BG, USER_S = "#eff6ff", "#bfdbfe"
ROUTER_BG, ROUTER_S = "#fff7ed", "#fed7aa"
RETR_BG, RETR_S = "#f0fdf4", "#bbf7d0"
SYNTH_BG, SYNTH_S = "#faf5ff", "#ddd6fe"
OBS_BG, OBS_S, OBS_TXT = "#1a5276", "#1a5276", "#ffffff"
REFUSE_BG, REFUSE_S, REFUSE_TXT = "#fef2f2", "#fecaca", "#7f1d1d"
FONT = "'Helvetica Neue', Helvetica, Arial, sans-serif"
W, H = 1280, 880

lines = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
         f'  <style>text {{ font-family: {FONT}; }}</style>',
         '  <defs>',
         f'    <marker id="a" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="{CTRL}"/></marker>',
         f'    <marker id="ar" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="{REFUSE_TXT}"/></marker>',
         f'    <marker id="ag" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="#9ca3af"/></marker>',
         '    <filter id="ds" x="-10%" y="-10%" width="120%" height="120%"><feDropShadow dx="0" dy="1" stdDeviation="1.5" flood-color="#000000" flood-opacity="0.07"/></filter>',
         '  </defs>',
         f'  <rect width="{W}" height="{H}" fill="{BG}"/>',
         f'  <text x="40" y="40" font-size="20" font-weight="700" fill="{TEXT_PRI}">Capstone A — Legal RAG Architecture</text>',
         f'  <text x="40" y="62" font-size="13" fill="{TEXT_SEC}">Lawyer/analyst query → Router → ACL filter → Qdrant (per-tenant) → Reranker → Refusal gate or Synthesis with citations. Audit + RAGAS eval as side flows.</text>']

def box(x, y, w, h, label, sub=None, fill="#fff", stroke="#cbd5e1", txt=TEXT_PRI):
    lines.append(f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" ry="8" fill="{fill}" stroke="{stroke}" stroke-width="1.5" filter="url(#ds)"/>')
    cx = x + w/2
    if sub:
        lines.append(f'  <text x="{cx}" y="{y + h/2 - 2}" text-anchor="middle" font-size="13" font-weight="600" fill="{txt}">{label}</text>')
        for i, sline in enumerate(sub.split("\n")):
            lines.append(f'  <text x="{cx}" y="{y + h/2 + 14 + i*12}" text-anchor="middle" font-size="11" fill="{TEXT_SEC if txt == TEXT_PRI else txt}">{sline}</text>')
    else:
        lines.append(f'  <text x="{cx}" y="{y + h/2 + 5}" text-anchor="middle" font-size="13" font-weight="600" fill="{txt}">{label}</text>')

def cylinder(x, y, w, h, label, sub=None, fill="#fff", stroke="#cbd5e1", txt=TEXT_PRI):
    cx = x + w/2
    lines.append(f'  <ellipse cx="{cx}" cy="{y}" rx="{w/2}" ry="10" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    lines.append(f'  <rect x="{x}" y="{y}" width="{w}" height="{h - 20}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    lines.append(f'  <ellipse cx="{cx}" cy="{y + h - 20}" rx="{w/2}" ry="10" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    lines.append(f'  <text x="{cx}" y="{y + h/2 - 4}" text-anchor="middle" font-size="13" font-weight="600" fill="{txt}">{label}</text>')
    if sub:
        for i, sline in enumerate(sub.split("\n")):
            lines.append(f'  <text x="{cx}" y="{y + h/2 + 14 + i*12}" text-anchor="middle" font-size="10" fill="{TEXT_SEC}">{sline}</text>')

def arr(x1, y1, x2, y2, label=None, marker="a"):
    color = REFUSE_TXT if marker == "ar" else ("#9ca3af" if marker == "ag" else CTRL)
    dash = ' stroke-dasharray="5,3"' if marker == "ag" else ''
    lines.append(f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1.5"{dash} marker-end="url(#{marker})"/>')
    if label:
        lx, ly = (x1+x2)/2, (y1+y2)/2
        tw = len(label)*6 + 8
        lines.append(f'  <rect x="{lx - tw/2}" y="{ly - 8}" width="{tw}" height="14" fill="{BG}" opacity="0.95"/>')
        lines.append(f'  <text x="{lx}" y="{ly + 3}" text-anchor="middle" font-size="10" font-style="italic" fill="{TEXT_SEC}">{label}</text>')

# Main pipeline (top row, left to right)
y0 = 130
box(60,    y0, 180, 80, "👤 User", "lawyer / analyst", fill=USER_BG, stroke=USER_S)
box(280,   y0, 200, 80, "Query Router", "intent classifier", fill=ROUTER_BG, stroke=ROUTER_S)
box(520,   y0, 200, 80, "ACL Gate", "tenant_id filter\npre-retrieval", fill=ROUTER_BG, stroke=ROUTER_S)
cylinder(760, y0, 200, 100, "Qdrant", "per-tenant collections\nBGE-M3 embeddings", fill=RETR_BG, stroke=RETR_S)
box(1000,  y0, 220, 80, "BGE-reranker-v2-m3", "cross-encoder rerank", fill=RETR_BG, stroke=RETR_S)

arr(240,  y0 + 40, 280, y0 + 40)
arr(480,  y0 + 40, 520, y0 + 40)
arr(720,  y0 + 40, 760, y0 + 50, "filtered")
arr(960,  y0 + 50, 1000, y0 + 40, "top-50")

# Refusal gate (middle row, branching)
y1 = 290
box(490,   y1, 240, 100, "Refusal Layer", "out-of-scope detector", fill=REFUSE_BG, stroke=REFUSE_S, txt=REFUSE_TXT)
arr(1110,  y0 + 80, 730, y1 + 50, "top-5")

# Synthesis
y2 = 450
box(490,   y2, 280, 90, "Gemma-4-26B Synthesis", "+ citation\n(sonnet tier)", fill=SYNTH_BG, stroke=SYNTH_S)
arr(610, y1 + 100, 610, y2, "in-scope: pass")

# Output
y3 = 600
box(420,   y3, 360, 80, "Cited Answer + source spans", None, fill="#dcfce7", stroke="#86efac")
arr(630, y2 + 90, 600, y3)
arr(490, y1 + 50, 420, y3 + 30, "out-of-scope: block", marker="ar")

# Side: Audit + Phoenix + RAGAS (right column, dashed connections)
side_x = 880
box(side_x, y2 - 20, 280, 60, "Audit Log", "SQLite append-only · who/what/when", fill=OBS_BG, stroke=OBS_S, txt=OBS_TXT)
box(side_x, y2 + 60, 280, 60, "Phoenix Traces", "span per retrieval + synthesis", fill=OBS_BG, stroke=OBS_S, txt=OBS_TXT)
box(side_x, y2 + 140, 280, 80, "RAGAS Eval (offline)", "faithfulness · context-precision\nanswer-relevancy", fill="#374151", stroke="#6b7280", txt="#ffffff")
arr(770, y2 + 30, side_x, y2 + 10, "audit", marker="ag")
arr(770, y2 + 60, side_x, y2 + 90, "trace", marker="ag")
arr(side_x + 140, y2 + 220, 870, y0 + 90, "offline eval", marker="ag")

lines.append('</svg>')
OUT_SVG.parent.mkdir(parents=True, exist_ok=True)
OUT_SVG.write_text("\n".join(lines))
print(f"OK: {OUT_SVG.stat().st_size} bytes")
