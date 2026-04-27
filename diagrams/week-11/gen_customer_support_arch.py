"""Week 11 — Customer support agent reference architecture."""
from pathlib import Path
OUT_SVG = Path("/Users/yuxinliu/code/agent-prep/diagrams/week-11/customer-support-arch.svg")
BG, TEXT_PRI, TEXT_SEC, CTRL = "#ffffff", "#111827", "#6b7280", "#1f2937"
INTAKE_BG, INTAKE_S = "#eff6ff", "#bfdbfe"
TRIAGE_BG, TRIAGE_S = "#fff7ed", "#fed7aa"
SPEC_BG, SPEC_S = "#f0fdf4", "#bbf7d0"
OUT_BG, OUT_S = "#faf5ff", "#ddd6fe"
EVAL_BG, EVAL_S, EVAL_TXT = "#1b2631", "#1b2631", "#ffffff"
HUMAN_BG, HUMAN_S, HUMAN_TXT = "#fef2f2", "#fecaca", "#7f1d1d"
FONT = "'Helvetica Neue', Helvetica, Arial, sans-serif"
W, H = 1280, 1100

lines = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
         f'  <style>text {{ font-family: {FONT}; }}</style>',
         '  <defs>',
         f'    <marker id="a" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="{CTRL}"/></marker>',
         f'    <marker id="ah" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="{HUMAN_TXT}"/></marker>',
         '    <filter id="ds" x="-10%" y="-10%" width="120%" height="120%"><feDropShadow dx="0" dy="1" stdDeviation="1.5" flood-color="#000000" flood-opacity="0.07"/></filter>',
         '  </defs>',
         f'  <rect width="{W}" height="{H}" fill="{BG}"/>',
         f'  <text x="40" y="40" font-size="20" font-weight="700" fill="{TEXT_PRI}">Customer Support Agent — Reference Architecture</text>',
         f'  <text x="40" y="62" font-size="13" fill="{TEXT_SEC}">5-stage system: Intake → Triage (escalate to human if low confidence) → Specialist agents → Output validation → Eval / SLA monitoring.</text>']

def box(x, y, w, h, label, sub=None, fill="#fff", stroke="#cbd5e1", txt=TEXT_PRI):
    lines.append(f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" ry="8" fill="{fill}" stroke="{stroke}" stroke-width="1.5" filter="url(#ds)"/>')
    cx = x + w/2
    if sub:
        lines.append(f'  <text x="{cx}" y="{y + h/2 - 2}" text-anchor="middle" font-size="12" font-weight="600" fill="{txt}">{label}</text>')
        lines.append(f'  <text x="{cx}" y="{y + h/2 + 13}" text-anchor="middle" font-size="10" fill="{TEXT_SEC}">{sub}</text>')
    else:
        lines.append(f'  <text x="{cx}" y="{y + h/2 + 4}" text-anchor="middle" font-size="13" font-weight="600" fill="{txt}">{label}</text>')

def diamond(x, y, w, h, label, sub=None):
    cx, cy = x + w/2, y + h/2
    lines.append(f'  <polygon points="{cx},{y} {x+w},{cy} {cx},{y+h} {x},{cy}" fill="#fff" stroke="{CTRL}" stroke-width="1.5"/>')
    lines.append(f'  <text x="{cx}" y="{cy - 2}" text-anchor="middle" font-size="11" font-weight="600" fill="{TEXT_PRI}">{label}</text>')
    if sub:
        lines.append(f'  <text x="{cx}" y="{cy + 12}" text-anchor="middle" font-size="9" fill="{TEXT_SEC}">{sub}</text>')

def arr(x1, y1, x2, y2, label=None, marker="a"):
    color = HUMAN_TXT if marker == "ah" else CTRL
    lines.append(f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1.5" marker-end="url(#{marker})"/>')
    if label:
        lx, ly = (x1+x2)/2, (y1+y2)/2
        tw = len(label)*6 + 8
        lines.append(f'  <rect x="{lx - tw/2}" y="{ly - 8}" width="{tw}" height="14" fill="{BG}" opacity="0.95"/>')
        lines.append(f'  <text x="{lx}" y="{ly + 3}" text-anchor="middle" font-size="10" font-style="italic" fill="{TEXT_SEC}">{label}</text>')

# Intake (top)
IX, IY, IW, IH = 60, 100, 1160, 130
lines.append(f'  <rect x="{IX}" y="{IY}" width="{IW}" height="{IH}" rx="12" ry="12" fill="{INTAKE_BG}" stroke="{INTAKE_S}" stroke-width="2"/>')
lines.append(f'  <text x="{IX + 18}" y="{IY + 24}" font-size="13" font-weight="700" fill="#1e3a8a">Intake — multi-channel normalization</text>')
box(IX + 40, IY + 50, 140, 60, "Email", None, stroke=INTAKE_S)
box(IX + 200, IY + 50, 140, 60, "Chat", None, stroke=INTAKE_S)
box(IX + 360, IY + 50, 140, 60, "In-app event", None, stroke=INTAKE_S)
box(IX + 540, IY + 50, 240, 60, "Ticket normalizer", "body · channel · customer · tier · ts", stroke=INTAKE_S)
arr(180, IY + 80, IX + 540, IY + 80)
arr(340, IY + 80, IX + 540, IY + 80)
arr(500, IY + 80, IX + 540, IY + 80)
# Cylinder for state store
sx, sy = IX + 920, IY + 60
lines.append(f'  <ellipse cx="{sx + 70}" cy="{sy}" rx="70" ry="10" fill="#fff" stroke="{INTAKE_S}" stroke-width="1.5"/>')
lines.append(f'  <rect x="{sx}" y="{sy}" width="140" height="40" fill="#fff" stroke="{INTAKE_S}" stroke-width="1.5"/>')
lines.append(f'  <ellipse cx="{sx + 70}" cy="{sy + 40}" rx="70" ry="10" fill="#fff" stroke="{INTAKE_S}" stroke-width="1.5"/>')
lines.append(f'  <text x="{sx + 70}" y="{sy + 22}" text-anchor="middle" font-size="11" font-weight="600" fill="{TEXT_PRI}">Postgres / Redis</text>')
lines.append(f'  <text x="{sx + 70}" y="{sy + 38}" text-anchor="middle" font-size="9" fill="{TEXT_SEC}">+ SLA clock</text>')
arr(IX + 780, IY + 80, sx, IY + 80)

# Triage
TX, TY, TW, TH = 60, 260, 1160, 140
lines.append(f'  <rect x="{TX}" y="{TY}" width="{TW}" height="{TH}" rx="12" ry="12" fill="{TRIAGE_BG}" stroke="{TRIAGE_S}" stroke-width="2"/>')
lines.append(f'  <text x="{TX + 18}" y="{TY + 24}" font-size="13" font-weight="700" fill="#9a3412">Triage — classifier agent</text>')
box(TX + 40, TY + 50, 280, 70, "Intent classifier", "small model — billing/tech/account/escalate", stroke=TRIAGE_S)
diamond(TX + 380, TY + 45, 220, 80, "Confidence ≥ thresh?")
box(TX + 660, TY + 50, 240, 70, "Human queue", "SLA-aware priority", fill=HUMAN_BG, stroke=HUMAN_S, txt=HUMAN_TXT)
box(TX + 940, TY + 50, 200, 70, "Router", None, stroke=TRIAGE_S)
arr(IX + 990, IY + IH, TX + 180, TY + 50)  # state store → classifier
arr(TX + 320, TY + 85, TX + 380, TY + 85)
arr(TX + 600, TY + 85, TX + 660, TY + 85, "No / escalate", marker="ah")
arr(TX + 600, TY + 85, TX + 940, TY + 85, "Yes")

# Specialists
SX, SY, SW, SH = 60, 430, 1160, 130
lines.append(f'  <rect x="{SX}" y="{SY}" width="{SW}" height="{SH}" rx="12" ry="12" fill="{SPEC_BG}" stroke="{SPEC_S}" stroke-width="2"/>')
lines.append(f'  <text x="{SX + 18}" y="{SY + 24}" font-size="13" font-weight="700" fill="#14532d">Specialist agents</text>')
box(SX + 60, SY + 50, 320, 70, "Billing agent", "tools: payment API · invoices · subscription", stroke=SPEC_S)
box(SX + 420, SY + 50, 320, 70, "Tech agent", "tools: error logs · known-issues DB · docs", stroke=SPEC_S)
box(SX + 780, SY + 50, 320, 70, "Account agent", "tools: user profile · permissions · org mgmt", stroke=SPEC_S)
arr(TX + 1040, TY + TH, SX + 220, SY + 50, "")
arr(TX + 1040, TY + TH, SX + 580, SY + 50, "")
arr(TX + 1040, TY + TH, SX + 940, SY + 50, "")

# Output
OX, OY, OW, OH = 60, 590, 1160, 200
lines.append(f'  <rect x="{OX}" y="{OY}" width="{OW}" height="{OH}" rx="12" ry="12" fill="{OUT_BG}" stroke="{OUT_S}" stroke-width="2"/>')
lines.append(f'  <text x="{OX + 18}" y="{OY + 24}" font-size="13" font-weight="700" fill="#5b21b6">Output + feedback loop</text>')
box(OX + 40, OY + 50, 300, 70, "Output validator", "tone · PII scrub · policy check", stroke=OUT_S)
box(OX + 380, OY + 50, 240, 70, "Draft response", None, stroke=OUT_S)
diamond(OX + 660, OY + 45, 220, 80, "Human review?")
box(OX + 920, OY + 50, 200, 70, "Send to customer", None, fill="#dcfce7", stroke=SPEC_S)
arr(SX + 220, SY + SH, OX + 190, OY + 50)
arr(SX + 580, SY + SH, OX + 190, OY + 50)
arr(SX + 940, SY + SH, OX + 190, OY + 50)
arr(OX + 340, OY + 85, OX + 380, OY + 85)
arr(OX + 620, OY + 85, OX + 660, OY + 85)
arr(OX + 880, OY + 85, OX + 920, OY + 85, "No")
arr(OX + 770, OY + 125, TX + 780, TY + 120, "Yes", marker="ah")  # to human queue
box(OX + 380, OY + 140, 240, 50, "Edit log", "edit distance → training signal", fill="#1a5276", stroke="#1a5276", txt="#fff")
arr(TX + 780, TY + 120, OX + 500, OY + 140, "human edits", marker="ah")

# Eval
EX, EY, EW, EH = 60, 820, 1160, 200
lines.append(f'  <rect x="{EX}" y="{EY}" width="{EW}" height="{EH}" rx="12" ry="12" fill="{EVAL_BG}" stroke="{EVAL_S}" stroke-width="2"/>')
lines.append(f'  <text x="{EX + 18}" y="{EY + 24}" font-size="13" font-weight="700" fill="#a5b4fc">Eval + monitoring</text>')
box(EX + 40, EY + 60, 320, 70, "Edit-distance metric", "high = poor draft", fill="#374151", stroke="#6b7280", txt="#fff")
box(EX + 400, EY + 60, 320, 70, "CSAT score", "linked to ticket_id", fill="#374151", stroke="#6b7280", txt="#fff")
box(EX + 760, EY + 60, 320, 70, "SLA breach monitor", "→ escalation trigger", fill="#374151", stroke="#6b7280", txt="#fff")
arr(OX + 500, OY + 190, EX + 200, EY + 60)
arr(OX + 1020, OY + 120, EX + 560, EY + 60)
arr(IX + 990, IY + IH, EX + 920, EY + 60, "SLA")

lines.append('</svg>')
OUT_SVG.parent.mkdir(parents=True, exist_ok=True)
OUT_SVG.write_text("\n".join(lines))
print(f"OK: {OUT_SVG.stat().st_size} bytes")
