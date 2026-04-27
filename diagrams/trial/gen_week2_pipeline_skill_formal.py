"""Week 2 retrieval pipeline — UML Activity Diagram.
Generated under formal /fireworks-tech-graph skill invocation.

Skill rules followed strictly:
- Style 1 (Flat Icon) tokens from references/style-1-flat-icon.md
- UML Activity notation: ● init, pill activities, ◇ decision, ▽ merge, ◉ final
- 120px vertical between layers, 80px+ horizontal between same-layer nodes
- Edge-anchored arrows (not center-to-center)
- Arrow labels with background rect (4px h / 2px v padding) per "Arrow Labels (CRITICAL)"
- «datastore» stereotype on Measurement Store
- Python list method (mandatory per skill's "SVG Generation & Error Prevention")
- Validation via rsvg-convert before PNG export
"""
from pathlib import Path

OUT_SVG = Path("/Users/yuxinliu/code/agent-prep/diagrams/trial/week2-pipeline-skill-formal.svg")
OUT_PNG = OUT_SVG.with_suffix(".png")

# ---------- Style 1 tokens (from references/style-1-flat-icon.md) ----------
BG          = "#ffffff"
BOX_FILL    = "#ffffff"
BOX_STROKE  = "#d1d5db"
TEXT_PRI    = "#111827"   # gray-900
TEXT_SEC    = "#6b7280"   # gray-500
DIVIDER     = "#e5e7eb"

# UML control flow: dark; object flow: gray-dashed (per skill's "Async/event" semantic)
CTRL = "#1f2937"
OBJ  = "#6b7280"   # gray-500 per skill async semantic

# Accent tints for activities (subtle, per Style 1 "icon accent backgrounds")
TINT_BLUE,  TINT_BLUE_S = "#eff6ff", "#bfdbfe"
TINT_ORG,   TINT_ORG_S  = "#fff7ed", "#fed7aa"   # ORANGE for Hybrid (Phase 1 new)
TINT_GRN,   TINT_GRN_S  = "#f0fdf4", "#bbf7d0"   # GREEN for rerank
TINT_PRP,   TINT_PRP_S  = "#faf5ff", "#ddd6fe"   # PURPLE for compression
TINT_GRAY               = "#f3f4f6"

ORANGE, GREEN, PURPLE = "#ea580c", "#16a34a", "#7c3aed"

FONT = "'Helvetica Neue', Helvetica, Arial, 'PingFang SC', 'Microsoft YaHei', sans-serif"

# ---------- Layout: 120px vertical grid (skill mandate) ----------
W, H = 1280, 1080
CX = 540   # main flow centerline (left of canvas center; right side for datastore)

# 120px vertical spacing per skill rule "120px vertical between layers"
LAYER_PITCH = 120

# Y-positions on a 120px grid where reasonable (decision→activity pairs need tighter pitch)
Y_TITLE   = 30
Y_SUB     = 52
Y_INIT    = 95     # initial circle center
Y_ENC     = 130    # BGE-M3 Encode top  (activity height 50; gap = 35)
Y_DEC1    = 230    # Decision: Retrieval Mode? (top)
Y_RETR    = 340    # Three retrieval activities (top)
Y_MERGE1  = 470    # Merge after retrieval (center)
Y_DEC2    = 510    # Decision: Rerank? (top)
Y_RR      = 620    # Two rerank activities (top)
Y_MERGE2  = 730    # Merge after rerank (center)
Y_DEC3    = 770    # Decision: Compress? (top)
Y_COMP    = 880    # Two compression activities (top)
Y_MERGE3  = 980    # Merge after compress (center)
Y_SYN     = 1010   # Synthesis (top)
Y_FINAL   = 1085   # Final node center (slightly past viewBox = will increase H)

# Bumping H to fit final node + legend
H = 1180

# X-positions for fan-out (snap to ~120px grid where possible)
X_LEFT    = 140    # left of 3-way retrieval fan
X_CENTER  = CX     # centerline (also center of 3-way fan)
X_RIGHT   = 940    # right of 3-way retrieval fan
X_PAIR_L  = 290    # left of 2-way pair (rerank/compress skip side)
X_PAIR_R  = 790    # right of 2-way pair (rerank/compress default side)

# Activity sizes
AW_STD, AH_STD = 220, 70    # standard activity (3-way fan)
AW_PILL, AH_PILL = 200, 50  # narrower pill (Encode, Synthesis on centerline)
DEC_W, DEC_H = 130, 65      # decision diamond
MRG_W, MRG_H = 30, 24       # merge diamond (small, unlabeled)

# Measurement panel (right side, well clear of fan-out columns)
MX, MY, MW, MH = 1040, 410, 220, 380


# ---------- SVG construction (Python list method per skill) ----------
lines = []
lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">')
lines.append(f'  <style>text {{ font-family: {FONT}; }}</style>')

# Markers (skill: "All arrows: <marker> with markerEnd, sized markerWidth=10 markerHeight=7")
lines.append('  <defs>')
for mid, color in [("ctrl", CTRL), ("obj", OBJ)]:
    lines.append(f'    <marker id="arr-{mid}" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">')
    lines.append(f'      <polygon points="0 0, 10 3.5, 0 7" fill="{color}"/>')
    lines.append('    </marker>')
# Drop shadow filter (skill: "apply sparingly (key nodes only)")
lines.append('    <filter id="ds" x="-10%" y="-10%" width="120%" height="120%">')
lines.append('      <feDropShadow dx="0" dy="1" stdDeviation="1.5" flood-color="#000000" flood-opacity="0.06"/>')
lines.append('    </filter>')
lines.append('  </defs>')

# Background
lines.append(f'  <rect width="{W}" height="{H}" fill="{BG}"/>')

# Title (skill: titles 16-18px, semi-bold)
lines.append(f'  <text x="40" y="{Y_TITLE}" font-size="18" font-weight="600" fill="{TEXT_PRI}">Week 2 — Retrieval Pipeline (UML Activity Diagram)</text>')
lines.append(f'  <text x="40" y="{Y_SUB}" font-size="12" fill="{TEXT_SEC}">3 retrieval modes × 2 rerank options × 2 compression options = 12 measurable A/B combinations</text>')


# ---------- Helpers ----------

def activity(cx, ytop, w, h, label, sub=None, fill=BOX_FILL, stroke=BOX_STROKE):
    """UML activity — pill rect (fully rounded, rx = h/2)."""
    x = cx - w / 2
    rx = h / 2
    lines.append(f'  <rect x="{x}" y="{ytop}" width="{w}" height="{h}" rx="{rx}" ry="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="1.5" filter="url(#ds)"/>')
    if sub:
        lines.append(f'  <text x="{cx}" y="{ytop + h/2 - 2}" text-anchor="middle" font-size="13" font-weight="600" fill="{TEXT_PRI}">{label}</text>')
        lines.append(f'  <text x="{cx}" y="{ytop + h/2 + 14}" text-anchor="middle" font-size="11" fill="{TEXT_SEC}">{sub}</text>')
    else:
        lines.append(f'  <text x="{cx}" y="{ytop + h/2 + 5}" text-anchor="middle" font-size="14" font-weight="600" fill="{TEXT_PRI}">{label}</text>')


def decision(cx, ytop, w, h, question):
    """UML decision — diamond, question text BELOW (so the diamond stays symbolic)."""
    cy = ytop + h / 2
    pts = f"{cx},{ytop} {cx + w/2},{cy} {cx},{ytop + h} {cx - w/2},{cy}"
    lines.append(f'  <polygon points="{pts}" fill="{BOX_FILL}" stroke="{CTRL}" stroke-width="1.5"/>')
    lines.append(f'  <text x="{cx}" y="{ytop + h + 14}" text-anchor="middle" font-size="12" font-weight="600" fill="{TEXT_PRI}">{question}</text>')


def merge(cx, cy):
    """Unlabeled merge diamond (small)."""
    pts = f"{cx},{cy - MRG_H/2} {cx + MRG_W/2},{cy} {cx},{cy + MRG_H/2} {cx - MRG_W/2},{cy}"
    lines.append(f'  <polygon points="{pts}" fill="{BOX_FILL}" stroke="{CTRL}" stroke-width="1.5"/>')


def initial_node(cx, cy):
    lines.append(f'  <circle cx="{cx}" cy="{cy}" r="8" fill="{CTRL}"/>')


def final_node(cx, cy):
    lines.append(f'  <circle cx="{cx}" cy="{cy}" r="14" fill="{BOX_FILL}" stroke="{CTRL}" stroke-width="2"/>')
    lines.append(f'  <circle cx="{cx}" cy="{cy}" r="7" fill="{CTRL}"/>')


def label_with_bg(cx, cy, text, font_size=11, italic=True, bg=BG):
    """Label with background rect for readability (skill MANDATE: arrow labels MUST have bg rect)."""
    text_w = len(text) * 6.2 + 8
    lines.append(f'  <rect x="{cx - text_w/2}" y="{cy - 9}" width="{text_w}" height="14" fill="{bg}" opacity="0.95"/>')
    style = ' font-style="italic"' if italic else ''
    lines.append(f'  <text x="{cx}" y="{cy + 2}" text-anchor="middle" font-size="{font_size}"{style} fill="{TEXT_PRI}">{text}</text>')


def ctrl_arrow(x1, y1, x2, y2):
    """Simple straight control flow arrow."""
    lines.append(f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{CTRL}" stroke-width="1.6" marker-end="url(#arr-ctrl)"/>')


def ctrl_orth_v_then_h(x1, y1, x2, y2, guard=None, junction_offset=30):
    """Orthogonal: vertical from (x1,y1), horizontal junction, vertical to (x2,y2). Used for decision fan-out."""
    junction_y = y1 + junction_offset
    path = f"M {x1} {y1} L {x1} {junction_y} L {x2} {junction_y} L {x2} {y2}"
    lines.append(f'  <path d="{path}" stroke="{CTRL}" stroke-width="1.6" fill="none" marker-end="url(#arr-ctrl)"/>')
    if guard:
        # Guard label centered on the horizontal segment, with bg rect (skill mandate)
        gx = (x1 + x2) / 2
        gy = junction_y - 8
        label_with_bg(gx, gy, guard)


def ctrl_orth_h_then_v(x1, y1, x2, y2):
    """Orthogonal converge: down from source, horizontal toward target, down to target. For merge convergence."""
    junction_y = y1 + (y2 - y1) * 0.55
    path = f"M {x1} {y1} L {x1} {junction_y} L {x2} {junction_y} L {x2} {y2}"
    lines.append(f'  <path d="{path}" stroke="{CTRL}" stroke-width="1.6" fill="none" marker-end="url(#arr-ctrl)"/>')


def obj_flow(x1, y1, x2, y2, obj_label):
    """Dashed object flow arrow with [object_label] mid-arrow. Curved for visual separation from control flow."""
    cp1x = x1 + (x2 - x1) * 0.45
    cp1y = y1
    cp2x = x2 - 40
    cp2y = y2
    path = f"M {x1} {y1} C {cp1x} {cp1y}, {cp2x} {cp2y}, {x2} {y2}"
    lines.append(f'  <path d="{path}" stroke="{OBJ}" stroke-width="1.4" fill="none" stroke-dasharray="5,3" marker-end="url(#arr-obj)"/>')
    # Label in [brackets] mid-arrow (UML object flow convention) with bg rect
    lx = (x1 + x2) / 2 + 25
    ly = (y1 + y2) / 2 - 8
    label_with_bg(lx, ly, obj_label, italic=False)


# ---------- Build the diagram ----------

# Initial node
initial_node(CX, Y_INIT)

# BGE-M3 Encode (blue tint — primary path)
activity(CX, Y_ENC, AW_PILL, AH_PILL, "BGE-M3 Encode",
         fill=TINT_BLUE, stroke=TINT_BLUE_S)
# Sub-label below the activity (since the pill itself is single-line)
lines.append(f'  <text x="{CX}" y="{Y_ENC + AH_PILL + 14}" text-anchor="middle" font-size="11" fill="{TEXT_SEC}">one forward pass → dense + sparse outputs</text>')

# Decision 1: Retrieval Mode?
decision(CX, Y_DEC1, DEC_W, DEC_H, "Retrieval Mode?")

# Three retrieval activities
activity(X_LEFT,    Y_RETR, AW_STD, AH_STD, "Dense-only",
         sub="bge_m3_hnsw · using=dense", fill=TINT_GRAY)
activity(X_CENTER,  Y_RETR, AW_STD, AH_STD, "Hybrid (RRF k=60)",
         sub="dense + sparse → fuse", fill=TINT_ORG, stroke=TINT_ORG_S)
activity(X_RIGHT,   Y_RETR, AW_STD, AH_STD, "Sparse-only",
         sub="bge_m3_hybrid · using=sparse", fill=TINT_GRAY)
# Phase 1 highlight badge below the hybrid activity
lines.append(f'  <text x="{X_CENTER}" y="{Y_RETR + AH_STD + 14}" text-anchor="middle" font-size="10" font-weight="600" fill="{ORANGE}">PHASE 1 NEW</text>')

# Merge after retrieval
merge(CX, Y_MERGE1)

# Decision 2: Rerank?
decision(CX, Y_DEC2, DEC_W, DEC_H, "Rerank?")

# Two rerank activities
activity(X_PAIR_L, Y_RR, AW_STD, AH_STD, "Skip rerank",
         sub="top-5 by retrieval score", fill=TINT_GRAY)
activity(X_PAIR_R, Y_RR, AW_STD, AH_STD, "BGE-reranker-v2-m3",
         sub="cross-encoder · ~80-140ms", fill=TINT_GRN, stroke=TINT_GRN_S)

# Merge after rerank
merge(CX, Y_MERGE2)

# Decision 3: Compress?
decision(CX, Y_DEC3, DEC_W, DEC_H, "Compress?")

# Two compression activities
activity(X_PAIR_L, Y_COMP, AW_STD, AH_STD, "Raw context",
         sub="5 full passages", fill=TINT_GRAY)
activity(X_PAIR_R, Y_COMP, AW_STD, AH_STD, "Gemma 26B Compressor",
         sub="extract-only · temp=0", fill=TINT_PRP, stroke=TINT_PRP_S)

# Merge after compression
merge(CX, Y_MERGE3)

# Synthesis
activity(CX, Y_SYN, AW_PILL, AH_PILL, "Synthesis Model",
         fill=TINT_BLUE, stroke=TINT_BLUE_S)

# Final node
final_node(CX, Y_FINAL)


# ---------- Control flow arrows (UML solid black, edge-anchored per skill) ----------

# Initial → Encode
ctrl_arrow(CX, Y_INIT + 8, CX, Y_ENC - 2)

# Encode → Decision 1
ctrl_arrow(CX, Y_ENC + AH_PILL + 18, CX, Y_DEC1 - 2)

# Decision 1 → 3 retrieval activities (orthogonal fan-out with guard labels)
ctrl_orth_v_then_h(CX - DEC_W/2, Y_DEC1 + DEC_H/2, X_LEFT,   Y_RETR - 2, guard="[A/B: dense]")
ctrl_arrow(CX, Y_DEC1 + DEC_H + 22, CX, Y_RETR - 2)
# Center guard label (placed mid-segment with bg)
label_with_bg(CX + 80, Y_DEC1 + DEC_H + 14, "[default: hybrid]")
ctrl_orth_v_then_h(CX + DEC_W/2, Y_DEC1 + DEC_H/2, X_RIGHT,  Y_RETR - 2, guard="[A/B: sparse]")

# 3 retrieval activities → merge node
RETR_BOTTOM = Y_RETR + AH_STD
ctrl_orth_h_then_v(X_LEFT,   RETR_BOTTOM, CX, Y_MERGE1 - MRG_H/2 - 2)
ctrl_arrow(CX, RETR_BOTTOM, CX, Y_MERGE1 - MRG_H/2 - 2)
ctrl_orth_h_then_v(X_RIGHT,  RETR_BOTTOM, CX, Y_MERGE1 - MRG_H/2 - 2)

# Merge → Decision 2
ctrl_arrow(CX, Y_MERGE1 + MRG_H/2, CX, Y_DEC2 - 2)

# Decision 2 → 2 rerank activities
ctrl_orth_v_then_h(CX - DEC_W/2, Y_DEC2 + DEC_H/2, X_PAIR_L, Y_RR - 2, guard="[skip]")
ctrl_orth_v_then_h(CX + DEC_W/2, Y_DEC2 + DEC_H/2, X_PAIR_R, Y_RR - 2, guard="[default: rerank]")

# 2 rerank → merge
RR_BOTTOM = Y_RR + AH_STD
ctrl_orth_h_then_v(X_PAIR_L, RR_BOTTOM, CX, Y_MERGE2 - MRG_H/2 - 2)
ctrl_orth_h_then_v(X_PAIR_R, RR_BOTTOM, CX, Y_MERGE2 - MRG_H/2 - 2)

# Merge → Decision 3
ctrl_arrow(CX, Y_MERGE2 + MRG_H/2, CX, Y_DEC3 - 2)

# Decision 3 → 2 compression activities
ctrl_orth_v_then_h(CX - DEC_W/2, Y_DEC3 + DEC_H/2, X_PAIR_L, Y_COMP - 2, guard="[skip]")
ctrl_orth_v_then_h(CX + DEC_W/2, Y_DEC3 + DEC_H/2, X_PAIR_R, Y_COMP - 2, guard="[default: compress]")

# 2 compression → merge
COMP_BOTTOM = Y_COMP + AH_STD
ctrl_orth_h_then_v(X_PAIR_L, COMP_BOTTOM, CX, Y_MERGE3 - MRG_H/2 - 2)
ctrl_orth_h_then_v(X_PAIR_R, COMP_BOTTOM, CX, Y_MERGE3 - MRG_H/2 - 2)

# Merge → Synthesis
ctrl_arrow(CX, Y_MERGE3 + MRG_H/2, CX, Y_SYN - 2)

# Synthesis → Final
ctrl_arrow(CX, Y_SYN + AH_PILL, CX, Y_FINAL - 14)


# ---------- Measurement Store («datastore» stereotype) ----------

# Datastore rect (rounded, with stereotype text on top)
lines.append(f'  <rect x="{MX}" y="{MY}" width="{MW}" height="{MH}" rx="6" ry="6" fill="{TINT_GRAY}" stroke="{BOX_STROKE}" stroke-width="1.5"/>')
lines.append(f'  <text x="{MX + MW/2}" y="{MY + 22}" text-anchor="middle" font-size="11" font-style="italic" fill="{TEXT_SEC}">«datastore»</text>')
lines.append(f'  <text x="{MX + MW/2}" y="{MY + 42}" text-anchor="middle" font-size="13" font-weight="600" fill="{TEXT_PRI}">Measurement Store</text>')
lines.append(f'  <line x1="{MX + 12}" y1="{MY + 56}" x2="{MX + MW - 12}" y2="{MY + 56}" stroke="{DIVIDER}" stroke-width="1"/>')

# Slots (metric : source)
slot_labels = [
    ("recall@10",    "from hybrid retrieval"),
    ("nDCG@10",      "from hybrid retrieval"),
    ("recall@5",     "from rerank"),
    ("latency_ms",   "from rerank"),
    ("token_ratio",  "from compression"),
    ("judge_winner", "from synthesis"),
]
for i, (metric, source) in enumerate(slot_labels):
    yy = MY + 80 + i * 50
    lines.append(f'  <text x="{MX + 14}" y="{yy}" font-size="13" font-weight="600" fill="{TEXT_PRI}">{metric}</text>')
    lines.append(f'  <text x="{MX + 14}" y="{yy + 16}" font-size="10" fill="{TEXT_SEC}">{source}</text>')

# Object flow arrows from each measurement source → datastore
# (curved cubic bezier for separation from straight control flow)
HYB_RIGHT_X = X_CENTER + AW_STD/2
RR_RIGHT_X  = X_PAIR_R + AW_STD/2
CMP_RIGHT_X = X_PAIR_R + AW_STD/2
SYN_RIGHT_X = CX + AW_PILL/2

obj_flow(HYB_RIGHT_X, Y_RETR + AH_STD/2, MX, MY + 86,  "[recall@10, nDCG@10]")
obj_flow(RR_RIGHT_X,  Y_RR + AH_STD/2,   MX, MY + 186, "[recall@5, latency]")
obj_flow(CMP_RIGHT_X, Y_COMP + AH_STD/2, MX, MY + 286, "[token_ratio]")
obj_flow(SYN_RIGHT_X, Y_SYN + AH_PILL/2, MX, MY + 336, "[judge_winner]")


# ---------- Compact legend (UML notation primer) ----------
LX, LY = 40, H - 70
lines.append(f'  <text x="{LX}" y="{LY}" font-size="11" font-weight="600" fill="{TEXT_SEC}">UML NOTATION</text>')
lines.append(f'  <text x="{LX}" y="{LY + 18}" font-size="11" fill="{TEXT_PRI}">●  initial node     ◇  decision (with [guard] labels)     ▽  merge node     ◉  final node</text>')
lines.append(f'  <text x="{LX}" y="{LY + 36}" font-size="11" fill="{TEXT_PRI}">solid arrow = control flow        dashed arrow + [object] = object flow to «datastore»</text>')
lines.append(f'  <text x="{LX}" y="{LY + 56}" font-size="11" font-style="italic" fill="{ORANGE}">Orange = Phase 1 new (hybrid)</text>')
lines.append(f'  <text x="{LX + 240}" y="{LY + 56}" font-size="11" font-style="italic" fill="{GREEN}">Green = rerank stage</text>')
lines.append(f'  <text x="{LX + 410}" y="{LY + 56}" font-size="11" font-style="italic" fill="{PURPLE}">Purple = compression stage</text>')


# ---------- Close ----------
lines.append('</svg>')

OUT_SVG.parent.mkdir(parents=True, exist_ok=True)
OUT_SVG.write_text("\n".join(lines))
print(f"SVG written to: {OUT_SVG}")
print(f"  size: {OUT_SVG.stat().st_size} bytes")
print(f"  lines: {len(lines)}")
