"""Generate UML Activity Diagram for Week 2 hybrid retrieval pipeline (Style 1).

Source: same Mermaid block in
  ~/Documents/Obsidian Vault/Agent Development Curriculum/Week 2 - Rerank and Context Compression.md
Compare against: ./week2-pipeline-trial.svg (architecture-flowchart style)

UML conventions used:
  ● initial node (filled black circle, r=8)
  ◇ decision node (diamond) — question text BELOW diamond
  ▽ merge node (small unlabeled diamond) — converges branches
  rounded rect = activity (rx=20, pill shape)
  [guard] = condition labels on outgoing decision arrows
  ◉ final node (filled black circle inside hollow circle)
  dashed arrow + [object_name] = object flow (data passing as side-effect)
"""
from pathlib import Path

OUT = Path("/Users/yuxinliu/code/agent-prep/diagrams/trial/week2-pipeline-uml-activity.svg")

# Style 1 tokens
BG, BOX_FILL, BOX_STROKE = "#ffffff", "#ffffff", "#d1d5db"
TEXT_PRI, TEXT_SEC       = "#111827", "#6b7280"
DIVIDER                  = "#e5e7eb"

# UML control flow: black; object flow: dashed gray
CTRL = "#1f2937"     # near-black for control flow arrows (UML convention)
OBJ  = "#9ca3af"     # gray for object flow / measurement (dashed)

# Subtle accent tints (kept from v1 to preserve "Phase 1 highlight" narrative)
TINT_BLUE,  TINT_BLUE_S = "#eff6ff", "#bfdbfe"
TINT_ORG,   TINT_ORG_S  = "#fff7ed", "#fed7aa"
TINT_GRN,   TINT_GRN_S  = "#f0fdf4", "#bbf7d0"
TINT_PRP,   TINT_PRP_S  = "#faf5ff", "#ddd6fe"
TINT_GRAY               = "#f3f4f6"

ORANGE, GREEN, PURPLE = "#ea580c", "#16a34a", "#7c3aed"

FONT = "'Helvetica Neue', Helvetica, Arial, 'PingFang SC', 'Microsoft YaHei', sans-serif"

# ---------- Layout ----------
W, H = 1200, 1100
CX = 580   # main flow centerline (left of center to leave room for measurement panel)

# Y-positions (top edge of each element, except for circles where it's center y)
Y_TITLE   = 35
Y_SUB     = 58
Y_INIT    = 100   # initial circle center
Y_ENC     = 130   # BGE-M3 Encode top
Y_DEC1    = 235   # Decision: Retrieval Mode (top)
Y_RETR    = 340   # Three retrieval activities (top)
Y_MERGE1  = 460   # Merge after retrieval (center)
Y_DEC2    = 500   # Decision: Rerank? (top)
Y_RR      = 605   # Two rerank activities (top)
Y_MERGE2  = 705   # Merge after rerank (center)
Y_DEC3    = 740   # Decision: Compress? (top)
Y_COMP    = 845   # Two compression activities (top)
Y_MERGE3  = 945   # Merge after compress (center)
Y_SYN     = 975   # Synthesis (top)
Y_FINAL   = 1060  # Final node center

# X-positions for fan-out
X_LEFT    = 200    # left of 3-way fan
X_CENTER  = CX     # centerline
X_RIGHT   = 960    # right of 3-way fan
X_PAIR_L  = 350    # left of 2-way pair
X_PAIR_R  = 810    # right of 2-way pair

# Activity sizes
AW_STD, AH_STD = 200, 70    # standard activity
AW_PILL, AH_PILL = 160, 50  # narrower pill (Encode, Synthesis)
DEC_W, DEC_H = 120, 60      # decision diamond
MRG_W, MRG_H = 28, 22       # merge diamond (small, unlabeled)

# Measurement panel
MX, MY, MW, MH = 1000, 460, 170, 320


# ---------- SVG ----------
lines = []
lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">')
lines.append(f'  <style>text {{ font-family: {FONT}; }}</style>')

# Markers
lines.append('  <defs>')
for mid, color in [("ctrl", CTRL), ("obj", OBJ)]:
    lines.append(f'    <marker id="arr-{mid}" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">')
    lines.append(f'      <polygon points="0 0, 10 3.5, 0 7" fill="{color}"/>')
    lines.append('    </marker>')
lines.append('    <filter id="ds" x="-10%" y="-10%" width="120%" height="120%">')
lines.append('      <feDropShadow dx="0" dy="1" stdDeviation="1.5" flood-color="#000000" flood-opacity="0.06"/>')
lines.append('    </filter>')
lines.append('  </defs>')

# Background
lines.append(f'  <rect width="{W}" height="{H}" fill="{BG}"/>')

# Title
lines.append(f'  <text x="40" y="{Y_TITLE}" font-size="18" font-weight="600" fill="{TEXT_PRI}">Week 2 — Retrieval Pipeline (UML Activity Diagram)</text>')
lines.append(f'  <text x="40" y="{Y_SUB}" font-size="12" fill="{TEXT_SEC}">3 retrieval modes × 2 rerank options × 2 compression options = 12 measurable A/B combinations</text>')


# ---------- Helpers ----------

def activity(cx, ytop, w, h, label, sub=None, fill=BOX_FILL, stroke=BOX_STROKE):
    """Rounded-rect activity (UML pill shape)."""
    x = cx - w / 2
    rx = h / 2  # full pill — UML convention
    lines.append(f'  <rect x="{x}" y="{ytop}" width="{w}" height="{h}" rx="{rx}" ry="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="1.5" filter="url(#ds)"/>')
    if sub:
        lines.append(f'  <text x="{cx}" y="{ytop + h/2 - 2}" text-anchor="middle" font-size="13" font-weight="600" fill="{TEXT_PRI}">{label}</text>')
        lines.append(f'  <text x="{cx}" y="{ytop + h/2 + 14}" text-anchor="middle" font-size="11" fill="{TEXT_SEC}">{sub}</text>')
    else:
        lines.append(f'  <text x="{cx}" y="{ytop + h/2 + 5}" text-anchor="middle" font-size="14" font-weight="600" fill="{TEXT_PRI}">{label}</text>')


def decision(cx, ytop, w, h, question):
    """UML decision node — diamond with question text BELOW the diamond."""
    cy = ytop + h / 2
    pts = f"{cx},{ytop} {cx + w/2},{cy} {cx},{ytop + h} {cx - w/2},{cy}"
    lines.append(f'  <polygon points="{pts}" fill="{BOX_FILL}" stroke="{CTRL}" stroke-width="1.5"/>')
    # Question label below the diamond, centered
    lines.append(f'  <text x="{cx}" y="{ytop + h + 14}" text-anchor="middle" font-size="12" font-weight="600" fill="{TEXT_PRI}">{question}</text>')


def merge(cx, cy):
    """Unlabeled merge diamond (small)."""
    pts = f"{cx},{cy - MRG_H/2} {cx + MRG_W/2},{cy} {cx},{cy + MRG_H/2} {cx - MRG_W/2},{cy}"
    lines.append(f'  <polygon points="{pts}" fill="{BOX_FILL}" stroke="{CTRL}" stroke-width="1.5"/>')


def initial_node(cx, cy):
    """UML initial node — filled black circle."""
    lines.append(f'  <circle cx="{cx}" cy="{cy}" r="8" fill="{CTRL}"/>')


def final_node(cx, cy):
    """UML final node — filled black circle inside hollow ring (bullseye)."""
    lines.append(f'  <circle cx="{cx}" cy="{cy}" r="14" fill="{BOX_FILL}" stroke="{CTRL}" stroke-width="2"/>')
    lines.append(f'  <circle cx="{cx}" cy="{cy}" r="7" fill="{CTRL}"/>')


def ctrl_arrow(x1, y1, x2, y2, guard=None, guard_offset_x=8, guard_offset_y=-4):
    """Control flow arrow (solid black) with optional [guard] label near source."""
    lines.append(f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{CTRL}" stroke-width="1.6" marker-end="url(#arr-ctrl)"/>')
    if guard:
        # Place guard label near the source end, offset slightly
        gx = x1 + guard_offset_x
        gy = y1 + guard_offset_y
        text_w = len(guard) * 6.2 + 6
        lines.append(f'  <rect x="{gx - 2}" y="{gy - 11}" width="{text_w}" height="14" fill="{BG}" opacity="0.95"/>')
        lines.append(f'  <text x="{gx}" y="{gy}" font-size="11" font-style="italic" fill="{TEXT_PRI}">{guard}</text>')


def ctrl_arrow_orth_v_then_h(x1, y1, x2, y2, guard=None):
    """Orthogonal arrow: vertical from source, horizontal to target. Used for fan-out from decisions."""
    # Vertical first to a junction, then horizontal
    junction_y = y1 + (y2 - y1) * 0.4
    path = f"M {x1} {y1} L {x1} {junction_y} L {x2} {junction_y} L {x2} {y2}"
    lines.append(f'  <path d="{path}" stroke="{CTRL}" stroke-width="1.6" fill="none" marker-end="url(#arr-ctrl)"/>')
    if guard:
        # Place guard label on the horizontal segment
        gx = (x1 + x2) / 2
        gy = junction_y - 6
        text_w = len(guard) * 6.2 + 6
        lines.append(f'  <rect x="{gx - text_w/2}" y="{gy - 11}" width="{text_w}" height="14" fill="{BG}" opacity="0.95"/>')
        lines.append(f'  <text x="{gx}" y="{gy}" text-anchor="middle" font-size="11" font-style="italic" fill="{TEXT_PRI}">{guard}</text>')


def ctrl_arrow_orth_h_then_v(x1, y1, x2, y2):
    """Orthogonal arrow: horizontal then vertical. Used for converge into merge nodes."""
    path = f"M {x1} {y1} L {x1} {y2 - 30} L {x2} {y2 - 30} L {x2} {y2}"
    # Actually simpler: go down to merge level, then across
    # But for converging into merge: go across to centerline, then down
    junction_y = y1 + (y2 - y1) * 0.6
    path = f"M {x1} {y1} L {x1} {junction_y} L {x2} {junction_y} L {x2} {y2}"
    lines.append(f'  <path d="{path}" stroke="{CTRL}" stroke-width="1.6" fill="none" marker-end="url(#arr-ctrl)"/>')


def obj_flow(x1, y1, x2, y2, obj_label):
    """Dashed object-flow arrow with [object_label] mid-arrow. Curved path for clarity."""
    # Cubic bezier from source to target
    cp1x = x1 + (x2 - x1) * 0.5
    cp1y = y1
    cp2x = x2 - 30
    cp2y = y2
    path = f"M {x1} {y1} C {cp1x} {cp1y}, {cp2x} {cp2y}, {x2} {y2}"
    lines.append(f'  <path d="{path}" stroke="{OBJ}" stroke-width="1.4" fill="none" stroke-dasharray="5,3" marker-end="url(#arr-obj)"/>')
    # Object label mid-arrow with background
    lx = (x1 + x2) / 2 + 30
    ly = (y1 + y2) / 2
    text_w = len(obj_label) * 6.2 + 8
    lines.append(f'  <rect x="{lx - text_w/2}" y="{ly - 9}" width="{text_w}" height="14" fill="{BG}" opacity="0.95"/>')
    lines.append(f'  <text x="{lx}" y="{ly + 2}" text-anchor="middle" font-size="11" font-style="italic" fill="{TEXT_SEC}">{obj_label}</text>')


# ---------- Build the diagram ----------

# Initial node
initial_node(CX, Y_INIT)

# BGE-M3 Encode (blue tint — primary path)
activity(CX, Y_ENC, AW_PILL, AH_PILL, "BGE-M3 Encode",
         fill=TINT_BLUE, stroke=TINT_BLUE_S)

# Decision 1: Retrieval Mode?
decision(CX, Y_DEC1, DEC_W, DEC_H, "Retrieval Mode?")

# Three retrieval activities
activity(X_LEFT, Y_RETR, AW_STD, AH_STD, "Dense-only",
         sub="bge_m3_hnsw · using=dense", fill=TINT_GRAY)
activity(X_CENTER, Y_RETR, AW_STD, AH_STD, "Hybrid (RRF k=60)",
         sub="dense + sparse → fuse", fill=TINT_ORG, stroke=TINT_ORG_S)
activity(X_RIGHT, Y_RETR, AW_STD, AH_STD, "Sparse-only",
         sub="bge_m3_hybrid · using=sparse", fill=TINT_GRAY)

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


# ---------- Control flow arrows ----------

# Initial → Encode
ctrl_arrow(CX, Y_INIT + 8, CX, Y_ENC - 2)

# Encode → Decision 1
ctrl_arrow(CX, Y_ENC + AH_PILL, CX, Y_DEC1 - 2)

# Decision 1 → 3 retrieval activities (orthogonal fan-out with guards)
DEC1_BOTTOM = Y_DEC1 + DEC_H
ctrl_arrow_orth_v_then_h(CX - DEC_W/2 + 5, Y_DEC1 + DEC_H/2, X_LEFT,   Y_RETR - 2, guard="[A/B: dense]")
ctrl_arrow(CX, DEC1_BOTTOM + 30, CX, Y_RETR - 2, guard="[default: hybrid]", guard_offset_x=10, guard_offset_y=-4)
# vertical line for the default
ctrl_arrow_orth_v_then_h(CX + DEC_W/2 - 5, Y_DEC1 + DEC_H/2, X_RIGHT, Y_RETR - 2, guard="[A/B: sparse]")

# 3 retrieval activities → merge (orthogonal converge)
RETR_BOTTOM = Y_RETR + AH_STD
ctrl_arrow_orth_h_then_v(X_LEFT,   RETR_BOTTOM, CX, Y_MERGE1 - MRG_H/2 - 2)
ctrl_arrow(CX, RETR_BOTTOM, CX, Y_MERGE1 - MRG_H/2 - 2)
ctrl_arrow_orth_h_then_v(X_RIGHT,  RETR_BOTTOM, CX, Y_MERGE1 - MRG_H/2 - 2)

# Merge → Decision 2
ctrl_arrow(CX, Y_MERGE1 + MRG_H/2, CX, Y_DEC2 - 2)

# Decision 2 → 2 rerank activities
ctrl_arrow_orth_v_then_h(CX - DEC_W/2 + 5, Y_DEC2 + DEC_H/2, X_PAIR_L, Y_RR - 2, guard="[skip]")
ctrl_arrow_orth_v_then_h(CX + DEC_W/2 - 5, Y_DEC2 + DEC_H/2, X_PAIR_R, Y_RR - 2, guard="[default: rerank]")

# 2 rerank → merge
RR_BOTTOM = Y_RR + AH_STD
ctrl_arrow_orth_h_then_v(X_PAIR_L, RR_BOTTOM, CX, Y_MERGE2 - MRG_H/2 - 2)
ctrl_arrow_orth_h_then_v(X_PAIR_R, RR_BOTTOM, CX, Y_MERGE2 - MRG_H/2 - 2)

# Merge → Decision 3
ctrl_arrow(CX, Y_MERGE2 + MRG_H/2, CX, Y_DEC3 - 2)

# Decision 3 → 2 compression activities
ctrl_arrow_orth_v_then_h(CX - DEC_W/2 + 5, Y_DEC3 + DEC_H/2, X_PAIR_L, Y_COMP - 2, guard="[skip]")
ctrl_arrow_orth_v_then_h(CX + DEC_W/2 - 5, Y_DEC3 + DEC_H/2, X_PAIR_R, Y_COMP - 2, guard="[default: compress]")

# 2 compression → merge
COMP_BOTTOM = Y_COMP + AH_STD
ctrl_arrow_orth_h_then_v(X_PAIR_L, COMP_BOTTOM, CX, Y_MERGE3 - MRG_H/2 - 2)
ctrl_arrow_orth_h_then_v(X_PAIR_R, COMP_BOTTOM, CX, Y_MERGE3 - MRG_H/2 - 2)

# Merge → Synthesis
ctrl_arrow(CX, Y_MERGE3 + MRG_H/2, CX, Y_SYN - 2)

# Synthesis → Final
ctrl_arrow(CX, Y_SYN + AH_PILL, CX, Y_FINAL - 14)


# ---------- Measurement Store (right side, with object flows) ----------

# Datastore-style rect (UML datastore convention: rect with double horizontal lines top/bottom or stereotype)
lines.append(f'  <rect x="{MX}" y="{MY}" width="{MW}" height="{MH}" rx="6" ry="6" fill="{TINT_GRAY}" stroke="{BOX_STROKE}" stroke-width="1.5"/>')
lines.append(f'  <text x="{MX + MW/2}" y="{MY + 22}" text-anchor="middle" font-size="11" font-style="italic" fill="{TEXT_SEC}">«datastore»</text>')
lines.append(f'  <text x="{MX + MW/2}" y="{MY + 42}" text-anchor="middle" font-size="13" font-weight="600" fill="{TEXT_PRI}">Measurement Store</text>')
lines.append(f'  <line x1="{MX + 12}" y1="{MY + 56}" x2="{MX + MW - 12}" y2="{MY + 56}" stroke="{DIVIDER}" stroke-width="1"/>')
# Slots inside the datastore
slot_labels = [
    ("recall@10",       "hybrid retrieval"),
    ("nDCG@10",         "hybrid retrieval"),
    ("recall@5",        "rerank"),
    ("latency_ms",      "rerank"),
    ("token_ratio",     "compression"),
    ("judge_winner",    "synthesis"),
]
for i, (metric, source) in enumerate(slot_labels):
    yy = MY + 80 + i * 36
    lines.append(f'  <text x="{MX + 14}" y="{yy}" font-size="12" font-weight="600" fill="{TEXT_PRI}">{metric}</text>')
    lines.append(f'  <text x="{MX + 14}" y="{yy + 14}" font-size="10" fill="{TEXT_SEC}">from {source}</text>')

# Object flows (dashed arrows from each measurement source to datastore)
# From hybrid retrieval (right edge of Hybrid activity)
HYB_RIGHT_X = X_CENTER + AW_STD/2
obj_flow(HYB_RIGHT_X, Y_RETR + AH_STD/2,        MX, MY + 86,   "[recall@10, nDCG@10]")
# From BGE-reranker
RR_RIGHT_X = X_PAIR_R + AW_STD/2
obj_flow(RR_RIGHT_X, Y_RR + AH_STD/2,           MX, MY + 158,  "[recall@5, latency]")
# From compressor
CMP_RIGHT_X = X_PAIR_R + AW_STD/2
obj_flow(CMP_RIGHT_X, Y_COMP + AH_STD/2,        MX, MY + 230,  "[token_ratio]")
# From synthesis
SYN_RIGHT_X = CX + AW_PILL/2
obj_flow(SYN_RIGHT_X, Y_SYN + AH_PILL/2,        MX, MY + 266,  "[judge_winner]")


# ---------- Mini legend (only what's not self-evident from UML notation) ----------

LX, LY = 40, 1010
lines.append(f'  <text x="{LX}" y="{LY}" font-size="11" font-weight="600" fill="{TEXT_SEC}">UML NOTATION</text>')
lines.append(f'  <text x="{LX}" y="{LY + 18}" font-size="11" fill="{TEXT_PRI}">●  initial node     ◇  decision (with [guard] labels)     ▽  merge node     ◉  final node</text>')
lines.append(f'  <text x="{LX}" y="{LY + 36}" font-size="11" fill="{TEXT_PRI}">solid arrow = control flow        dashed arrow + [object] = object flow to «datastore»</text>')
lines.append(f'  <text x="{LX}" y="{LY + 56}" font-size="11" font-style="italic" fill="{ORANGE}">Orange tint = Phase 1 new (hybrid retrieval)         </text>')
lines.append(f'  <text x="{LX + 380}" y="{LY + 56}" font-size="11" font-style="italic" fill="{GREEN}">Green = rerank stage         </text>')
lines.append(f'  <text x="{LX + 560}" y="{LY + 56}" font-size="11" font-style="italic" fill="{PURPLE}">Purple = compression stage</text>')


# ---------- Close ----------
lines.append('</svg>')

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(lines))
print(f"SVG written to: {OUT}")
print(f"  size: {OUT.stat().st_size} bytes")
print(f"  lines: {len(lines)}")
