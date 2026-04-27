"""Generate trial SVG for Week 2 hybrid retrieval pipeline (fireworks-tech-graph Style 1).

Source: Mermaid block in
  ~/Documents/Obsidian Vault/Agent Development Curriculum/Week 2 - Rerank and Context Compression.md
  (Architecture / Full Week 2 Pipeline)

Output:
  /Users/yuxinliu/code/agent-prep/diagrams/trial/week2-pipeline-trial.svg
"""
from pathlib import Path

OUT = Path("/Users/yuxinliu/code/agent-prep/diagrams/trial/week2-pipeline-trial.svg")

# ---------- Style 1 color tokens ----------
BG          = "#ffffff"
BOX_FILL    = "#ffffff"
BOX_STROKE  = "#d1d5db"
TEXT_PRI    = "#111827"
TEXT_SEC    = "#6b7280"
DIVIDER     = "#e5e7eb"

# Arrow / accent semantics
BLUE        = "#2563eb"   # primary data flow
ORANGE      = "#ea580c"   # hybrid / RRF accent (the new Phase 1 work)
GREEN       = "#16a34a"   # rerank
PURPLE      = "#7c3aed"   # compression
GRAY_ARROW  = "#9ca3af"   # measurement / async edges

# Tints
TINT_BLUE   = "#eff6ff"
TINT_BLUE_S = "#bfdbfe"
TINT_ORG    = "#fff7ed"
TINT_ORG_S  = "#fed7aa"
TINT_GRN    = "#f0fdf4"
TINT_GRN_S  = "#bbf7d0"
TINT_PRP    = "#faf5ff"
TINT_PRP_S  = "#ddd6fe"
TINT_GRAY   = "#f3f4f6"

FONT = "'Helvetica Neue', Helvetica, Arial, 'PingFang SC', 'Microsoft YaHei', sans-serif"

# ---------- Layout coordinates ----------
W, H = 1200, 1080

# Stage centerline
CX = 600

# Node sizes
NW_TIGHT = 160   # narrower nodes (User Query, Final Answer)
NW_STD   = 220   # standard width
NH_STD   = 70    # standard height
NW_WIDE  = 280   # wider for sub-labelled nodes
NH_WIDE  = 90    # taller for sub-labelled nodes

# Y-positions (top of each node)
Y_QUERY   = 60
Y_ENCODE  = 160
Y_LBL_R   = 270    # "Retrieval Mode" stage label
Y_RETR    = 295    # 3 retrieval-mode nodes
Y_LBL_RR  = 445    # "Rerank Stage" label
Y_RERANK  = 470    # 2 rerank nodes
Y_LBL_C   = 600    # "Compression Stage" label
Y_COMPR   = 625    # 2 compression nodes
Y_SYN     = 760    # synthesis
Y_FINAL   = 870    # final answer
Y_LEGEND  = 970    # legend

# X-positions for fan-out groups
X_DENSE   = 70     # left: dense-only
X_HYBRID  = 490    # center: hybrid (default)
X_SPARSE  = 910    # right: sparse-only
X_SKIP    = 200    # left of pair: skip
X_DEFAULT = 720    # right of pair: default downstream

# Measurement store position
MX, MY, MW, MH = 1020, 470, 130, 100


# ---------- SVG construction (Python list method per skill) ----------
lines = []
lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">')
lines.append(f'  <style>text {{ font-family: {FONT}; }}</style>')

# defs: arrow markers (one per color)
lines.append('  <defs>')
for mid, color in [("arr-blue", BLUE), ("arr-orange", ORANGE), ("arr-green", GREEN),
                   ("arr-purple", PURPLE), ("arr-gray", GRAY_ARROW)]:
    lines.append(f'    <marker id="{mid}" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">')
    lines.append(f'      <polygon points="0 0, 10 3.5, 0 7" fill="{color}"/>')
    lines.append('    </marker>')
# Drop shadow filter (used sparingly)
lines.append('    <filter id="ds" x="-10%" y="-10%" width="120%" height="120%">')
lines.append('      <feDropShadow dx="0" dy="1" stdDeviation="1.5" flood-color="#000000" flood-opacity="0.06"/>')
lines.append('    </filter>')
lines.append('  </defs>')

# Background
lines.append(f'  <rect width="{W}" height="{H}" fill="{BG}"/>')

# Title
lines.append(f'  <text x="40" y="35" font-size="18" font-weight="600" fill="{TEXT_PRI}">Week 2 — Retrieval Pipeline</text>')
lines.append(f'  <text x="40" y="55" font-size="12" fill="{TEXT_SEC}">3 retrieval modes × 2 rerank options × 2 compression options = 12 measurable A/B combinations</text>')


def std_node(x, y, w, h, label, sub=None, fill=BOX_FILL, stroke=BOX_STROKE, label_size=14, sub_size=11):
    """Standard rounded rect node with centered label and optional sub-label."""
    lines.append(f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" ry="8" fill="{fill}" stroke="{stroke}" stroke-width="1.5" filter="url(#ds)"/>')
    cx = x + w / 2
    if sub:
        lines.append(f'  <text x="{cx}" y="{y + h/2 - 3}" text-anchor="middle" font-size="{label_size}" font-weight="600" fill="{TEXT_PRI}">{label}</text>')
        lines.append(f'  <text x="{cx}" y="{y + h/2 + 16}" text-anchor="middle" font-size="{sub_size}" fill="{TEXT_SEC}">{sub}</text>')
    else:
        lines.append(f'  <text x="{cx}" y="{y + h/2 + 5}" text-anchor="middle" font-size="{label_size}" font-weight="600" fill="{TEXT_PRI}">{label}</text>')


def stage_label(x, y, text):
    lines.append(f'  <text x="{x}" y="{y}" font-size="12" font-weight="600" fill="{TEXT_SEC}" letter-spacing="0.5">{text.upper()}</text>')


def arrow(x1, y1, x2, y2, color="blue", dashed=False, label=None, label_x=None, label_y=None, label_bg=BG):
    """Straight arrow (optionally dashed) with optional mid-arrow label."""
    color_map = {"blue": BLUE, "orange": ORANGE, "green": GREEN, "purple": PURPLE, "gray": GRAY_ARROW}
    marker_map = {"blue": "arr-blue", "orange": "arr-orange", "green": "arr-green",
                  "purple": "arr-purple", "gray": "arr-gray"}
    stroke = color_map[color]
    marker = marker_map[color]
    dash_attr = ' stroke-dasharray="5,3"' if dashed else ''
    lines.append(f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="1.6"{dash_attr} marker-end="url(#{marker})"/>')
    if label:
        lx = label_x if label_x is not None else (x1 + x2) / 2
        ly = label_y if label_y is not None else (y1 + y2) / 2
        # Background rect for label readability
        text_w = len(label) * 6.5 + 8
        lines.append(f'  <rect x="{lx - text_w/2}" y="{ly - 9}" width="{text_w}" height="14" fill="{label_bg}" opacity="0.95"/>')
        lines.append(f'  <text x="{lx}" y="{ly + 2}" text-anchor="middle" font-size="11" fill="{TEXT_SEC}">{label}</text>')


def orth_arrow(x1, y1, x2, y2, color="blue", dashed=False):
    """L-shaped orthogonal arrow: down then across, or across then down."""
    color_map = {"blue": BLUE, "orange": ORANGE, "green": GREEN, "purple": PURPLE, "gray": GRAY_ARROW}
    marker_map = {"blue": "arr-blue", "orange": "arr-orange", "green": "arr-green",
                  "purple": "arr-purple", "gray": "arr-gray"}
    stroke = color_map[color]
    marker = marker_map[color]
    dash_attr = ' stroke-dasharray="5,3"' if dashed else ''
    # Path: vertical from (x1,y1) to (x1, y2) — but that's straight if x1==x2
    # For non-aligned: vertical then horizontal
    mid_y = (y1 + y2) / 2
    path = f"M {x1} {y1} L {x1} {mid_y} L {x2} {mid_y} L {x2} {y2}"
    lines.append(f'  <path d="{path}" stroke="{stroke}" stroke-width="1.6" fill="none"{dash_attr} marker-end="url(#{marker})"/>')


# ---------- Nodes ----------

# 1. User Query
std_node(CX - NW_TIGHT/2, Y_QUERY, NW_TIGHT, 50, "User Query")

# 2. BGE-M3 Encode (with sub-label for "dense + sparse, one forward pass")
std_node(CX - NW_WIDE/2, Y_ENCODE, NW_WIDE, NH_WIDE, "BGE-M3 Encode",
         sub="dense (1024-d) + sparse (lex weights) — one pass",
         fill=TINT_BLUE, stroke=TINT_BLUE_S)

# Stage label: Retrieval Mode
stage_label(40, Y_LBL_R + 12, "1. Retrieval Mode (3 A/B variants)")

# 3a. Dense-only (Week 1 baseline)
std_node(X_DENSE, Y_RETR, NW_WIDE, NH_WIDE, "Dense-only",
         sub="bge_m3_hnsw · using=dense", fill=TINT_GRAY, stroke=BOX_STROKE)
lines.append(f'  <text x="{X_DENSE + NW_WIDE/2}" y="{Y_RETR + NH_WIDE + 14}" text-anchor="middle" font-size="10" fill="{TEXT_SEC}">Week 1 baseline</text>')

# 3b. Hybrid (default — orange accent for new Phase 1 work)
std_node(X_HYBRID, Y_RETR, NW_WIDE, NH_WIDE, "Hybrid (RRF k=60)",
         sub="bge_m3_hybrid · dense + sparse prefetch → fuse",
         fill=TINT_ORG, stroke=TINT_ORG_S)
lines.append(f'  <text x="{X_HYBRID + NW_WIDE/2}" y="{Y_RETR + NH_WIDE + 14}" text-anchor="middle" font-size="10" font-weight="600" fill="{ORANGE}">DEFAULT — Phase 1 new</text>')

# 3c. Sparse-only
std_node(X_SPARSE, Y_RETR, NW_WIDE, NH_WIDE, "Sparse-only",
         sub="bge_m3_hybrid · using=sparse", fill=TINT_GRAY, stroke=BOX_STROKE)
lines.append(f'  <text x="{X_SPARSE + NW_WIDE/2}" y="{Y_RETR + NH_WIDE + 14}" text-anchor="middle" font-size="10" fill="{TEXT_SEC}">A/B comparison</text>')

# Stage label: Rerank
stage_label(40, Y_LBL_RR + 12, "2. Rerank Stage (skip / cross-encoder)")

# 4a. Skip rerank
std_node(X_SKIP, Y_RERANK, NW_WIDE, NH_STD, "Skip rerank",
         sub="top-5 by retrieval score", fill=TINT_GRAY, stroke=BOX_STROKE,
         label_size=13, sub_size=10)

# 4b. BGE-reranker (default)
std_node(X_DEFAULT, Y_RERANK, NW_WIDE, NH_STD, "BGE-reranker-v2-m3",
         sub="cross-encoder · batch=32 · ~80-140ms",
         fill=TINT_GRN, stroke=TINT_GRN_S, label_size=13, sub_size=10)
lines.append(f'  <text x="{X_DEFAULT + NW_WIDE/2}" y="{Y_RERANK + NH_STD + 14}" text-anchor="middle" font-size="10" font-weight="600" fill="{GREEN}">DEFAULT</text>')

# Stage label: Compression
stage_label(40, Y_LBL_C + 12, "3. Compression Stage (raw / extract-only)")

# 5a. Skip compression
std_node(X_SKIP, Y_COMPR, NW_WIDE, NH_STD, "Raw context",
         sub="5 full passages", fill=TINT_GRAY, stroke=BOX_STROKE,
         label_size=13, sub_size=10)

# 5b. Compressor (default)
std_node(X_DEFAULT, Y_COMPR, NW_WIDE, NH_STD, "Gemma 26B Compressor",
         sub="extract-only · temp=0 · max=500 tok",
         fill=TINT_PRP, stroke=TINT_PRP_S, label_size=13, sub_size=10)
lines.append(f'  <text x="{X_DEFAULT + NW_WIDE/2}" y="{Y_COMPR + NH_STD + 14}" text-anchor="middle" font-size="10" font-weight="600" fill="{PURPLE}">DEFAULT</text>')

# 6. Synthesis
std_node(CX - NW_WIDE/2, Y_SYN, NW_WIDE, NH_STD, "Synthesis Model",
         sub="answer generation · grounded on retrieved context",
         fill=TINT_BLUE, stroke=TINT_BLUE_S, label_size=13, sub_size=10)

# 7. Final Answer
std_node(CX - NW_TIGHT/2, Y_FINAL, NW_TIGHT, 50, "Final Answer")

# 8. Measurement Store (cylinder-ish — rect with horizontal lines)
lines.append(f'  <rect x="{MX}" y="{MY}" width="{MW}" height="{MH}" rx="6" ry="6" fill="{TINT_GRAY}" stroke="{BOX_STROKE}" stroke-width="1.5"/>')
lines.append(f'  <line x1="{MX+10}" y1="{MY+30}" x2="{MX+MW-10}" y2="{MY+30}" stroke="{DIVIDER}" stroke-width="1"/>')
lines.append(f'  <line x1="{MX+10}" y1="{MY+50}" x2="{MX+MW-10}" y2="{MY+50}" stroke="{DIVIDER}" stroke-width="1"/>')
lines.append(f'  <line x1="{MX+10}" y1="{MY+70}" x2="{MX+MW-10}" y2="{MY+70}" stroke="{DIVIDER}" stroke-width="1"/>')
lines.append(f'  <text x="{MX + MW/2}" y="{MY+18}" text-anchor="middle" font-size="12" font-weight="600" fill="{TEXT_PRI}">Measurement Store</text>')
lines.append(f'  <text x="{MX + MW/2}" y="{MY+44}" text-anchor="middle" font-size="10" fill="{TEXT_SEC}">recall@K · nDCG@K</text>')
lines.append(f'  <text x="{MX + MW/2}" y="{MY+64}" text-anchor="middle" font-size="10" fill="{TEXT_SEC}">latency · token ratio</text>')
lines.append(f'  <text x="{MX + MW/2}" y="{MY+84}" text-anchor="middle" font-size="10" fill="{TEXT_SEC}">judge A/B winner</text>')


# ---------- Arrows: primary path ----------

# Query → Encode
arrow(CX, Y_QUERY + 50, CX, Y_ENCODE - 2, color="blue")

# Encode → 3 retrieval modes (orange for the new Phase 1 fan)
ENC_BOTTOM_Y = Y_ENCODE + NH_WIDE
RETR_TOP_Y = Y_RETR
arrow(CX, ENC_BOTTOM_Y, X_DENSE + NW_WIDE/2, RETR_TOP_Y - 2, color="blue")
arrow(CX, ENC_BOTTOM_Y, X_HYBRID + NW_WIDE/2, RETR_TOP_Y - 2, color="orange")
arrow(CX, ENC_BOTTOM_Y, X_SPARSE + NW_WIDE/2, RETR_TOP_Y - 2, color="blue")

# Each retrieval mode → rerank fanout point (use orth arrows converging)
RETR_BOTTOM_Y = Y_RETR + NH_WIDE
RERANK_TOP_Y = Y_RERANK
# Converge to a midpoint just above the rerank stage
CONV_RR_Y = Y_LBL_RR

# Dense-only → both rerank options
arrow(X_DENSE + NW_WIDE/2, RETR_BOTTOM_Y, X_SKIP + NW_WIDE/2, RERANK_TOP_Y - 2, color="blue")
# Hybrid → both rerank options (orange — primary path)
arrow(X_HYBRID + NW_WIDE/2, RETR_BOTTOM_Y, X_SKIP + NW_WIDE/2, RERANK_TOP_Y - 2, color="orange")
arrow(X_HYBRID + NW_WIDE/2, RETR_BOTTOM_Y, X_DEFAULT + NW_WIDE/2, RERANK_TOP_Y - 2, color="orange")
# Sparse-only → both
arrow(X_SPARSE + NW_WIDE/2, RETR_BOTTOM_Y, X_DEFAULT + NW_WIDE/2, RERANK_TOP_Y - 2, color="blue")

# Rerank → both compression options
RERANK_BOTTOM_Y = Y_RERANK + NH_STD
COMPR_TOP_Y = Y_COMPR
arrow(X_SKIP + NW_WIDE/2, RERANK_BOTTOM_Y, X_SKIP + NW_WIDE/2, COMPR_TOP_Y - 2, color="green")
arrow(X_SKIP + NW_WIDE/2, RERANK_BOTTOM_Y, X_DEFAULT + NW_WIDE/2, COMPR_TOP_Y - 2, color="green")
arrow(X_DEFAULT + NW_WIDE/2, RERANK_BOTTOM_Y, X_SKIP + NW_WIDE/2, COMPR_TOP_Y - 2, color="green")
arrow(X_DEFAULT + NW_WIDE/2, RERANK_BOTTOM_Y, X_DEFAULT + NW_WIDE/2, COMPR_TOP_Y - 2, color="green")

# Compression → Synthesis
COMPR_BOTTOM_Y = Y_COMPR + NH_STD
SYN_TOP_Y = Y_SYN
arrow(X_SKIP + NW_WIDE/2, COMPR_BOTTOM_Y, CX - 30, SYN_TOP_Y - 2, color="purple")
arrow(X_DEFAULT + NW_WIDE/2, COMPR_BOTTOM_Y, CX + 30, SYN_TOP_Y - 2, color="purple")

# Synthesis → Final Answer
arrow(CX, Y_SYN + NH_STD, CX, Y_FINAL - 2, color="blue")

# ---------- Measurement edges (dashed gray) ----------

# Hybrid → Measurement
arrow(X_HYBRID + NW_WIDE, Y_RETR + NH_WIDE/2, MX, MY + 30, color="gray", dashed=True)
# BGE-reranker → Measurement
arrow(X_DEFAULT + NW_WIDE, Y_RERANK + NH_STD/2, MX, MY + 50, color="gray", dashed=True)
# Compressor → Measurement
arrow(X_DEFAULT + NW_WIDE, Y_COMPR + NH_STD/2, MX, MY + 70, color="gray", dashed=True)
# Synthesis → Measurement (curved up to the right)
lines.append(f'  <path d="M {CX + NW_WIDE/2} {Y_SYN + NH_STD/2} C 920 {Y_SYN + NH_STD/2}, 1000 {MY+90}, {MX} {MY+90}" stroke="{GRAY_ARROW}" stroke-width="1.6" fill="none" stroke-dasharray="5,3" marker-end="url(#arr-gray)"/>')


# ---------- Legend ----------

legend_x = 40
legend_y = Y_LEGEND
lines.append(f'  <text x="{legend_x}" y="{legend_y}" font-size="12" font-weight="600" fill="{TEXT_SEC}">LEGEND</text>')

def legend_row(idx, color, marker, label):
    yy = legend_y + 20 + idx * 22
    lines.append(f'  <line x1="{legend_x}" y1="{yy}" x2="{legend_x + 40}" y2="{yy}" stroke="{color}" stroke-width="1.6" marker-end="url(#{marker})"/>')
    lines.append(f'  <text x="{legend_x + 50}" y="{yy + 4}" font-size="12" fill="{TEXT_PRI}">{label}</text>')

def legend_row_dashed(idx, color, marker, label):
    yy = legend_y + 20 + idx * 22
    lines.append(f'  <line x1="{legend_x}" y1="{yy}" x2="{legend_x + 40}" y2="{yy}" stroke="{color}" stroke-width="1.6" stroke-dasharray="5,3" marker-end="url(#{marker})"/>')
    lines.append(f'  <text x="{legend_x + 50}" y="{yy + 4}" font-size="12" fill="{TEXT_PRI}">{label}</text>')

legend_row(0, BLUE,   "arr-blue",   "Primary data flow (encode, synthesis)")
legend_row(1, ORANGE, "arr-orange", "Hybrid retrieval path (Phase 1 — new)")
legend_row(2, GREEN,  "arr-green",  "Rerank stage (skip vs cross-encoder)")
legend_row(3, PURPLE, "arr-purple", "Compression stage (skip vs extract-only)")
legend_row_dashed(4, GRAY_ARROW, "arr-gray", "Measurement edges (recorded for RESULTS.md)")


# ---------- Close SVG ----------
lines.append('</svg>')

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(lines))
print(f"SVG written to: {OUT}")
print(f"  size: {OUT.stat().st_size} bytes")
print(f"  lines: {len(lines)}")
