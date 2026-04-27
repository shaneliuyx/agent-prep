"""Week 8 — 5-Layer Schema Reliability Defense (signature diagram).

Source: Week 8 - Schema Reliability Bench.md, mermaid block 1.
Style 1 (Flat Icon) per fireworks-tech-graph skill.

UML type: Architecture / layered-defense visual (closest UML: Activity Diagram with stacked
swim-lane layers, but rendered as a defense-in-depth tower for visual clarity).

Outputs:
  /Users/yuxinliu/code/agent-prep/diagrams/week-8/5layer-reliability.svg
  /Users/yuxinliu/code/agent-prep/diagrams/week-8/5layer-reliability.png
  /Users/yuxinliu/Documents/Obsidian Vault/Agent Development Curriculum/assets/diagrams/week-8/5layer-reliability.png
"""
from pathlib import Path

OUT_SVG  = Path("/Users/yuxinliu/code/agent-prep/diagrams/week-8/5layer-reliability.svg")
OUT_PNG  = Path("/Users/yuxinliu/code/agent-prep/diagrams/week-8/5layer-reliability.png")
VAULT_PNG = Path("/Users/yuxinliu/Documents/Obsidian Vault/Agent Development Curriculum/assets/diagrams/week-8/5layer-reliability.png")

# Style 1 tokens
BG = "#ffffff"
TEXT_PRI, TEXT_SEC = "#111827", "#6b7280"
DIVIDER = "#e5e7eb"

# Layered defense palette — light at top (best case), darkening downward
L1 = ("#dbeafe", "#60a5fa", "#1e40af")  # bg, stroke, text — blue
L2 = ("#dcfce7", "#4ade80", "#14532d")  # green
L3 = ("#fef3c7", "#facc15", "#854d0e")  # amber
L4 = ("#fce7f3", "#f472b6", "#831843")  # pink
L5 = ("#f3f4f6", "#9ca3af", "#374151")  # gray (last resort)

FONT = "'Helvetica Neue', Helvetica, Arial, 'PingFang SC', 'Microsoft YaHei', sans-serif"

W, H = 1200, 880

# Layer geometry — stacked top-to-bottom (L1 highest = first defense)
# Center the tower horizontally
TOWER_X = 240   # left edge of widest layer
TOWER_W = 720   # widest layer (L5 — base of defense)

# Each layer narrows toward the top to reinforce "fewer cases reach this layer"
# But for readability, keep them all the same width with consistent label space
LAYER_H = 110
LAYER_GAP = 18
LAYER_Y0 = 130

# Right column: cumulative reliability annotations
GAUGE_X = 1000
GAUGE_W = 160

# Defense-in-depth metaphor: lower is the LAST line, upper is FIRST line
# So in the SVG we'll draw L1 at top, L5 at bottom (reading order)


lines = []
lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">')
lines.append(f'  <style>text {{ font-family: {FONT}; }}</style>')

# Defs — drop shadow + arrow markers
lines.append('  <defs>')
lines.append('    <filter id="ds" x="-10%" y="-10%" width="120%" height="120%">')
lines.append('      <feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#000000" flood-opacity="0.08"/>')
lines.append('    </filter>')
lines.append('    <marker id="arr-down" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">')
lines.append('      <polygon points="0 0, 10 3.5, 0 7" fill="#1f2937"/>')
lines.append('    </marker>')
lines.append('  </defs>')

# White background
lines.append(f'  <rect width="{W}" height="{H}" fill="{BG}"/>')

# Title
lines.append(f'  <text x="40" y="40" font-size="22" font-weight="700" fill="{TEXT_PRI}">5-Layer Schema Reliability Defense</text>')
lines.append(f'  <text x="40" y="65" font-size="13" fill="{TEXT_SEC}">Each layer catches what the layer above it missed. Cumulative reliability rises from ~70% (L1 alone) to ~99.9% (full stack).</text>')
lines.append(f'  <text x="40" y="85" font-size="11" font-style="italic" fill="{TEXT_SEC}">Read top-down: L1 is your first line of defense (cheapest), L5 is the last-resort safety net.</text>')


# Layer definitions — (idx, name, sublabel, palette, cumulative_pct, fall_through_label)
LAYERS = [
    ("L1", "Schema + Prompt Design",  "reasoning-BEFORE-answer · enums · min nesting · 1 worked example",          L1, "~70%",   "fall-through: ambiguous schema cases"),
    ("L2", "Constrained Decoding",    "Outlines · xgrammar · provider-native strict JSON",                          L2, "~94%",   "fall-through: server doesn't support strict mode"),
    ("L3", "Instructor + Pydantic",   "auto-retry with ValidationError injected as conversational feedback",        L3, "~99.3%", "fall-through: semantic rules beyond shape"),
    ("L4", "Post-validation + Repair", "semantic rules (end_date > start_date, totals) · retry ≤ 1",                 L4, "~99.8%", "fall-through: corrupted output that even repair can't fix"),
    ("L5", "Defensive Parsing",       "regex scrape · safe defaults · log + page · last resort",                    L5, "~99.9%", "(last layer — DLQ + alert)"),
]

# Draw each layer
for i, (lid, name, sublabel, palette, cum_pct, fall_label) in enumerate(LAYERS):
    bg, stroke, text_color = palette
    y = LAYER_Y0 + i * (LAYER_H + LAYER_GAP)

    # Layer rect (full width)
    lines.append(f'  <rect x="{TOWER_X}" y="{y}" width="{TOWER_W}" height="{LAYER_H}" rx="10" ry="10" fill="{bg}" stroke="{stroke}" stroke-width="2" filter="url(#ds)"/>')

    # Layer ID badge (left side, large)
    badge_cx = TOWER_X + 50
    lines.append(f'  <circle cx="{badge_cx}" cy="{y + LAYER_H/2}" r="34" fill="{stroke}" stroke="{stroke}" stroke-width="2"/>')
    lines.append(f'  <text x="{badge_cx}" y="{y + LAYER_H/2 + 9}" text-anchor="middle" font-size="22" font-weight="700" fill="#ffffff">{lid}</text>')

    # Layer name + sublabel (right of badge)
    text_x = TOWER_X + 110
    lines.append(f'  <text x="{text_x}" y="{y + 38}" font-size="17" font-weight="700" fill="{text_color}">{name}</text>')
    lines.append(f'  <text x="{text_x}" y="{y + 60}" font-size="13" fill="{TEXT_PRI}">{sublabel}</text>')
    lines.append(f'  <text x="{text_x}" y="{y + 82}" font-size="11" font-style="italic" fill="{TEXT_SEC}">{fall_label}</text>')

    # Cumulative reliability gauge on right
    gauge_y = y + LAYER_H/2 - 22
    lines.append(f'  <rect x="{GAUGE_X}" y="{gauge_y}" width="{GAUGE_W}" height="44" rx="8" ry="8" fill="#ffffff" stroke="{stroke}" stroke-width="1.5"/>')
    lines.append(f'  <text x="{GAUGE_X + GAUGE_W/2}" y="{gauge_y + 18}" text-anchor="middle" font-size="10" font-weight="600" fill="{TEXT_SEC}">CUMULATIVE</text>')
    lines.append(f'  <text x="{GAUGE_X + GAUGE_W/2}" y="{gauge_y + 36}" text-anchor="middle" font-size="18" font-weight="700" fill="{text_color}">{cum_pct}</text>')

# Connector arrows between layers (showing fall-through flow downward)
for i in range(len(LAYERS) - 1):
    y_top = LAYER_Y0 + i * (LAYER_H + LAYER_GAP) + LAYER_H
    y_bot = LAYER_Y0 + (i + 1) * (LAYER_H + LAYER_GAP)
    # Vertical arrow at center of tower
    arrow_x = TOWER_X + TOWER_W/2
    lines.append(f'  <line x1="{arrow_x}" y1="{y_top + 2}" x2="{arrow_x}" y2="{y_bot - 4}" stroke="#1f2937" stroke-width="1.6" marker-end="url(#arr-down)"/>')
    # Mid-arrow label "if fail"
    mid_y = (y_top + y_bot) / 2
    lines.append(f'  <rect x="{arrow_x - 30}" y="{mid_y - 8}" width="60" height="14" fill="{BG}" opacity="0.95"/>')
    lines.append(f'  <text x="{arrow_x}" y="{mid_y + 3}" text-anchor="middle" font-size="11" font-style="italic" fill="{TEXT_SEC}">if fail</text>')


# Bottom annotation: "DLQ + alert" terminal
y_bottom = LAYER_Y0 + 5 * (LAYER_H + LAYER_GAP) + 10
arrow_x = TOWER_X + TOWER_W/2
lines.append(f'  <line x1="{arrow_x}" y1="{y_bottom - 8}" x2="{arrow_x}" y2="{y_bottom + 18}" stroke="#7f1d1d" stroke-width="1.6" stroke-dasharray="5,3" marker-end="url(#arr-down)"/>')
lines.append(f'  <rect x="{arrow_x - 90}" y="{y_bottom + 22}" width="180" height="32" rx="6" ry="6" fill="#fef2f2" stroke="#fecaca" stroke-width="1.5"/>')
lines.append(f'  <text x="{arrow_x}" y="{y_bottom + 42}" text-anchor="middle" font-size="13" font-weight="600" fill="#7f1d1d">DLQ + alert (failure escapes)</text>')


# Left side: defense-direction annotation
arrow_top_y = LAYER_Y0
arrow_bot_y = y_bottom + 10
lines.append(f'  <line x1="160" y1="{arrow_top_y}" x2="160" y2="{arrow_bot_y}" stroke="{TEXT_SEC}" stroke-width="1.5" stroke-dasharray="4,3"/>')
lines.append(f'  <text x="155" y="{(arrow_top_y + arrow_bot_y)/2 - 14}" text-anchor="end" font-size="11" font-style="italic" fill="{TEXT_SEC}">cheaper</text>')
lines.append(f'  <text x="155" y="{(arrow_top_y + arrow_bot_y)/2}" text-anchor="end" font-size="11" font-style="italic" fill="{TEXT_SEC}">first line</text>')
lines.append(f'  <text x="155" y="{(arrow_top_y + arrow_bot_y)/2 + 18}" text-anchor="end" font-size="11" font-style="italic" fill="{TEXT_SEC}">↓</text>')
lines.append(f'  <text x="155" y="{(arrow_top_y + arrow_bot_y)/2 + 36}" text-anchor="end" font-size="11" font-style="italic" fill="{TEXT_SEC}">last resort</text>')


lines.append('</svg>')

OUT_SVG.parent.mkdir(parents=True, exist_ok=True)
OUT_SVG.write_text("\n".join(lines))
print(f"SVG written: {OUT_SVG}")
print(f"  size: {OUT_SVG.stat().st_size} bytes / {len(lines)} lines")
