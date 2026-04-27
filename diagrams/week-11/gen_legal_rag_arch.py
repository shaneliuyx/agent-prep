"""Week 11 — Legal RAG reference architecture (Ingestion + Query + Eval)."""
from pathlib import Path

OUT_SVG = Path("/Users/yuxinliu/code/agent-prep/diagrams/week-11/legal-rag-arch.svg")
BG, TEXT_PRI, TEXT_SEC = "#ffffff", "#111827", "#6b7280"
CTRL = "#1f2937"
INGEST_BG, INGEST_S = "#eff6ff", "#bfdbfe"
QUERY_BG,  QUERY_S  = "#f0fdf4", "#bbf7d0"
EVAL_BG,   EVAL_S   = "#faf5ff", "#ddd6fe"
REFUSE_BG, REFUSE_S, REFUSE_TXT = "#fef2f2", "#fecaca", "#7f1d1d"
AUDIT_BG, AUDIT_S, AUDIT_TXT = "#1a5276", "#1a5276", "#ffffff"
FONT = "'Helvetica Neue', Helvetica, Arial, sans-serif"
W, H = 1280, 1100

lines = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
         f'  <style>text {{ font-family: {FONT}; }}</style>',
         '  <defs>',
         f'    <marker id="a" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="{CTRL}"/></marker>',
         f'    <marker id="ar" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="{REFUSE_TXT}"/></marker>',
         '    <filter id="ds" x="-10%" y="-10%" width="120%" height="120%"><feDropShadow dx="0" dy="1" stdDeviation="1.5" flood-color="#000000" flood-opacity="0.07"/></filter>',
         '  </defs>',
         f'  <rect width="{W}" height="{H}" fill="{BG}"/>',
         f'  <text x="40" y="40" font-size="20" font-weight="700" fill="{TEXT_PRI}">Legal RAG Reference Architecture</text>',
         f'  <text x="40" y="62" font-size="13" fill="{TEXT_SEC}">Three-stage system: offline ingestion (left) → online query path (center) → audit + eval (right). ACL filtering at query time enforces tenant isolation.</text>']

def box(x, y, w, h, label, sub=None, fill="#fff", stroke="#cbd5e1", txt=TEXT_PRI):
    lines.append(f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" ry="8" fill="{fill}" stroke="{stroke}" stroke-width="1.5" filter="url(#ds)"/>')
    cx = x + w/2
    if sub:
        lines.append(f'  <text x="{cx}" y="{y + h/2 - 2}" text-anchor="middle" font-size="13" font-weight="600" fill="{txt}">{label}</text>')
        lines.append(f'  <text x="{cx}" y="{y + h/2 + 14}" text-anchor="middle" font-size="11" fill="{TEXT_SEC}">{sub}</text>')
    else:
        lines.append(f'  <text x="{cx}" y="{y + h/2 + 5}" text-anchor="middle" font-size="13" font-weight="600" fill="{txt}">{label}</text>')

def diamond(x, y, w, h, label, sub=None):
    cx = x + w/2; cy = y + h/2
    lines.append(f'  <polygon points="{cx},{y} {x+w},{cy} {cx},{y+h} {x},{cy}" fill="#fff" stroke="{CTRL}" stroke-width="1.5"/>')
    lines.append(f'  <text x="{cx}" y="{cy - 2}" text-anchor="middle" font-size="11" font-weight="600" fill="{TEXT_PRI}">{label}</text>')
    if sub:
        lines.append(f'  <text x="{cx}" y="{cy + 13}" text-anchor="middle" font-size="10" fill="{TEXT_SEC}">{sub}</text>')

def arr(x1, y1, x2, y2, label=None, marker="a"):
    color = REFUSE_TXT if marker == "ar" else CTRL
    lines.append(f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1.6" marker-end="url(#{marker})"/>')
    if label:
        lx, ly = (x1+x2)/2, (y1+y2)/2
        tw = len(label)*6 + 8
        lines.append(f'  <rect x="{lx - tw/2}" y="{ly - 8}" width="{tw}" height="14" fill="{BG}" opacity="0.95"/>')
        lines.append(f'  <text x="{lx}" y="{ly + 3}" text-anchor="middle" font-size="10" font-style="italic" fill="{TEXT_SEC}">{label}</text>')

# Ingestion subgraph (left)
IX, IY, IW, IH = 60, 100, 360, 700
lines.append(f'  <rect x="{IX}" y="{IY}" width="{IW}" height="{IH}" rx="14" ry="14" fill="{INGEST_BG}" stroke="{INGEST_S}" stroke-width="2"/>')
lines.append(f'  <text x="{IX + 18}" y="{IY + 26}" font-size="14" font-weight="700" fill="#1e3a8a">Ingestion (offline)</text>')
ing_nodes = [
    ("SharePoint / DMS / Email", None,                     IY + 60),
    ("Format normalizer",        "PDF · DOCX · MSG → text", IY + 140),
    ("Metadata extractor",       "author · date · ACL · doc-type", IY + 230),
    ("Semantic chunker",         "+ overlap",               IY + 320),
    ("Embedding model",          "BGE-M3 or Nomic",         IY + 410),
    ("Vector store",             "namespace per ACL group", IY + 500),
]
for i, (name, sub, y) in enumerate(ing_nodes):
    box(IX + 30, y, 300, 70, name, sub, stroke=INGEST_S)
    if i > 0:
        arr(IX + 180, ing_nodes[i-1][2] + 70, IX + 180, y - 2)

# Audit log (Ingestion side)
box(IX + 30, IY + 605, 300, 60, "Audit log (ingest events)", None, fill=AUDIT_BG, stroke=AUDIT_S, txt="#ffffff")
arr(IX + 180, IY + 230 + 70, IX + 30, IY + 635, "metadata events")

# Query subgraph (center)
QX, QY, QW, QH = 460, 100, 420, 850
lines.append(f'  <rect x="{QX}" y="{QY}" width="{QW}" height="{QH}" rx="14" ry="14" fill="{QUERY_BG}" stroke="{QUERY_S}" stroke-width="2"/>')
lines.append(f'  <text x="{QX + 18}" y="{QY + 26}" font-size="14" font-weight="700" fill="#14532d">Query Path (online)</text>')
q_nodes = [
    ("Lawyer query",        None,                         QY + 60),
    ("Query classifier",    "factual / comparative / multi-hop", QY + 145),
    ("Query rewriter",      "+ legal vocab expansion",    QY + 230),
    ("ACL filter",          "user role → allowed namespaces", QY + 315),
    ("Retrieval",           "top-k with cosine threshold", QY + 400),
    ("Cross-encoder reranker", None,                       QY + 485),
]
for i, (name, sub, y) in enumerate(q_nodes):
    box(QX + 50, y, 320, 70, name, sub, stroke=QUERY_S)
    if i > 0:
        arr(QX + 210, q_nodes[i-1][2] + 70, QX + 210, y - 2)

# Confidence diamond
diamond(QX + 80, QY + 575, 260, 70, "Confidence ≥ threshold?")
arr(QX + 210, QY + 485 + 70, QX + 210, QY + 575)

# Refusal (left of Confidence)
box(QX - 60, QY + 580, 130, 60, "No relevant docs", None, fill=REFUSE_BG, stroke=REFUSE_S, txt=REFUSE_TXT)
arr(QX + 80, QY + 610, QX + 5, QY + 610, "No", marker="ar")

# Yes path: Context assembler → Generation → Output validator → Response
box(QX + 50, QY + 670, 320, 60, "Context assembler", "chunk + [SOURCE:id] binding", stroke=QUERY_S)
arr(QX + 210, QY + 645, QX + 210, QY + 670, "Yes")
box(QX + 50, QY + 745, 320, 60, "Generation", "(cite SOURCE ids only)", stroke=QUERY_S)
arr(QX + 210, QY + 730, QX + 210, QY + 745)
box(QX + 50, QY + 815, 320, 60, "Output validator + Response", "citation presence check", stroke=QUERY_S, fill="#dcfce7")
arr(QX + 210, QY + 805, QX + 210, QY + 815)

# Eval subgraph (right)
EX, EY, EW, EH = 920, 100, 320, 700
lines.append(f'  <rect x="{EX}" y="{EY}" width="{EW}" height="{EH}" rx="14" ry="14" fill="{EVAL_BG}" stroke="{EVAL_S}" stroke-width="2"/>')
lines.append(f'  <text x="{EX + 18}" y="{EY + 26}" font-size="14" font-weight="700" fill="#5b21b6">Eval + Observability</text>')

# Audit log Q (top of eval)
box(EX + 25, EY + 50, 270, 60, "Audit log", "query · chunks · answer · user · ts", fill=AUDIT_BG, stroke=AUDIT_S, txt="#ffffff")

eval_nodes = [
    ("Faithfulness eval",    "(LLM-as-judge)",                  EY + 160),
    ("Citation accuracy",    "exact-match string search",       EY + 260),
    ("Retrieval hit-rate",   "monitoring dashboard",            EY + 360),
]
for name, sub, y in eval_nodes:
    box(EX + 25, y, 270, 80, name, sub, stroke=EVAL_S)
    arr(EX + 160, EY + 110, EX + 160, y - 2)

# Audit out from query
arr(QX + QW, QY + 845, EX, EY + 80, "audit")

lines.append('</svg>')
OUT_SVG.parent.mkdir(parents=True, exist_ok=True)
OUT_SVG.write_text("\n".join(lines))
print(f"OK: {OUT_SVG.stat().st_size} bytes")
