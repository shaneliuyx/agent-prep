# services/hypermem_shim.py — L3 HyperMem-compatible shim (lab-local)
"""A small, self-contained L3 relational store that speaks the API the lab's
ThreeTierMemory + consolidate_with_l3 expect.

WHY A SHIM, NOT THE PAPER REPO: the real HyperMem (EverOS/methods/HyperMem,
ACL-2026) is an OFFLINE eval pipeline (run_eval.sh, stages 1-6) over a
topic->episode->fact hypergraph built from dialogue. It has no HTTP server and a
different data model than this lab's L3 contract (typed entity hyperedges queried
by multi-entity intersection). Rather than force-fit the research code, this shim
implements EXACTLY the three endpoints the lab calls, backed by SQLite. It is the
L3 STORE, not the paper's algorithm — enough to make Phase 8 (consolidate_with_l3)
and query_relations() run end-to-end.

Endpoints (match src/three_tier_memory.py + src/consolidation.py):
  GET  /health                     -> {"status": "healthy"}
  POST /api/v1/edges               <- {nodes:[{type,id}...], relation, user_id?,
                                        provenance_scroll?, idempotency_key?}
  POST /api/v1/query/relations     <- {intersection:[{node:{type,id}}...],
                                        return_type, limit?, user_id?}
                                     -> {"results": [{type,id,relation,...}, ...]}

Run:  python -m services.hypermem_shim            (serves on :1996)
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

DB_PATH = Path(os.getenv("HYPERMEM_DB", str(Path(__file__).resolve().parent.parent / ".hypermem_l3.sqlite")))
PORT = int(os.getenv("HYPERMEM_PORT", "1996"))

app = FastAPI(title="HyperMem L3 shim", version="0.1.0")


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH)
    c.execute("""CREATE TABLE IF NOT EXISTS edges (
        edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
        idempotency_key TEXT UNIQUE,
        relation TEXT,
        user_id TEXT DEFAULT '',
        provenance_scroll TEXT DEFAULT ''
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS edge_nodes (
        edge_id INTEGER, ntype TEXT, nid TEXT,
        FOREIGN KEY(edge_id) REFERENCES edges(edge_id)
    )""")
    c.execute("CREATE INDEX IF NOT EXISTS ix_nodes ON edge_nodes(ntype, nid)")
    return c


# ── request models ───────────────────────────────────────────────────
class Node(BaseModel):
    type: str
    id: str


class EdgeIn(BaseModel):
    nodes: list[Node]
    relation: str = ""
    user_id: str = ""
    provenance_scroll: str = ""
    idempotency_key: str | None = None


class RelQuery(BaseModel):
    intersection: list[dict[str, Any]]   # [{"node": {"type","id"}}, ...]
    return_type: str
    limit: int = 10
    user_id: str = ""


# ── endpoints ────────────────────────────────────────────────────────
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy", "service": "hypermem-l3-shim"}


@app.post("/api/v1/edges")
def create_edge(edge: EdgeIn) -> dict[str, Any]:
    """Store one typed hyperedge. Idempotent on idempotency_key."""
    c = _conn()
    try:
        if edge.idempotency_key:
            row = c.execute("SELECT edge_id FROM edges WHERE idempotency_key = ?",
                            (edge.idempotency_key,)).fetchone()
            if row:
                return {"status": "exists", "edge_id": row[0]}
        cur = c.execute(
            "INSERT INTO edges (idempotency_key, relation, user_id, provenance_scroll) VALUES (?,?,?,?)",
            (edge.idempotency_key, edge.relation, edge.user_id, edge.provenance_scroll),
        )
        eid = cur.lastrowid
        c.executemany("INSERT INTO edge_nodes (edge_id, ntype, nid) VALUES (?,?,?)",
                      [(eid, n.type, n.id) for n in edge.nodes])
        c.commit()
        return {"status": "created", "edge_id": eid}
    finally:
        c.close()


@app.post("/api/v1/query/relations")
def query_relations(q: RelQuery) -> dict[str, list[dict[str, Any]]]:
    """Return distinct return_type nodes that co-occur in edges containing ALL
    of the intersection nodes (for this user)."""
    pins = [(item["node"]["type"], item["node"]["id"]) for item in q.intersection
            if item.get("node")]
    if not pins:
        return {"results": []}
    c = _conn()
    try:
        # edge_ids containing each pinned node (scoped to user_id), then intersect
        candidate: set[int] | None = None
        for ntype, nid in pins:
            rows = c.execute(
                """SELECT en.edge_id FROM edge_nodes en JOIN edges e ON e.edge_id = en.edge_id
                   WHERE en.ntype = ? AND en.nid = ? AND e.user_id = ?""",
                (ntype, nid, q.user_id),
            ).fetchall()
            ids = {r[0] for r in rows}
            candidate = ids if candidate is None else (candidate & ids)
            if not candidate:
                return {"results": []}
        if not candidate:
            return {"results": []}
        # collect return_type nodes from candidate edges, excluding the pins
        out: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set(pins)
        placeholders = ",".join("?" * len(candidate))
        rows = c.execute(
            f"""SELECT DISTINCT en.ntype, en.nid, e.relation, e.provenance_scroll
                FROM edge_nodes en JOIN edges e ON e.edge_id = en.edge_id
                WHERE en.edge_id IN ({placeholders}) AND en.ntype = ?""",
            (*candidate, q.return_type),
        ).fetchall()
        for ntype, nid, relation, prov in rows:
            if (ntype, nid) in seen:
                continue
            seen.add((ntype, nid))
            out.append({"type": ntype, "id": nid, "relation": relation,
                        "provenance_scroll": prov})
            if len(out) >= q.limit:
                break
        return {"results": out}
    finally:
        c.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
