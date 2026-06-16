"""graph_store.py — SQLite persistence for durable agent graphs.

The single most important module in the lab: it holds the *execution state*
that is deliberately separated from the LLM loop. A process can die anywhere
(kill -9, OOM, Ctrl-C, panic) and the run survives, because the run lives in
four SQLite tables — not in Python locals:

    graphs      — DAG *templates* (node defs + edges), authored once
    runs        — one row per trigger fire (an *instance* of a graph)
    nodes       — per-(run, node) state machine: pending→ready→running→done/failed
    executions  — append-only event log (event-sourcing); the audit + replay trail

Durability invariant (BCJ Entry 3): every status mutation and its event row are
written in the SAME transaction. There is never a node marked `done` whose
`done` event failed to append — the two commit together or not at all.

Resume protocol: on restart, `recover_run` resets any node left `running`
(claimed by a now-dead worker) back to `ready`. The READY frontier is then
re-derived from the table — `nodes WHERE status='ready'` — never from memory.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from typing import Any

from file_lock import FileLock

# Node lifecycle states. The state machine is intentionally tiny — four live
# states plus two terminal — because every extra state is an extra recovery case.
PENDING = "pending"   # waiting on upstream deps
READY = "ready"       # all deps done; eligible to be claimed by a worker
RUNNING = "running"   # claimed by a worker; the only state needing recovery
DONE = "done"         # terminal success
FAILED = "failed"     # terminal failure (attempts exhausted)


@dataclass(frozen=True)
class Node:
    """A claimed unit of work handed to a worker. Immutable snapshot — the
    worker reads it, runs the handler, then calls back with a result."""

    run_id: str
    name: str
    node_type: str          # "llm" | "tool" | "branch"
    payload: dict[str, Any]
    attempts: int


class GraphStore:
    """Durable graph persistence over a single SQLite file.

    Thread-safety by construction: every method opens its own short-lived
    connection. SQLite in WAL mode supports concurrent readers + one writer,
    and the cross-process *claim* is serialized by a sibling `FileLock`, not by
    a shared in-process connection (which would not survive process death)."""

    def __init__(self, db_path: str, lock_path: str | None = None) -> None:
        self.db_path = db_path
        self.lock = FileLock(lock_path or f"{db_path}.claim.lock")
        self._init_schema()

    # ── connection + schema ────────────────────────────────────────────────
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")      # concurrent reads + durable writes
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")    # WAL-safe; survives app crash
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS graphs (
                    graph_id   TEXT PRIMARY KEY,
                    name       TEXT NOT NULL,
                    dag_json   TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs (
                    run_id      TEXT PRIMARY KEY,
                    graph_id    TEXT NOT NULL REFERENCES graphs(graph_id),
                    trigger     TEXT NOT NULL,
                    status      TEXT NOT NULL,
                    started_at  REAL NOT NULL,
                    finished_at REAL
                );
                CREATE TABLE IF NOT EXISTS nodes (
                    run_id      TEXT NOT NULL REFERENCES runs(run_id),
                    name        TEXT NOT NULL,
                    node_type   TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    deps_json   TEXT NOT NULL,
                    status      TEXT NOT NULL,
                    result_json TEXT,
                    attempts    INTEGER NOT NULL DEFAULT 0,
                    claimed_by  TEXT,
                    updated_at  REAL NOT NULL,
                    PRIMARY KEY (run_id, name)
                );
                CREATE TABLE IF NOT EXISTS executions (
                    event_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id      TEXT NOT NULL,
                    node_name   TEXT,
                    event_type  TEXT NOT NULL,
                    payload_json TEXT,
                    ts          REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_nodes_ready
                    ON nodes(run_id, status);
                CREATE INDEX IF NOT EXISTS idx_exec_run
                    ON executions(run_id, event_id);
                """
            )

    # ── append-only event log ───────────────────────────────────────────────
    @staticmethod
    def _append(conn: sqlite3.Connection, run_id: str, node: str | None,
                event_type: str, payload: dict[str, Any] | None = None) -> None:
        """Append one event. Always called inside the caller's transaction so
        the event commits atomically with the state change it records."""
        conn.execute(
            "INSERT INTO executions(run_id, node_name, event_type, payload_json, ts)"
            " VALUES (?,?,?,?,?)",
            (run_id, node, event_type, json.dumps(payload or {}), time.time()),
        )

    # ── authoring: graph template ────────────────────────────────────────────
    def create_graph(self, name: str, dag: dict[str, Any]) -> str:
        """Persist a DAG template. `dag` = {"nodes": {name: {type, payload}},
        "edges": [[from, to], ...]}. Returns the graph_id."""
        graph_id = f"g_{uuid.uuid4().hex[:12]}"
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO graphs(graph_id, name, dag_json, created_at) VALUES (?,?,?,?)",
                (graph_id, name, json.dumps(dag), time.time()),
            )
        return graph_id

    # ── instantiate: a run ───────────────────────────────────────────────────
    def start_run(self, graph_id: str, trigger: str) -> str:
        """Materialize a graph into a fresh run: one `nodes` row per template
        node, with deps derived from edges. Nodes with zero deps start READY;
        the rest start PENDING. Returns the run_id."""
        run_id = f"r_{uuid.uuid4().hex[:12]}"
        with self._connect() as conn:
            row = conn.execute(
                "SELECT dag_json FROM graphs WHERE graph_id=?", (graph_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"no such graph: {graph_id}")
            dag = json.loads(row["dag_json"])
            deps: dict[str, list[str]] = {n: [] for n in dag["nodes"]}
            for src, dst in dag.get("edges", []):
                deps[dst].append(src)

            now = time.time()
            conn.execute(
                "INSERT INTO runs(run_id, graph_id, trigger, status, started_at)"
                " VALUES (?,?,?,?,?)",
                (run_id, graph_id, trigger, "running", now),
            )
            for nm, spec in dag["nodes"].items():
                status = READY if not deps[nm] else PENDING
                conn.execute(
                    "INSERT INTO nodes(run_id, name, node_type, payload_json,"
                    " deps_json, status, attempts, updated_at) VALUES (?,?,?,?,?,?,?,?)",
                    (run_id, nm, spec.get("type", "tool"),
                     json.dumps(spec.get("payload", {})),
                     json.dumps(deps[nm]), status, 0, now),
                )
            self._append(conn, run_id, None, "run_started",
                         {"graph_id": graph_id, "trigger": trigger})
        return run_id

    # ── the claim: atomic READY → RUNNING under a cross-process lock ─────────
    def claim_ready_node(self, run_id: str, worker_id: str) -> Node | None:
        """Atomically claim one READY node for `worker_id`. The file lock makes
        this safe across processes (the fix for "two workers, same node"): only
        the lock holder may read-then-write the status, so no two workers can
        both transition the same node READY→RUNNING."""
        with self.lock:                              # cross-process mutual exclusion
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT name, node_type, payload_json, attempts FROM nodes"
                    " WHERE run_id=? AND status=? ORDER BY name LIMIT 1",
                    (run_id, READY),
                ).fetchone()
                if row is None:
                    return None
                conn.execute(
                    "UPDATE nodes SET status=?, claimed_by=?, attempts=attempts+1,"
                    " updated_at=? WHERE run_id=? AND name=?",
                    (RUNNING, worker_id, time.time(), run_id, row["name"]),
                )
                self._append(conn, run_id, row["name"], "claimed",
                             {"worker": worker_id, "attempt": row["attempts"] + 1})
                return Node(run_id, row["name"], row["node_type"],
                            json.loads(row["payload_json"]), row["attempts"] + 1)

    # ── completion: DONE + unblock downstream, one transaction ──────────────
    def mark_done(self, run_id: str, name: str, result: dict[str, Any]) -> list[str]:
        """Mark a node DONE, append its event, and promote any downstream node
        whose deps are now all satisfied PENDING→READY — all atomically.
        Returns the names newly promoted to READY."""
        promoted: list[str] = []
        with self._connect() as conn:
            conn.execute(
                "UPDATE nodes SET status=?, result_json=?, updated_at=?"
                " WHERE run_id=? AND name=?",
                (DONE, json.dumps(result), time.time(), run_id, name),
            )
            self._append(conn, run_id, name, "done",
                         {"tokens": result.get("tokens"), "ms": result.get("ms")})
            promoted = self._promote_downstream(conn, run_id)
            self._maybe_finish_run(conn, run_id)
        return promoted

    def mark_failed(self, run_id: str, name: str, error: str,
                    max_attempts: int = 3) -> str:
        """Fail a node. If attempts remain, requeue it READY (retry); otherwise
        mark FAILED terminally. Returns the resulting status. The retry counter
        lives in the `nodes` row, so it SURVIVES restart — the canonical fix for
        the classic-AutoGPT retry-storm (counter-resets-on-restart) failure."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT attempts FROM nodes WHERE run_id=? AND name=?",
                (run_id, name),
            ).fetchone()
            if row["attempts"] >= max_attempts:
                conn.execute(
                    "UPDATE nodes SET status=?, updated_at=? WHERE run_id=? AND name=?",
                    (FAILED, time.time(), run_id, name),
                )
                self._append(conn, run_id, name, "failed",
                             {"error": error[:200], "attempts": row["attempts"]})
                self._maybe_finish_run(conn, run_id)
                return FAILED
            conn.execute(
                "UPDATE nodes SET status=?, claimed_by=NULL, updated_at=?"
                " WHERE run_id=? AND name=?",
                (READY, time.time(), run_id, name),
            )
            self._append(conn, run_id, name, "retry",
                         {"error": error[:200], "attempt": row["attempts"]})
            return READY

    # ── recovery: the kill -9 → resume primitive ────────────────────────────
    def recover_run(self, run_id: str) -> list[str]:
        """Reset orphaned RUNNING nodes (claimed by a dead worker) back to READY.
        Called once on restart before any worker starts. This is the entire
        'resume from last persisted node' mechanism — everything else already
        lives in the tables. Returns the names recovered."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT name FROM nodes WHERE run_id=? AND status=?",
                (run_id, RUNNING),
            ).fetchall()
            names = [r["name"] for r in rows]
            for nm in names:
                conn.execute(
                    "UPDATE nodes SET status=?, claimed_by=NULL, updated_at=?"
                    " WHERE run_id=? AND name=?",
                    (READY, time.time(), run_id, nm),
                )
                self._append(conn, run_id, nm, "recovered", {"from": RUNNING})
            if names:
                conn.execute("UPDATE runs SET status=? WHERE run_id=?",
                             ("running", run_id))
        return names

    # ── internal helpers ─────────────────────────────────────────────────────
    def _promote_downstream(self, conn: sqlite3.Connection, run_id: str) -> list[str]:
        """Promote PENDING nodes whose every dep is DONE to READY."""
        done = {r["name"] for r in conn.execute(
            "SELECT name FROM nodes WHERE run_id=? AND status=?",
            (run_id, DONE)).fetchall()}
        promoted: list[str] = []
        for r in conn.execute(
            "SELECT name, deps_json FROM nodes WHERE run_id=? AND status=?",
            (run_id, PENDING)).fetchall():
            deps = json.loads(r["deps_json"])
            if all(d in done for d in deps):
                conn.execute(
                    "UPDATE nodes SET status=?, updated_at=? WHERE run_id=? AND name=?",
                    (READY, time.time(), run_id, r["name"]))
                self._append(conn, run_id, r["name"], "ready", {})
                promoted.append(r["name"])
        return promoted

    def _maybe_finish_run(self, conn: sqlite3.Connection, run_id: str) -> None:
        """Close the run when no node is still live (pending/ready/running)."""
        live = conn.execute(
            "SELECT COUNT(*) c FROM nodes WHERE run_id=? AND status IN (?,?,?)",
            (run_id, PENDING, READY, RUNNING)).fetchone()["c"]
        if live:
            return
        failed = conn.execute(
            "SELECT COUNT(*) c FROM nodes WHERE run_id=? AND status=?",
            (run_id, FAILED)).fetchone()["c"]
        final = "failed" if failed else "done"
        conn.execute("UPDATE runs SET status=?, finished_at=? WHERE run_id=?",
                     (final, time.time(), run_id))
        self._append(conn, run_id, None, "run_finished", {"status": final})

    # ── read side: status, replay ────────────────────────────────────────────
    def node_states(self, run_id: str) -> dict[str, str]:
        with self._connect() as conn:
            return {r["name"]: r["status"] for r in conn.execute(
                "SELECT name, status FROM nodes WHERE run_id=?", (run_id,)).fetchall()}

    def run_status(self, run_id: str) -> str:
        with self._connect() as conn:
            return conn.execute(
                "SELECT status FROM runs WHERE run_id=?", (run_id,)).fetchone()["status"]

    def replay_run(self, run_id: str) -> list[dict[str, Any]]:
        """Reconstruct the run by replaying its event log in append order.
        This is the event-sourcing payoff: the full causal history of the run
        is a query, available for forensics long after the process exited."""
        with self._connect() as conn:
            return [
                {"event_id": r["event_id"], "node": r["node_name"],
                 "type": r["event_type"], "payload": json.loads(r["payload_json"]),
                 "ts": r["ts"]}
                for r in conn.execute(
                    "SELECT * FROM executions WHERE run_id=? ORDER BY event_id",
                    (run_id,)).fetchall()
            ]
