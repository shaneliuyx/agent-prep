"""scheduler.py — external triggers that fire runs. The "who starts a run" seam.

THE CENTRAL TEACHING POINT (read this before anything else):
There is NO always-on, self-prompting agent loop in this runtime. The runtime
does not wake itself up and decide to do work. EVERY run is fired by an explicit
EXTERNAL trigger — a cron tick, an inbound webhook, or a manual call. This is the
deliberate architectural choice that separates a *durable workflow runtime* (n8n,
Temporal, AutoGPT-Platform) from a runaway "agent that prompts itself forever":
work is demand-driven and auditable. Every trigger creates exactly one run via
graph_store.start_run, and that run is the unit of durability + cost + replay.

Implementation is intentionally minimal (no APScheduler, no FastAPI required):
- register_cron     stores the registration + an optional in-thread timer
- register_webhook  stores a path→graph mapping; handle_webhook fires a run
- trigger_manually  fully functional: starts a run right now and returns run_id
The timers/HTTP are thin shims; the load-bearing idea is that a *trigger* is just
"call start_run", and the runtime owns everything after that.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from graph_store import GraphStore


@dataclass
class CronRegistration:
    """A cron registration. `interval_s` is the resolved tick period; we store
    the raw expression for the audit trail. A real deployment would parse a true
    cron expression — here a simple period keeps the timer dependency-free."""

    graph_id: str
    cron_expr: str
    interval_s: float
    timer: threading.Timer | None = field(default=None, repr=False)


class Scheduler:
    """Registry of external triggers over a GraphStore. Holds no run state of its
    own — a trigger's only job is to call start_run; the GraphStore owns the run
    from there. That clean handoff is the point: triggers are stateless edges
    into a durable core."""

    def __init__(self, store: GraphStore) -> None:
        self.store = store
        self._crons: dict[str, CronRegistration] = {}
        self._webhooks: dict[str, str] = {}  # path -> graph_id

    # ── manual trigger (fully functional) ───────────────────────────────────
    def trigger_manually(self, graph_id: str, payload: dict[str, Any] | None = None
                         ) -> str:
        """Fire a run NOW from an explicit manual call. The canonical external
        trigger; cron and webhook both funnel into this same start_run call."""
        trigger = "manual" if not payload else f"manual:{payload}"
        return self.store.start_run(graph_id, trigger)

    # ── webhook trigger ─────────────────────────────────────────────────────
    def register_webhook(self, graph_id: str, path: str) -> None:
        """Map an inbound webhook `path` to a graph. No server is started here —
        a host app (FastAPI, etc.) routes the request to handle_webhook. Keeping
        the mapping here, not in a web framework, keeps the trigger logic
        framework-agnostic and unit-testable."""
        self._webhooks[path] = graph_id

    def handle_webhook(self, path: str, payload: dict[str, Any] | None = None) -> str:
        """Fire the run mapped to `path`. This is the function a FastAPI route (or
        a test fixture) calls on an inbound request. Raises if the path is
        unregistered — fail loud on an unknown external trigger."""
        if path not in self._webhooks:
            raise KeyError(f"no webhook registered for path: {path}")
        return self.store.start_run(self._webhooks[path], f"webhook:{path}")

    # ── cron trigger ────────────────────────────────────────────────────────
    def register_cron(self, graph_id: str, cron_expr: str,
                      interval_s: float | None = None) -> CronRegistration:
        """Register a periodic trigger. `interval_s` overrides the parsed period;
        if omitted we fall back to a 60s default (this lab does not ship a full
        cron parser — APScheduler is deliberately avoided). The registration is
        stored but NOT started until start_cron, so tests can register without
        spawning a live timer."""
        reg = CronRegistration(graph_id, cron_expr, interval_s or 60.0)
        self._crons[graph_id] = reg
        return reg

    def start_cron(self, graph_id: str) -> None:
        """Arm the in-thread timer for a registered cron. Each tick fires one run
        and re-arms — a self-rescheduling threading.Timer, no daemon process."""
        reg = self._crons[graph_id]

        def _tick() -> None:
            self.store.start_run(graph_id, f"cron:{reg.cron_expr}")
            reg.timer = threading.Timer(reg.interval_s, _tick)
            reg.timer.daemon = True
            reg.timer.start()

        reg.timer = threading.Timer(reg.interval_s, _tick)
        reg.timer.daemon = True
        reg.timer.start()

    def stop_cron(self, graph_id: str) -> None:
        """Cancel a running cron timer. Idempotent."""
        reg = self._crons.get(graph_id)
        if reg and reg.timer is not None:
            reg.timer.cancel()
            reg.timer = None
