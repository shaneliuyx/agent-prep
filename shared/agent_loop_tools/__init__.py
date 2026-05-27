"""agent_loop_tools — shared utility module for iterative agent loops.

Source attribution:
  Patterns lifted from kunchenguid/gnhf (https://github.com/kunchenguid/gnhf,
  1.8K stars, TypeScript). gnhf is a production-grade overnight agent
  orchestrator (ralph/autoresearch-style commit-per-iteration with rollback).
  Its `src/core/` modules separate concerns cleanly with paired tests; we
  port the smallest + highest-leverage primitives to Python for use across
  agent-prep lab repos.

See `shared/agent_loop_tools/README.md` for usage examples and the
per-module source-file attribution.
"""

from .interrupt_state import (
    InterruptDisposition,
    InterruptHint,
    InterruptStateSnapshot,
    get_interrupt_disposition,
    get_interrupt_hint,
)
from .token_accounting import (
    TokenAccounting,
    UsageReport,
)

__all__ = [
    "InterruptDisposition",
    "InterruptHint",
    "InterruptStateSnapshot",
    "get_interrupt_disposition",
    "get_interrupt_hint",
    "TokenAccounting",
    "UsageReport",
]
