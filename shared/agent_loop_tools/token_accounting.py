"""Token usage accumulator with sticky-estimated flag for honest reporting.

Source: ported from kunchenguid/gnhf src/core/orchestrator.ts (the
`OrchestratorState` token-tracking fields + `tokensEstimated` sticky-flag
pattern; MIT-licensed). gnhf accumulates per-iteration token usage across
an overnight agent run and surfaces an honest "estimated vs exact" flag
in the exit summary.

Original TS reference:
  https://github.com/kunchenguid/gnhf/blob/main/src/core/orchestrator.ts
  (look for `totalInputTokens`, `totalOutputTokens`, `tokensEstimated`)

Why the sticky flag matters: some agent adapters (e.g. ACP protocol clients
without usage telemetry) can't emit exact `usage_update` events. Their
per-iteration token counts are estimates derived from message bytes. If
ANY single iteration in a run reports estimated counts, the WHOLE run's
totals are estimates — but reporting them as "exact" would silently lie.
`tokensEstimated` is set once and never unset; the exit summary prefixes
totals with `~` whenever this is True.

Usage:

    from agent_loop_tools import TokenAccounting

    accountant = TokenAccounting()
    for iter_n in range(max_iters):
        usage = run_one_iteration(...)   # returns UsageReport
        accountant.add(usage)
    print(accountant.summary_line())     # "~12345 in / ~6789 out (estimated)"
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UsageReport:
    """One iteration's token usage. `estimated=True` means input_tokens
    and output_tokens are derived from message-size heuristics, not from
    an exact usage_update event from the adapter."""

    input_tokens: int = 0
    output_tokens: int = 0
    estimated: bool = False


@dataclass
class TokenAccounting:
    """Accumulates token usage across iterations. Sticky-estimated flag
    once set stays set; the run's totals are honest about whether any
    iteration emitted estimates.

    Not thread-safe — single-writer assumption (the orchestrator is the
    only caller). For parallel iterations, accumulate per-task then merge
    via __add__ at the end.
    """

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    tokens_estimated: bool = False
    iteration_count: int = 0

    def add(self, usage: UsageReport) -> None:
        """Apply one iteration's usage to the running totals.
        Once tokens_estimated is set True, it stays True for the rest
        of this accountant's lifetime (sticky flag)."""
        self.total_input_tokens += usage.input_tokens
        self.total_output_tokens += usage.output_tokens
        if usage.estimated:
            self.tokens_estimated = True   # sticky: never reset
        self.iteration_count += 1

    @property
    def total_tokens(self) -> int:
        """Sum of input + output across all iterations."""
        return self.total_input_tokens + self.total_output_tokens

    def summary_line(self) -> str:
        """One-line summary for exit reporting. `~` prefix marks estimated
        totals — same convention gnhf uses in its exit summary card."""
        prefix = "~" if self.tokens_estimated else ""
        suffix = " (estimated)" if self.tokens_estimated else ""
        return (
            f"{prefix}{self.total_input_tokens} in / "
            f"{prefix}{self.total_output_tokens} out"
            f"{suffix}"
        )

    def __add__(self, other: "TokenAccounting") -> "TokenAccounting":
        """Merge two accountants (e.g. parallel iteration aggregation)."""
        return TokenAccounting(
            total_input_tokens=self.total_input_tokens + other.total_input_tokens,
            total_output_tokens=self.total_output_tokens + other.total_output_tokens,
            tokens_estimated=self.tokens_estimated or other.tokens_estimated,
            iteration_count=self.iteration_count + other.iteration_count,
        )
