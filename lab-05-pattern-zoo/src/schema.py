"""
src/schema.py — shared input/output schema for every pattern in the zoo.

Every pattern impl (ReAct, Plan-and-Solve, CodeAct, ReWOO, ...) exposes the same
entry point:

    run(task: str) -> AgentResult

so 05_compare.py can drive them identically and slot their numbers into one
matrix. AgentResult is the *measured* envelope: the final answer plus the four
columns the comparison matrix tracks (LLM calls, total tokens, wall-clock).

Immutable by construction (frozen dataclass): a result is a record of a run, not
a mutable accumulator. Patterns build their own counters internally and emit one
AgentResult at the end.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AgentResult:
    """The measured outcome of one pattern running one task.

    Fields map 1:1 to the comparison-matrix columns:
      answer            : the agent's final text answer (judged for correctness)
      llm_calls         : number of round-trips to the model
      prompt_tokens     : summed real response.usage.prompt_tokens
      completion_tokens : summed real response.usage.completion_tokens
      latency_ms        : wall-clock for the whole run
      pattern           : pattern name (for the matrix row label)
      trace             : optional human-readable step log (debugging only)
    """

    pattern: str
    answer: str
    llm_calls: int
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    trace: tuple[str, ...] = field(default_factory=tuple)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


# The single canonical task every pattern runs, so the matrix is apples-to-apples.
#
# It is deliberately multi-step AND tool-composable:
#   - look up two numeric facts from the knowledge_base (population of two cities)
#   - SUM them, then MULTIPLY by a looked-up growth factor
# ReAct/Plan-Solve must round-trip the LLM once per tool call (observation fed
# back each time). CodeAct can do the whole arithmetic chain in ONE code action.
# ReWOO plans all the tool calls up front and never re-feeds observations.
CANONICAL_TASK: str = (
    "Using the tools, find the population of 'Springfield' and the population of "
    "'Shelbyville' from the knowledge base, add them together, then multiply that "
    "sum by the 'growth_factor' value from the knowledge base. Report the final "
    "number."
)

# Ground truth, computed from the fixed knowledge_base in src/tools.py:
#   Springfield = 30000, Shelbyville = 12000, growth_factor = 3
#   (30000 + 12000) * 3 = 126000
CANONICAL_EXPECTED_ANSWER: str = "126000"
