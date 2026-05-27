# agent_loop_tools

Shared utility module for iterative agent loops across `agent-prep/lab-*` repos.

## Source attribution

Patterns ported from **[kunchenguid/gnhf](https://github.com/kunchenguid/gnhf)** (1.8K stars, MIT-licensed, TypeScript). gnhf is a production-grade autonomous agent overnight orchestrator (ralph/autoresearch-style commit-per-iteration with rollback). Its `src/core/` modules cleanly separate concerns with paired per-module tests; we port the smallest + highest-leverage primitives to Python here.

| Module | Lifted from | Notes |
|---|---|---|
| `interrupt_state.py` | `src/core/interrupt-state.ts` (33 LOC) | Pure-function state machine. Near-line-for-line port. |
| `token_accounting.py` | `src/core/orchestrator.ts` token fields + `tokensEstimated` sticky-flag pattern | Distilled from gnhf's `OrchestratorState` interface into a small dataclass. |

Both modules preserve gnhf's design choices:
- **Pure functions over methods with side effects** — easier to test, easier to reason about
- **Sticky flags over unset/reset** — once a run reports estimated tokens, the WHOLE run's totals are honest about being estimates (no silent re-honest)
- **Frozen dataclasses for snapshot types** — prevents accidental mutation of state passed between modules

## When to use which module

**`interrupt_state`** — any lab with a multi-iteration loop you want to interrupt cleanly. Examples:
- `lab-03-5-8-two-tier`'s `consolidate()` batch job: Ctrl-C should finish the current scroll then exit
- Eval drivers: Ctrl-C should let the current question complete then write partial results
- Phase 9 round-trip tests: Ctrl-C should drain in-flight Qdrant POSTs before exit

**`token_accounting`** — any lab calling LLMs across multiple iterations. Examples:
- §8.7 audit-wire-in dedup loop: accumulate per-scroll token cost
- §5.3 LongMemEval batch: per-question token cost + judge model cost
- Any consolidate run: emit "~12345 in / ~6789 out (estimated)" in RESULTS.md

## Usage

```python
from agent_loop_tools import (
    InterruptStateSnapshot,
    get_interrupt_disposition,
    get_interrupt_hint,
    TokenAccounting,
    UsageReport,
)

# Token accounting in an iteration loop
accountant = TokenAccounting()
for iter_n in range(max_iters):
    response = call_llm(...)
    accountant.add(UsageReport(
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
        estimated=False,                  # exact from OpenAI client
    ))
print(accountant.summary_line())   # "12345 in / 6789 out"

# Interrupt-state machine for Ctrl-C handling
state = InterruptStateSnapshot(status="running", graceful_stop_requested=False)
disposition = get_interrupt_disposition(state)   # "request-graceful-stop"
hint = get_interrupt_hint(state)                 # "resume"
```

## Tests

Run from any lab repo or from `agent-prep/`:

```bash
cd /Users/yuxinliu/code/agent-prep
uv run pytest shared/agent_loop_tools/tests/ -v
```

## License

This package contains derivative work from `kunchenguid/gnhf` (MIT). Original MIT notice preserved in source-file docstrings. Use under MIT terms.
