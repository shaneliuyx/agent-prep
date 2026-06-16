# lab-05-pattern-zoo

Compare agent **control-flow patterns** on one canonical, tool-composable task,
measured against the local **oMLX** model (zero cloud spend). Each pattern is a
self-contained impl exposing the same entry point so the comparison is
apples-to-apples.

Patterns:
- `src/impl_react.py` — **ReAct**: one structured tool call per LLM turn; observation re-fed each step.
- `src/impl_plan_solve.py` — **Plan-and-Solve**: explicit plan, then an observation-driven solve loop.
- `src/impl_codeact.py` — **CodeAct**: the action is *Python code* executed in a restricted namespace; composes all tools in one action.
- `src/impl_rewoo.py` — **ReWOO**: Planner (1 LLM, `#E` placeholders) → Worker (no LLM) → Solver (1 LLM); observations never re-fed.

Shared substrate: `src/schema.py` (`AgentResult`, `CANONICAL_TASK`),
`src/tools.py` (deterministic `kb_lookup`/`add`/`multiply`, both as plain callables
and OpenAI schemas), `src/llm_client.py` (the one oMLX client).

## Run

```bash
cd lab-05-pattern-zoo
set -a; source .env; set +a          # OMLX_URL + MODEL_SONNET

# Full comparison + LLM-as-judge + matrix:
PYTHONPATH=. /Users/yuxinliu/code/agent-prep/.venv/bin/python \
  -c "import runpy; runpy.run_module('src.05_compare', run_name='__main__')"

# One pattern on its own (prints its AgentResult):
PYTHONPATH=. /Users/yuxinliu/code/agent-prep/.venv/bin/python -m src.impl_codeact
PYTHONPATH=. /Users/yuxinliu/code/agent-prep/.venv/bin/python -m src.impl_rewoo

# Offline unit tests (no LLM server needed):
PYTHONPATH=. /Users/yuxinliu/code/agent-prep/.venv/bin/python -m pytest tests/ -q
```

Measured numbers live in `RESULTS.md`. The `05_compare.py` module name starts with
a digit, so it must be run via `runpy`/`-m`, not `import`.
