# lab-07-8-code-agent — W7.8

Companion lab for [[Week 7.8 - Code-Agent Patterns AST Coverage Mocks]].

## What this lab builds

A code-agent skill cluster:

1. **AST walk** — extract testable functions via Python stdlib `ast` + tree-sitter
2. **LSP queries** — references / type signatures via `multilspy` + pyright
3. **Coverage loop** — `coverage.py --branch` + LLM-guided edge-case test generation
4. **Mock helpers** — signature-validated `MagicMock` injection

## The 4-class testability filter

Functions get one of 5 classifications:

- `pure_stateless` → auto-generate tests
- `mock_required` → I/O present, generate with DI/patch
- `fixture_required` → framework decorators present, emit fixture sketch
- `escalate` → dynamic dispatch (`getattr` / `eval`), human review
- `property_test` → concurrency primitives, generate hypothesis property test

## Run

```bash
uv sync

# Phase 1 — extract function info from a file
uv run python -c "from src.ast_walk import extract; import pathlib; print(extract(pathlib.Path('src/ast_walk.py').read_text()))"

# Phase 3 — coverage measurement
uv run coverage run --branch -m pytest tests/
uv run coverage json -o coverage.json
uv run python -m src.coverage_loop coverage.json

# Tests
uv run pytest tests/ -v
```
