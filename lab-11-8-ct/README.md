# lab-11-8-ct — W11.8

Companion lab for [[Week 11.8 - Continuous Training and MLOps Pipelines]].

## What this lab ships

The CT loop: drift detector + eval gate + shadow ramp.

## Phases

| Phase | Script | What it does |
|---|---|---|
| 1 | `src/drift.py` | PSI per feature; `should_trigger_retrain` 3-day rule |
| 2 | `src/compare.py` | PR-gate: candidate metric >= main - epsilon |
| 3 | `src/router.py` | 5% traffic to candidate; rollback on health failure |
| 4 | `src/ct_loop.py` | Glue: cron → drift → trigger → eval → shadow → ramp |

## Run

```bash
uv sync

# Phase 1 — drift detector
uv run python -c "from src.drift import detect_drift; print(detect_drift('data/spans.parquet', feature='tokens_in'))"

# Phase 2 — eval gate comparator
uv run python -m src.compare \
    --candidate results/candidate.json \
    --baseline results/main_baseline.json \
    --tolerance 0.02 \
    --primary-metric ragas_faithfulness

uv run pytest tests/ -v
```
