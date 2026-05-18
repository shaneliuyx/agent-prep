# lab-11-6-tracing — W11.6

Companion lab for [[Week 11.6 - Production Tracing and Cost Telemetry]].

## What this lab measures

OpenTelemetry instrumentation of the W4 ReAct loop + Langfuse self-hosted UI + DuckDB cost rollups.

## Setup

```bash
uv sync

# Self-host Langfuse separately:
#   git clone https://github.com/langfuse/langfuse
#   cd langfuse && cp .env.dev.example .env && docker-compose up -d
# UI: http://localhost:3000

# .env (this lab):
#   OMLX_BASE_URL=http://127.0.0.1:8000/v1
#   OMLX_API_KEY=<your key>
#   OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
```

## Phases

| Phase | Script | What it measures |
|---|---|---|
| 1 | `src/tracing.py` | Instrument LLM call + tool call spans with cost attribution |
| 2 | `src/instrumented_react.py` | W4 ReAct loop with full span tree |
| 3 | `src/cost_rollup.py` | DuckDB queries — cost per role per day, p99 latency per model per hour |

## Run

```bash
# Phase 2 — run a sample ReAct task, send spans to Langfuse
uv run python -m src.instrumented_react "What's the weather in Tokyo?"

# Phase 3 — query cost rollups (after 24h of traffic)
uv run python -c "from src.cost_rollup import cost_per_role_per_day; print(cost_per_role_per_day().to_string())"
uv run python -c "from src.cost_rollup import p99_latency_per_model; print(p99_latency_per_model(hours=24).to_string())"

uv run pytest tests/ -v
```
