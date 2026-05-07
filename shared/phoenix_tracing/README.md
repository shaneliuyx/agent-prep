# phoenix_tracing — one-call Phoenix observability for any RAG / agent lab

Distilled from W3 lab `lab-03-rag-eval/src/05_trace.py`. The W3 implementation
worked but every future lab that wants tracing had to re-derive the
`register() + OpenAIInstrumentor + LangChainInstrumentor + tracer.start_as_current_span`
ceremony. This shared library reduces that to one function call.

## Three APIs (most ergonomic first)

### 1. `trace_run(project_name, server_url, label, fn, ...)` — one-call

The user's preferred shape — inputs: project_name, server_url, label, function.
Setup + parent span + run + return — all in one call.

```python
from phoenix_tracing import trace_run

result = trace_run(
    project_name="lab-03-rag-eval",
    server_url="http://127.0.0.1:6006",
    label="baseline",
    fn=run_pipeline,
    question=q,                 # forwarded to fn
    attrs={"question_index": i, "variant": "baseline"},
)
```

Phoenix UI then shows a `baseline` root span with auto-instrumented OpenAI /
LangChain children nested under it. `result` is whatever `fn` returned.

### 2. `phoenix_span(label, attrs=None)` — context manager

For inline tracing of arbitrary code blocks. Requires `init_phoenix(...)` first.

```python
from phoenix_tracing import init_phoenix, phoenix_span

init_phoenix(project_name="lab-X", server_url="http://127.0.0.1:6006")

for i, q in enumerate(dev_set):
    with phoenix_span(label="baseline", attrs={"question_index": i}):
        result = run_pipeline(q["question"])
    with phoenix_span(label="hyde", attrs={"question_index": i}):
        result_hyde = run_pipeline_hyde(q["question"])
```

### 3. `init_phoenix(project_name, server_url)` — setup-only

For projects that just want auto-instrumentation without manual parent spans.
Every OpenAI / LangChain call after `init_phoenix(...)` emits spans
automatically.

```python
from phoenix_tracing import init_phoenix

init_phoenix(project_name="lab-X", server_url="http://127.0.0.1:6006")

# All OpenAI / LangChain calls anywhere in the process now emit spans.
# No manual wrapping needed.
result = some_openai_call(...)
```

## What you get in the Phoenix UI

| Tab | What it shows | When to use |
|---|---|---|
| **Projects** | Project list (matches `project_name` arg) | Switch between labs |
| **Traces** | Root spans (one per `trace_run` / `phoenix_span` call) — sortable by name, latency, error | Find slow / errored runs |
| **Trace detail** | Waterfall of nested spans + LLM input/output + retrieval docs | Debug a single call end-to-end |

## Idempotency + reuse

`init_phoenix(...)` is **idempotent** — calling it more than once with the
same args is a no-op. Calling it with DIFFERENT args raises a clear
`RuntimeError` (you'd be silently switching projects mid-process otherwise).

`trace_run(...)` calls `init_phoenix` on every invocation; after the first,
that's a free no-op.

## Installation

`phoenix_tracing` itself has no install — it's pure Python in `shared/`.
Its dependencies must be installed in your lab's venv:

```bash
uv pip install \
  arize-phoenix \
  arize-phoenix-otel \
  openinference-instrumentation-openai \
  openinference-instrumentation-langchain \
  opentelemetry-sdk
```

Then run the Phoenix collector once per machine (long-running):

```bash
python -m phoenix.server.main serve &
# UI at http://127.0.0.1:6006
```

## Reference implementation

`lab-03-rag-eval/src/05_trace.py` — the W3 lab where this pattern was originally
built. Comparing W3's pre-shared-lib trace script to a post-shared-lib version
shows what the abstraction earns:

```python
# Pre-shared-lib (W3 src/05_trace.py — ~30 LOC of ceremony):
import phoenix as px
from opentelemetry import trace as otel_trace
from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.instrumentation.langchain import LangChainInstrumentor

os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://127.0.0.1:6006"
tracer_provider = register(project_name="lab-03-rag-eval", auto_instrument=True)
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
_tracer = otel_trace.get_tracer("lab-03-rag-eval")

for i, q in enumerate(dev):
    for label, fn in [("baseline", baseline.run_pipeline), ...]:
        with _tracer.start_as_current_span(f"pipeline.{label}") as span:
            span.set_attribute("pipeline.variant", label)
            span.set_attribute("question_index", i)
            _ = fn(q["question"])

# Post-shared-lib (~10 LOC):
from phoenix_tracing import trace_run

for i, q in enumerate(dev):
    for label, fn in [("baseline", baseline.run_pipeline), ...]:
        trace_run(
            project_name="lab-03-rag-eval",
            server_url="http://127.0.0.1:6006",
            label=label,
            fn=fn,
            question=q["question"],
            attrs={"question_index": i},
        )
```

## What's NOT in here (yet)

- **Phoenix Datasets / Evals API** — Phoenix has a structured way to register
  dev sets and store eval scores per trace. The W3 lab uses RAGAS + a separate
  `RESULTS.md` table for now; Phoenix's Datasets+Evals would supersede that
  once a future lab needs cross-run comparison. Worth promoting to this
  library when adopted.
- **Custom span attribute schemas** — currently `attrs={...}` is freeform. A
  future enrichment could add typed `RAGAS_eval_attrs(faithfulness=0.99, ...)`
  helpers that set OpenInference standard attribute names.
- **Tracing decorator** — `@traced(label="baseline")` on functions. Trivial to
  add (~5 LOC) once a second lab has a use case.
