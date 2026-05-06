"""Re-run baseline + HyDE + multi-query, emitting OpenTelemetry traces to Phoenix.

Each pipeline call is wrapped in a NAMED parent span (`pipeline.baseline`,
`pipeline.hyde`, `pipeline.mq`) with a `pipeline.variant` attribute. Without
this wrapper, OpenAI auto-instrumentation emits ~5 raw `ChatCompletion`
spans per query — Phoenix UI shows 150 indistinguishable rows. With the
wrapper, root spans group cleanly into 30 entries (10 queries × 3 variants)
and child spans nest under each variant for waterfall inspection.

Filter in Phoenix UI:
  - "Root Spans" tab shows only the 30 pipeline.* parents
  - Filter `name == "pipeline.baseline"` (or .hyde / .mq) for one variant
  - Filter `attributes["pipeline.variant"] == "baseline"` for the same effect
"""
import json
import os
import phoenix as px
from opentelemetry import trace as otel_trace
from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.instrumentation.langchain import LangChainInstrumentor
from src.script_wrap import load

os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://127.0.0.1:6006"
tracer_provider = register(project_name="lab-03-rag-eval", auto_instrument=True)
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

# Manual tracer for variant-tagged parent spans.
_tracer = otel_trace.get_tracer("lab-03-rag-eval")

baseline = load("02_pipeline.py")
hyde = load("03_hyde.py")
mq = load("04_multiquery.py")

with open("data/dev_set.jsonl", encoding="utf-8") as _f:
    dev = [json.loads(l) for l in _f][:10]

for i, q in enumerate(dev):
    for label, fn in [
        ("baseline", baseline.run_pipeline),
        ("hyde", hyde.run_pipeline_hyde),
        ("mq", mq.run_pipeline_mq),
    ]:
        # Parent span — every child OpenAI / LangChain span nests under this
        # so Phoenix groups all retrieve / rerank / LLM calls per variant.
        with _tracer.start_as_current_span(f"pipeline.{label}") as span:
            span.set_attribute("pipeline.variant", label)
            span.set_attribute("question", q["question"])
            span.set_attribute("question_index", i)
            print(f"  [{label}] {q['question'][:60]}")
            _ = fn(q["question"])

print("\ntraces in Phoenix: http://127.0.0.1:6006")
print("Filter root spans by name: pipeline.baseline | pipeline.hyde | pipeline.mq")
