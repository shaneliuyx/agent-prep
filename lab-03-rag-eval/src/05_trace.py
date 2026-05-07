"""Re-run baseline + HyDE + multi-query, emitting OpenTelemetry traces to Phoenix.

Refactored 2026-05-07 from inline 30-line OTel ceremony to a single
`trace_run(...)` call from `shared/phoenix_tracing/`. The shared lib does:
  - cached idempotent register() setup on first call
  - explicit OpenAIInstrumentor + LangChainInstrumentor
  - parent span with `pipeline.variant` attribute
  - all auto-instrumented children nest under the parent

Filter in Phoenix UI:
  - "Root Spans" tab shows only the 30 pipeline.* parents
  - Filter `name == "baseline"` (or "hyde" / "mq") for one variant
  - Filter `attributes["pipeline.variant"] == "baseline"` for the same effect
"""
import json
import sys
from pathlib import Path

# Bootstrap shared/phoenix_tracing onto sys.path
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "shared"))

from phoenix_tracing import trace_run  # noqa: E402
from src.script_wrap import load  # noqa: E402

PROJECT = "lab-03-rag-eval"
PHOENIX_URL = "http://127.0.0.1:6006"

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
        print(f"  [{label}] {q['question'][:60]}")
        trace_run(
            PROJECT, PHOENIX_URL, label, fn,
            q["question"],
            attrs={"question": q["question"], "question_index": i},
        )

print(f"\ntraces in Phoenix: {PHOENIX_URL}")
print("Filter root spans by name: baseline | hyde | mq")
