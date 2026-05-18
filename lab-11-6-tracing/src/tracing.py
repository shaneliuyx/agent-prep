"""Phase 1 — OpenTelemetry instrumentation primitive for W4 ReAct loop.

Wraps LLM calls + tool calls with spans carrying:
  - agent.role        (loop / tool_arg / classify / reason / compose / finisher / hard_loop)
  - model.name        (gateway model id)
  - tokens.in, tokens.out
  - cost_usd          (computed inline via RATE_CARD)
  - duration_ms

Spans flow via OTLP gRPC to a Langfuse self-hosted instance.

Cost formula:
  $C_{\\text{call}} = t_{\\text{in}} \\cdot p_{\\text{in}} + t_{\\text{out}} \\cdot p_{\\text{out}}$
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from functools import wraps
from typing import Callable, Iterator

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter


# Cloud-equivalent rate card (USD per 1M tokens). Local-MLX rates can be 0
# OR set to opportunity cost (what you'd pay on cloud for the same work).
RATE_CARD: dict[str, dict[str, float]] = {
    "MLX-Qwen3.5-9B-GLM5.1-Distill-v1-8bit": {"in": 0.25, "out": 1.00},
    "gemma-4-26B-A4B-it-heretic-4bit":       {"in": 1.00, "out": 3.00},
    "Gemma-4-31B-JANG_4M-CRACK":             {"in": 2.00, "out": 6.00},
    "gpt-oss-20b-MXFP4-Q8":                  {"in": 0.50, "out": 2.00},
    "Qwen3.6-35B-A3B-nvfp4":                 {"in": 2.50, "out": 7.50},
}

# strip `models/` prefix from gateway-style model names before lookup
def _normalize_model(name: str) -> str:
    return name.removeprefix("models/")


def init_tracing(service_name: str = "w11-6-tracing-lab",
                 otlp_endpoint: str = "http://localhost:4317") -> None:
    """Wire OTEL to Langfuse via OTLP gRPC. Call once at process start."""
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)


_tracer = trace.get_tracer(__name__)


def compute_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    """$C = t_{in} \\cdot p_{in} + t_{out} \\cdot p_{out}$ per 1M-token rates."""
    rates = RATE_CARD.get(_normalize_model(model), {"in": 0.0, "out": 0.0})
    return (tokens_in / 1e6) * rates["in"] + (tokens_out / 1e6) * rates["out"]


@contextmanager
def llm_call_span(role: str, model: str) -> Iterator:
    """Span context manager for one LLM call.
    Caller writes tokens + cost attributes after the call returns."""
    with _tracer.start_as_current_span("llm_call") as span:
        span.set_attribute("agent.role", role)
        span.set_attribute("model.name", model)
        t0 = time.perf_counter()
        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            raise
        finally:
            span.set_attribute("duration_ms", (time.perf_counter() - t0) * 1000)


def annotate_usage(span, model: str, tokens_in: int, tokens_out: int) -> None:
    """Set tokens + cost attributes on an open span. Call inside llm_call_span."""
    span.set_attribute("tokens.in", tokens_in)
    span.set_attribute("tokens.out", tokens_out)
    span.set_attribute("cost_usd", compute_cost(model, tokens_in, tokens_out))


def traced(span_name: str) -> Callable:
    """Decorator for non-LLM spans (tool calls, parsing, etc)."""
    def deco(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            with _tracer.start_as_current_span(span_name) as span:
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    raise
        return wrapper
    return deco
