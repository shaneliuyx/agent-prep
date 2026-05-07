"""High-level Phoenix tracing helper — one-call API for the W3 lab pattern.

Usage spectrum (most ergonomic first):

    # 1. One-call wrapper (user's preferred shape):
    result = trace_run(
        project_name="lab-03-rag-eval",
        server_url="http://127.0.0.1:6006",
        label="baseline",
        fn=run_pipeline,
        question=q,
    )

    # 2. Context manager (inline tracing of arbitrary blocks):
    init_phoenix(project_name="lab-X", server_url="http://...")
    with phoenix_span(label="baseline", attrs={"question": q}):
        result = run_pipeline(q)

    # 3. Setup-only (then auto-instrumentation captures all OpenAI / LangChain calls):
    init_phoenix(project_name="lab-X", server_url="http://...")
    # ...your pipeline now emits spans without manual wrapping...

All three paths flow through the same `init_phoenix` cached setup. Calling
`init_phoenix` more than once with the same args is a no-op (idempotent);
calling with DIFFERENT args raises to surface the misconfiguration.

Encapsulates the W3 lab's `src/05_trace.py` pattern (parent span with
variant attribute + nested auto-instrumented children) so future labs
can import this once instead of copying the boilerplate.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable

# Module-level cache so init_phoenix is idempotent across multi-call workflows
_INIT_STATE: dict[str, Any] = {"initialized": False, "project_name": None, "server_url": None}
_TRACER = None


def init_phoenix(
    project_name: str,
    server_url: str = "http://127.0.0.1:6006",
    *,
    auto_instrument: bool = True,
    instrument_openai: bool = True,
    instrument_langchain: bool = True,
) -> Any:
    """Configure Phoenix tracing for this Python process. Idempotent.

    Args:
        project_name:        groups runs in the Phoenix UI by project.
        server_url:          Phoenix collector endpoint. Defaults to local.
        auto_instrument:     hand to `phoenix.otel.register` — auto-detects
                             OpenAI / LangChain / etc. via SDK patches.
        instrument_openai:   explicit OpenAIInstrumentor call. Belt-and-
                             suspenders for SDK-version-drift safety.
        instrument_langchain: explicit LangChainInstrumentor call. RAGAS
                             internals + LangChain runnables get spans.

    Returns the OpenTelemetry tracer for this project (caller can grab
    the same tracer via `_get_tracer()` later).

    Raises if called twice with conflicting (project_name, server_url).
    """
    global _TRACER

    if _INIT_STATE["initialized"]:
        if (_INIT_STATE["project_name"] != project_name
                or _INIT_STATE["server_url"] != server_url):
            raise RuntimeError(
                f"init_phoenix already initialized with "
                f"project_name={_INIT_STATE['project_name']!r} "
                f"server_url={_INIT_STATE['server_url']!r}; cannot reinit "
                f"with project_name={project_name!r} server_url={server_url!r}. "
                "Restart the process to switch projects."
            )
        return _TRACER

    # Lazy imports — only require phoenix-otel + openinference when init is called
    import os
    from opentelemetry import trace as otel_trace
    from phoenix.otel import register

    os.environ.setdefault("PHOENIX_COLLECTOR_ENDPOINT", server_url)

    register(
        project_name=project_name,
        auto_instrument=auto_instrument,
        endpoint=f"{server_url.rstrip('/')}/v1/traces",
    )

    if instrument_openai:
        try:
            from openinference.instrumentation.openai import OpenAIInstrumentor
            OpenAIInstrumentor().instrument()
        except ImportError:
            pass  # silently skip if not installed; auto_instrument may cover it

    if instrument_langchain:
        try:
            from openinference.instrumentation.langchain import LangChainInstrumentor
            LangChainInstrumentor().instrument()
        except ImportError:
            pass

    _TRACER = otel_trace.get_tracer(project_name)
    _INIT_STATE["initialized"] = True
    _INIT_STATE["project_name"] = project_name
    _INIT_STATE["server_url"] = server_url
    return _TRACER


@contextmanager
def phoenix_span(label: str, attrs: dict | None = None):
    """Context manager wrapping a Phoenix parent span.

    Requires `init_phoenix(...)` to have been called first. Spans emitted
    by OpenAI / LangChain auto-instrumentation inside the `with` block
    nest under this span automatically.

    Args:
        label:  span name (also recorded as `pipeline.variant` attribute
                so Phoenix UI can filter root spans by variant).
        attrs:  additional OpenTelemetry span attributes (str/int/float/bool
                or sequences thereof). e.g. {"question": q, "iteration": 3}.
    """
    if not _INIT_STATE["initialized"]:
        raise RuntimeError(
            "phoenix_span called before init_phoenix; "
            "call init_phoenix(project_name=..., server_url=...) once at process start"
        )
    assert _TRACER is not None
    with _TRACER.start_as_current_span(label) as span:
        span.set_attribute("pipeline.variant", label)
        for k, v in (attrs or {}).items():
            try:
                span.set_attribute(k, v)
            except Exception:  # noqa: BLE001 — never let a bad attr break the run
                pass
        yield span


def trace_run(
    project_name: str,
    server_url: str,
    label: str,
    fn: Callable[..., Any],
    *args: Any,
    attrs: dict | None = None,
    **kwargs: Any,
) -> Any:
    """One-call API: init + parent span + run + return fn's result.

    The shape the user asked for: 'inputs are project_name, server_url,
    label, function. Then we can get tracing data.'

    Example:
        result = trace_run(
            project_name="lab-03-rag-eval",
            server_url="http://127.0.0.1:6006",
            label="baseline",
            fn=run_pipeline,
            question=q,
            attrs={"question_index": i},
        )
        # → result is whatever fn returned
        # → Phoenix UI now shows a 'baseline' root span with auto-
        #   instrumented OpenAI / LangChain children nested under it
    """
    init_phoenix(project_name=project_name, server_url=server_url)
    with phoenix_span(label=label, attrs=attrs):
        return fn(*args, **kwargs)
