"""Phoenix tracing primitives — distilled from W3 lab `src/05_trace.py`.

Three APIs at different ergonomic levels:

  - trace_run(project_name, server_url, label, fn, *args, **kwargs) — one-call
  - phoenix_span(label, attrs=None) — context manager around a code block
  - init_phoenix(project_name, server_url) — setup once, auto-instrument the rest

See README.md for usage + the lab-03-rag-eval reference implementation.
"""

from .tracing import init_phoenix, phoenix_span, trace_run

__all__ = ["init_phoenix", "phoenix_span", "trace_run"]
