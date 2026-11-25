"""OpenTelemetry setup, shared by the harness, the gateway and the server.

Deliberately degrade-safe: if the `tracing` extra is not installed, every
function still works and returns a no-op. Instrumentation must never be the
reason a benchmark run fails.

The point of tracing here is the one thing neither metrics nor benchmarks give
you: a single request's full journey — gateway wait -> server queue -> prefill
-> decode — with per-phase durations, for the p99 that looked anomalous.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator

_ENABLED = False
_TRACER = None


def setup_tracing(service_name: str, endpoint: str | None = None) -> bool:
    """Configure the OTLP exporter. Returns True if tracing is active.

    Safe to call more than once; the first successful call wins.
    """
    global _ENABLED, _TRACER
    if _ENABLED:
        return True

    endpoint = endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        return False

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)

    _TRACER = trace.get_tracer(service_name)
    _ENABLED = True
    return True


@contextlib.contextmanager
def span(name: str, **attributes: object) -> Iterator[object]:
    """A span if tracing is on, a no-op otherwise.

    Usage:
        with span("prefill", request_id=rid) as s:
            ...
    """
    if not _ENABLED or _TRACER is None:
        yield None
        return
    with _TRACER.start_as_current_span(name) as s:
        for k, v in attributes.items():
            if v is not None:
                s.set_attribute(k, v)
        yield s


def current_trace_id() -> str | None:
    """Hex trace id of the active span, or None when tracing is off.

    The gateway and the server put this on responses so a Prometheus exemplar
    can link a metric spike to the exact trace.
    """
    if not _ENABLED:
        return None
    from opentelemetry import trace

    ctx = trace.get_current_span().get_span_context()
    return format(ctx.trace_id, "032x") if ctx.is_valid else None


def shutdown() -> None:
    """Flush pending spans. Call before process exit."""
    global _ENABLED
    if not _ENABLED:
        return
    from opentelemetry import trace

    provider = trace.get_tracer_provider()
    flush = getattr(provider, "shutdown", None)
    if flush is not None:
        flush()
    _ENABLED = False
