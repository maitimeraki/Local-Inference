"""Shared observability library for the vLLM platform.

`metric_names` owns the vocabulary (so emitters, dashboards and alerts can't
drift). `tracing` sets up OpenTelemetry once. `slo` and `burn` turn measured
results into compliance verdicts.
"""

from inferobs.metric_names import (  # noqa: F401
    BENCH_COST_PER_MTOK,
    BENCH_DURATION_SECONDS,
    BENCH_FAILURES_TOTAL,
    BENCH_METRIC,
    ENGINE_ITERATION_SECONDS,
    ENGINE_KV_CACHE_USAGE,
    ENGINE_PREEMPTIONS_TOTAL,
    ENGINE_PREFIX_CACHE_HIT_RATE,
    ENGINE_REQUESTS_TOTAL,
    ENGINE_RUNNING_REQUESTS,
    ENGINE_SCHEDULED_TOKENS_TOTAL,
    ENGINE_WAITING_REQUESTS,
    GATE_CACHE_HITS_TOTAL,
    GATE_CACHE_MISSES_TOTAL,
    GATE_IN_FLIGHT,
    GATE_QUEUE_TIME,
    GATE_REJECTED_TOTAL,
    GATE_REQUEST_DURATION,
    GATE_REQUESTS_TOTAL,
    GATE_UPSTREAM_DURATION,
    MEASUREMENTS,
    STATS,
)

__version__ = "0.1.0"
