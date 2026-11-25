"""Gateway Prometheus instruments.

Names come from `inferobs.metric_names` so the dashboards and alert rules that
query them cannot drift from the code that emits them.
"""

from __future__ import annotations

from inferobs import metric_names as N
from prometheus_client import Counter, Gauge, Histogram

_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)

_METRICS: dict = {}


def metrics() -> dict:
    """Create instruments once per process (prometheus_client rejects duplicates)."""
    if _METRICS:
        return _METRICS
    _METRICS.update(
        requests=Counter(N.GATE_REQUESTS_TOTAL, "Requests handled", [N.L_BACKEND, N.L_STATUS]),
        duration=Histogram(
            N.GATE_REQUEST_DURATION,
            "End-to-end gateway latency",
            [N.L_BACKEND],
            buckets=_LATENCY_BUCKETS,
        ),
        queue=Histogram(
            N.GATE_QUEUE_TIME,
            "Time between accept and forward — the gateway's own queueing delay",
            buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0),
        ),
        in_flight=Gauge(N.GATE_IN_FLIGHT, "Requests currently running on the engine"),
        waiting=Gauge(N.GATE_WAITING, "Requests held in the scheduler's queue"),
        rejected=Counter(N.GATE_REJECTED_TOTAL, "Requests shed by admission control", [N.L_REASON]),
        cache_hits=Counter(N.GATE_CACHE_HITS_TOTAL, "Exact-match cache hits"),
        cache_misses=Counter(N.GATE_CACHE_MISSES_TOTAL, "Exact-match cache misses"),
        cache_saved=Histogram(
            N.GATE_CACHE_SAVED,
            "Upstream latency avoided by a cache hit",
            buckets=(0.001, 0.01, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
        ),
        upstream=Histogram(
            N.GATE_UPSTREAM_DURATION,
            "Latency of the upstream engine call",
            [N.L_BACKEND],
            buckets=_LATENCY_BUCKETS,
        ),
        admitted=Counter(N.GATE_ADMITTED_TOTAL, "Requests the scheduler admitted"),
        held=Counter(N.GATE_HELD_TOTAL, "Requests the scheduler held", [N.L_REASON]),
        idempotent_replays=Counter(
            N.GATE_IDEMPOTENT_REPLAYS_TOTAL, "Duplicate submits refused by idempotency"
        ),
        hold_seconds=Histogram(
            N.GATE_HOLD_SECONDS,
            "Time the scheduler held a request before dispatching it",
            buckets=(0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
        ),
        user_queue_depth=Histogram(
            N.GATE_USER_QUEUE_DEPTH,
            "The request's position in its user's hold queue",
            buckets=(0, 1, 2, 4, 8, 16, 32, 64),
        ),
        ttft=Histogram(
            N.GATE_TTFT,
            "Time to first token, as the client sees it — accept to first token",
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
        ),
        tpot=Histogram(
            N.GATE_TPOT,
            "Time per output token after the first — vLLM's definition",
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
        ),
    )
    return _METRICS
