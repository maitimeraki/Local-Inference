"""The scheduler tier's Prometheus instruments."""

from __future__ import annotations

from inferobs import metric_names as N
from prometheus_client import Counter, Gauge, Histogram

_METRICS: dict = {}


def metrics() -> dict:
    """Create the instruments once per process.

    Guarded because the module can be imported more than once and
    prometheus_client raises on duplicate registration.
    """
    if _METRICS:
        return _METRICS
    _METRICS.update(
        free_blocks=Gauge(N.SCHED_FREE_BLOCKS, "KV blocks the engine has free"),
        total_blocks=Gauge(N.SCHED_TOTAL_BLOCKS, "KV blocks the engine was given"),
        reserved_blocks=Gauge(N.SCHED_RESERVED_BLOCKS, "Blocks the ledger has promised"),
        allocated_blocks=Gauge(N.SCHED_ALLOCATED_BLOCKS, "Blocks the engine has granted"),
        drift=Gauge(
            N.SCHED_RESERVATION_DRIFT_BLOCKS,
            "Engine-held blocks the ledger does not account for; must stay 0",
        ),
        dispatch=Counter(N.SCHED_DISPATCH_TOTAL, "Admission decisions", [N.L_DECISION]),
        hold_depth=Gauge(N.SCHED_HOLD_DEPTH, "Requests waiting for capacity"),
        hold_seconds=Histogram(
            N.SCHED_HOLD_SECONDS,
            "Time a request waited for capacity",
            buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
        ),
        dispatch_latency=Histogram(
            N.SCHED_DISPATCH_LATENCY,
            "Time from a slot becoming free to the engine starting the request",
            buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 30),
        ),
        preemptions=Counter(N.SCHED_PREEMPTION_TOTAL, "Engine preemptions seen by the scheduler"),
        capacity_breaches=Counter(
            N.SCHED_CAPACITY_BREACH_TOTAL,
            "Requests the scheduler shed after the engine ran out of KV blocks",
        ),
        inflight=Gauge(N.SCHED_USER_INFLIGHT, "Requests in flight across all users"),
        requests=Counter(N.SCHED_REQUESTS_TOTAL, "Request outcomes", [N.L_STATUS]),
    )
    return _METRICS
