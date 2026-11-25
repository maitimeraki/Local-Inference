"""Canonical metric names and label keys.

One place that owns the vocabulary, so the engines that *emit* metrics, the
dashboards that *query* them, and the alerts that *fire* on them can never
drift apart. If you rename something here, the PromQL in
`observability/alerts/slo_rules.yml` and the Grafana JSON are the files to fix.
"""

from __future__ import annotations

# --- Gateway (infergw) ------------------------------------------------------
# Emitted by the request router.
GATE_REQUESTS_TOTAL = "gate_requests_total"  # labels: backend, status
GATE_REQUEST_DURATION = "gate_request_duration_seconds"  # labels: backend
GATE_QUEUE_TIME = "gate_queue_time_seconds"  # accept -> forward; the queueing delay
GATE_IN_FLIGHT = "gate_in_flight"  # gauge, no labels; running on the engine
GATE_WAITING = "gate_waiting"  # gauge, no labels; held by the scheduler's queue
GATE_REJECTED_TOTAL = "gate_rejected_total"  # labels: reason ("saturated")
GATE_CACHE_HITS_TOTAL = "gate_cache_hits_total"
GATE_CACHE_MISSES_TOTAL = "gate_cache_misses_total"
GATE_CACHE_SAVED = "gate_cache_saved_seconds"  # latency avoided by a hit
GATE_UPSTREAM_DURATION = "gate_upstream_duration_seconds"  # labels: backend
GATE_ADMITTED_TOTAL = "gate_admitted_total"  # admitted by the scheduler
GATE_HELD_TOTAL = "gate_held_total"  # labels: reason; held by the scheduler
GATE_IDEMPOTENT_REPLAYS_TOTAL = "gate_idempotent_replays_total"  # duplicate submit refused
GATE_HOLD_SECONDS = "gate_hold_seconds"  # time the scheduler held the request
GATE_USER_QUEUE_DEPTH = "gate_user_queue_depth"  # the request's place in its user's queue
GATE_TTFT = "gate_ttft_seconds"  # accept -> first token; the number the user feels
GATE_TPOT = "gate_tpot_seconds"  # (last token - first token) / (output tokens - 1)

# --- Engine (inferbench.server, tapped in-process) --------------------------
# Emitted by the server we own around AsyncLLMEngine. These mirror the
# scheduler/KV-cache state the step-through inspector prints.
ENGINE_RUNNING_REQUESTS = "engine_running_requests"
ENGINE_WAITING_REQUESTS = "engine_waiting_requests"
ENGINE_KV_CACHE_USAGE = "engine_kv_cache_usage_ratio"  # 0..1
ENGINE_PREFIX_CACHE_HIT_RATE = "engine_prefix_cache_hit_rate"  # 0..1
ENGINE_ITERATION_SECONDS = "engine_iteration_seconds"  # one decode/step duration
ENGINE_PREEMPTIONS_TOTAL = "engine_preemptions_total"
ENGINE_SCHEDULED_TOKENS_TOTAL = "engine_scheduled_tokens_total"  # labels: phase
ENGINE_REQUESTS_TOTAL = "engine_requests_total"  # labels: status

# --- Scheduler (infersched) --------------------------------------------------
# Emitted by the scheduler tier, which owns the KV-block ledger and the hold
# queue. `reserved` is what the ledger has *promised*; `allocated` is what the
# engine has actually granted. The difference is the predictor's error.
SCHED_FREE_BLOCKS = "sched_free_blocks"  # engine's free KV blocks right now
SCHED_TOTAL_BLOCKS = "sched_total_blocks"  # engine's total KV blocks
SCHED_RESERVED_BLOCKS = "sched_reserved_blocks"  # promised but not yet granted
SCHED_ALLOCATED_BLOCKS = "sched_allocated_blocks"  # granted to tracked requests
SCHED_RESERVATION_DRIFT_BLOCKS = "sched_reservation_drift_blocks"  # must stay 0
SCHED_DISPATCH_TOTAL = "sched_dispatch_total"  # labels: decision
SCHED_HOLD_DEPTH = "sched_hold_depth"
SCHED_HOLD_SECONDS = "sched_hold_seconds"  # time spent in the hold queue
SCHED_DISPATCH_LATENCY = "sched_dispatch_latency_seconds"  # slot free -> engine start
SCHED_PREEMPTION_TOTAL = "sched_preemption_total"
SCHED_CAPACITY_BREACH_TOTAL = "sched_capacity_breach_total"  # requests shed by the scheduler
SCHED_USER_INFLIGHT = "sched_user_inflight"  # no labels; total in flight across users
SCHED_REQUESTS_TOTAL = "sched_requests_total"  # labels: status

# --- Benchmark runs (inferbench.runner -> pushgateway) ----------------------
# Deliberately one generic metric with a `metric`/`stat` label pair rather than
# dozens of hardcoded names: adding a new measurement is then a data change.
BENCH_METRIC = "bench_metric"  # labels: run_id, workload, config, concurrency, metric, stat
BENCH_COST_PER_MTOK = "bench_cost_usd_per_million_output_tokens"
BENCH_FAILURES_TOTAL = "bench_failures_total"  # labels: run_id, reason
BENCH_DURATION_SECONDS = "bench_duration_seconds"  # labels: run_id

# --- External exporters -----------------------------------------------------
# Emitted by the GPU exporter container (observability/compose.yaml), not by
# this repo. They are in the vocabulary because the dashboards and the alerts
# query them, and a metric nobody emits is a rule that never fires.
NVIDIA_SMI_UTILIZATION = "nvidia_smi_utilization_gpu_ratio"
NVIDIA_SMI_POWER_DRAW = "nvidia_smi_power_draw_watts"
NVIDIA_SMI_TEMPERATURE = "nvidia_smi_temperature_gpu"
NVIDIA_SMI_MEMORY_USED = "nvidia_smi_memory_used_bytes"
NVIDIA_SMI_MEMORY_TOTAL = "nvidia_smi_memory_total_bytes"

# --- Label keys -------------------------------------------------------------
L_BACKEND = "backend"
L_STATUS = "status"
L_REASON = "reason"
L_PHASE = "phase"
L_RUN_ID = "run_id"
L_WORKLOAD = "workload"
L_CONFIG = "config"
L_CONCURRENCY = "concurrency"
L_METRIC = "metric"
L_STAT = "stat"
L_DECISION = "decision"

# Statistical labels used on BENCH_METRIC.
STATS = ("mean", "p50", "p95", "p99", "min", "max")

# Measurement names used on BENCH_METRIC.
MEASUREMENTS = (
    "ttft",  # time to first token, seconds
    "itl",  # inter-token latency, seconds
    "tpot",  # time per output token, seconds
    "e2e",  # end-to-end request latency, seconds
    "queue",  # server-side queue time, seconds
    "input_tps",  # input (prefill) tokens/sec
    "output_tps",  # output (decode) tokens/sec
    "request_tps",  # requests/sec
    "gpu_mem",  # peak GPU memory, bytes
)


def slo_metric_name(name: str) -> str:
    """Map a human SLO id to the `bench_metric` selector it is derived from.

    Kept here rather than in the SLO module so the rule file and the evaluator
    agree on spelling.
    """
    return BENCH_METRIC
