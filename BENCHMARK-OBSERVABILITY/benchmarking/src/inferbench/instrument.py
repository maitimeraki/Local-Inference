"""Tap vLLM's stats logger to read the scheduler and KV cache in-process.

vLLM exposes a `StatLogger` protocol and a `StatLoggerManager`. Registering a
second logger alongside its Prometheus one gives us the same numbers the
step-through inspector prints, sampled over a whole run instead of one request.

Attribute names on `SchedulerStats` have moved between vLLM versions, so
`extract_sample` reads defensively and records `None` for anything absent.
A missing field is visible; a wrong guess is not.

Two numbers cannot be read off a single step, so this logger owns them:

  * the prefix-cache hit rate — V1 drains its prefix counters at the end of
    every step, so only a logger that lives across steps can turn them into a
    run-level rate;
  * the iteration duration — V1's `IterationStats` carries no timing field, so
    the wall clock between two taps is the honest measure of one iteration.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


@dataclass
class EngineSample:
    """One observation of engine state."""

    ts: float
    running: int | None = None
    waiting: int | None = None
    kv_cache_usage: float | None = None
    prefix_cache_hit_rate: float | None = None
    preemptions: int | None = None  # preempted during this step
    elapsed_ms: float | None = None  # one scheduler/iteration step


def _get(obj: Any, *names: str) -> Any:
    """First attribute that exists and is not None, else None."""
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


def extract_sample(
    scheduler_stats: Any = None,
    iteration_stats: Any = None,
    *,
    prev_preemptions: int | None = None,
    now: float | None = None,
) -> EngineSample:
    """Pull the fields we care about off whatever vLLM handed us.

    Anything not found stays None rather than defaulting to zero — a zero
    waiting-queue looks like a healthy run, and that is the failure mode this
    whole project exists to catch.

    The prefix-cache rate and the iteration duration are not read here: V1
    hands out one step's worth of prefix counters and no timing at all, so
    `TapStatLogger` owns both.
    """
    sample = EngineSample(ts=now if now is not None else time.time())

    if scheduler_stats is not None:
        sample.running = _get(scheduler_stats, "num_running_reqs", "num_running")
        sample.waiting = _get(scheduler_stats, "num_waiting_reqs", "num_waiting")
        sample.kv_cache_usage = _get(
            scheduler_stats, "kv_cache_usage", "gpu_cache_usage_perc", "kv_cache_usage_perc"
        )

    if iteration_stats is not None:
        # V1 counts preemptions per iteration, on IterationStats, and that is
        # the only place the number actually lives. `SchedulerStats` declares
        # `num_preemptions` but no vLLM call site ever populates it, so it
        # always reads 0 — trusting it first would report "no preemptions" for
        # the whole run, and the abort path that watches this counter with it.
        per_iteration = _get(iteration_stats, "num_preempted_reqs")
        if per_iteration is not None:
            # Fold the per-iteration count into a running total, so the delta
            # below is the number preempted since the last record.
            sample.preemptions = (prev_preemptions or 0) + per_iteration

    if sample.preemptions is None and scheduler_stats is not None:
        # Older releases carried a cumulative counter here instead.
        sample.preemptions = _get(scheduler_stats, "num_preemptions", "preemptions")

    if prev_preemptions is not None and sample.preemptions is not None:
        # Normalise a cumulative counter into a per-step delta, which is what a
        # rate panel needs.
        sample.preemptions = max(0, sample.preemptions - prev_preemptions)

    return sample


class SampleSink(Protocol):
    def emit(self, sample: EngineSample) -> None: ...


@dataclass
class CollectingSink:
    """Keeps samples in memory. The runner drains these into the store."""

    samples: list[EngineSample] = field(default_factory=list)
    max_samples: int = 100_000

    def emit(self, sample: EngineSample) -> None:
        if len(self.samples) < self.max_samples:
            self.samples.append(sample)

    def as_rows(self) -> list[dict]:
        return [asdict(s) for s in self.samples]


class TapStatLogger:
    """A vLLM stat logger that forwards to a sink.

    Deliberately not subclassing vLLM's `StatLoggerBase`: the base class's
    signature has changed across releases, and a duck-typed `record()` keeps
    this working across them. Register with `attach()`.
    """

    def __init__(self, sink: SampleSink, local_interval: float = 1.0) -> None:
        self.sink = sink
        self.local_interval = local_interval
        self._prev_preemptions: int | None = None
        self._prefix_hits = 0
        self._prefix_queries = 0
        self._last_tap: float | None = None

    def record(self, *args: Any, **kwargs: Any) -> None:
        # vLLM's StatLoggerManager calls record(scheduler_stats, iteration_stats)
        # positionally; older releases used keywords.
        scheduler_stats = args[0] if args else kwargs.get("scheduler_stats")
        iteration_stats = args[1] if len(args) > 1 else kwargs.get("iteration_stats")
        sample = extract_sample(
            scheduler_stats,
            iteration_stats,
            prev_preemptions=self._prev_preemptions,
        )
        if sample.preemptions is not None:
            # Reconstruct the cumulative value for the next delta.
            self._prev_preemptions = (self._prev_preemptions or 0) + sample.preemptions
        self._fold_prefix(scheduler_stats, sample)
        sample.elapsed_ms = self._wall_clock_ms()
        self.sink.emit(sample)

    def _fold_prefix(self, scheduler_stats: Any, sample: EngineSample) -> None:
        """Turn V1's per-step prefix counters into a run-level hit rate.

        V1 drains its prefix-cache counters at the end of every step, so what
        it hands us covers one step only. A gauge has to describe the whole
        run, so the counters are summed here and divided at the end.
        """
        prefix = _get(scheduler_stats, "prefix_cache_stats")
        if prefix is None:
            return
        hits = _get(prefix, "hits")
        queries = _get(prefix, "queries")
        if hits is None or queries is None:
            return
        self._prefix_hits += hits
        self._prefix_queries += queries
        if self._prefix_queries:
            sample.prefix_cache_hit_rate = self._prefix_hits / self._prefix_queries

    def _wall_clock_ms(self) -> float | None:
        """Time between two taps: the duration of one engine iteration."""
        now = time.monotonic()
        if self._last_tap is None:
            self._last_tap = now
            return None
        elapsed_ms = (now - self._last_tap) * 1000.0
        self._last_tap = now
        return elapsed_ms

    # Methods vLLM may call; no-ops so attaching us never crashes the engine.
    def log(self) -> None:
        pass

    def log_engine_initialized(self) -> None:
        pass

    def record_sleep_state(self, *args: Any, **kwargs: Any) -> None:
        pass


def attach(engine: Any, logger: TapStatLogger) -> bool:
    """Best-effort registration onto a running engine.

    V1 keeps the loggers on `engine.logger_manager.stat_loggers`; the manager's
    own attribute name has moved between releases, so this walks the likely
    paths and reports whether it worked instead of raising. When it returns
    False, `inspect` is still usable — it drives the engine directly and needs
    no logger for anything but the prefix rate and preemption counts.
    """
    for attr in ("logger_manager", "stat_loggers", "_stat_loggers"):
        manager = getattr(engine, attr, None)
        if manager is None:
            continue
        loggers = (
            manager
            if isinstance(manager, (list, dict))
            else _get(manager, "stat_loggers", "loggers")
        )
        if isinstance(loggers, dict):
            loggers[f"tap_{id(logger)}"] = logger
            return True
        if isinstance(loggers, list):
            loggers.append(logger)
            return True
    return False
