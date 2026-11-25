"""Stat-logger tap extraction.

vLLM's `SchedulerStats` attribute names have moved between releases, so the
extraction is defensive. These tests pin both behaviours that matter: it reads
what is there, and it reports `None` — never zero — for what is not.

The tap also owns the two numbers V1 cannot hand out per step: a cumulative
prefix-cache hit rate (V1 drains its counters every step) and an iteration
duration (V1's `IterationStats` carries no timing).
"""

from __future__ import annotations

from inferbench.instrument import (
    CollectingSink,
    TapStatLogger,
    attach,
    extract_sample,
)


class FakePrefixStats:
    def __init__(self, hits=None, queries=None):
        if hits is not None:
            self.hits = hits
        if queries is not None:
            self.queries = queries


class FakeSchedulerStats:
    def __init__(self, running=3, waiting=1, kv=0.42, prefix=None, preemptions=None):
        self.num_running_reqs = running
        self.num_waiting_reqs = waiting
        self.kv_cache_usage = kv
        if prefix is not None:
            self.prefix_cache_stats = prefix
        if preemptions is not None:
            self.num_preemptions = preemptions


class FakeIterationStats:
    def __init__(self, preempted=None):
        if preempted is not None:
            self.num_preempted_reqs = preempted


def test_reads_known_fields():
    sample = extract_sample(FakeSchedulerStats())
    assert sample.running == 3
    assert sample.waiting == 1
    assert sample.kv_cache_usage == 0.42


def test_missing_stats_yield_none_not_zero():
    """A zero waiting-queue reads as a healthy run. That is the failure mode."""
    sample = extract_sample(None)
    assert sample.running is None
    assert sample.waiting is None
    assert sample.kv_cache_usage is None


def test_alternate_attribute_names_are_accepted():
    class Alt:
        num_running_reqs = 5
        num_waiting_reqs = 0
        gpu_cache_usage_perc = 0.9

    sample = extract_sample(Alt())
    assert sample.kv_cache_usage == 0.9
    assert sample.waiting == 0


def test_prefix_hit_rate_is_cumulative_across_records():
    """V1 drains its prefix counters every step; the tap has to add them up."""
    sink = CollectingSink()
    logger = TapStatLogger(sink)

    logger.record(FakeSchedulerStats(prefix=FakePrefixStats(hits=3, queries=4)))
    logger.record(FakeSchedulerStats(prefix=FakePrefixStats(hits=1, queries=4)))

    assert sink.samples[0].prefix_cache_hit_rate == 0.75
    assert sink.samples[1].prefix_cache_hit_rate == 4 / 8


def test_prefix_hit_rate_is_none_without_counters():
    sink = CollectingSink()
    TapStatLogger(sink).record(FakeSchedulerStats())
    assert sink.samples[0].prefix_cache_hit_rate is None


def test_preemptions_are_normalised_to_a_delta():
    """A cumulative counter is useless on a per-step panel."""
    raw = FakeSchedulerStats(preemptions=7)
    first = extract_sample(raw, prev_preemptions=None)
    assert first.preemptions == 7

    second = extract_sample(raw, prev_preemptions=5)
    assert second.preemptions == 2


def test_preemption_delta_never_goes_negative():
    raw = FakeSchedulerStats(preemptions=2)
    assert extract_sample(raw, prev_preemptions=10).preemptions == 0


def test_logger_tracks_cumulative_preemptions_across_records():
    sink = CollectingSink()
    logger = TapStatLogger(sink)

    logger.record(FakeSchedulerStats(preemptions=3))
    logger.record(FakeSchedulerStats(preemptions=5))

    assert [s.preemptions for s in sink.samples] == [3, 2]


def test_logger_reads_v1_iteration_preemptions():
    """V1 counts preemptions per iteration, on IterationStats."""
    sink = CollectingSink()
    logger = TapStatLogger(sink)

    logger.record(FakeSchedulerStats(), FakeIterationStats(preempted=2))
    logger.record(FakeSchedulerStats(), FakeIterationStats(preempted=1))

    assert [s.preemptions for s in sink.samples] == [2, 1]


def test_the_iteration_counter_beats_the_unpopulated_scheduler_field():
    """vLLM declares `SchedulerStats.num_preemptions` and never fills it in.

    It always reads 0, so reading it first reports "no preemptions" for the
    whole run — and the abort path that watches this counter with it.
    """
    sample = extract_sample(
        FakeSchedulerStats(preemptions=0),
        FakeIterationStats(preempted=2),
    )
    assert sample.preemptions == 2


def test_logger_accepts_keyword_arguments_too():
    """Different vLLM versions pass stats positionally and by keyword."""
    sink = CollectingSink()
    logger = TapStatLogger(sink)
    logger.record(scheduler_stats=FakeSchedulerStats(running=9))
    assert sink.samples[0].running == 9


def test_logger_measures_the_wall_clock_between_taps():
    """V1 reports no iteration duration, so the tap times it itself."""
    sink = CollectingSink()
    logger = TapStatLogger(sink)
    logger.record(FakeSchedulerStats())
    logger.record(FakeSchedulerStats())
    assert sink.samples[0].elapsed_ms is None
    assert sink.samples[1].elapsed_ms is not None


def test_collecting_sink_is_bounded():
    sink = CollectingSink(max_samples=2)
    for i in range(5):
        logger = TapStatLogger(sink)
        logger.record(FakeSchedulerStats(running=i))
    assert len(sink.samples) == 2


def test_sink_rows_are_plain_dicts():
    sink = CollectingSink()
    TapStatLogger(sink).record(FakeSchedulerStats())
    assert sink.as_rows()[0]["running"] == 3


def test_attach_finds_the_v1_stat_loggers():
    class Manager:
        def __init__(self):
            self.stat_loggers: list = []

    class Engine:
        def __init__(self):
            self.logger_manager = Manager()

    engine = Engine()
    logger = TapStatLogger(CollectingSink())
    assert attach(engine, logger) is True
    assert logger in engine.logger_manager.stat_loggers


def test_attach_reports_failure():
    class Bare:
        pass

    assert attach(Bare(), TapStatLogger(CollectingSink())) is False
