"""Step-through rendering and V1 scheduler reading.

The vLLM-driving half needs a GPU; the parsing and formatting half does not,
and it is the half that decides whether the output is readable. Tested here
against a fake engine shaped like vLLM's V1 engine, where the scheduler hangs
two levels below the frontend, off the engine core.
"""

from __future__ import annotations

import pytest
from inferbench.inspect import (
    SchedulerUnreachable,
    StepReport,
    format_step,
    require_scheduler,
    snapshot,
)
from inferbench.instrument import EngineSample

# --- formatting -------------------------------------------------------------


def test_format_step_matches_the_documented_shape():
    report = StepReport(
        index=7,
        running=5,
        waiting=2,
        tokens_this_step=4096,
        kv_blocks_used=612,
        kv_blocks_total=1600,
        elapsed_ms=41.0,
    )
    line = format_step(report)
    assert line.startswith("step 0007")
    assert "2 new + 5 running" in line
    assert "kv 38%" in line
    assert "612/1600 blk" in line
    assert "41ms" in line


def test_format_step_marks_a_preemption_on_its_own_line():
    report = StepReport(
        index=9, running=5, preempted=1, kv_blocks_used=94, kv_blocks_total=100
    )
    line = format_step(report)
    assert line.count("\n") == 1
    assert "preempted 1" in line
    assert "kv 94% full" in line


def test_format_step_renders_unknown_as_question_mark():
    """Unknown must not look like zero."""
    line = format_step(StepReport(index=0, running=None))
    assert "? running" in line


def test_kv_pct_is_none_without_data():
    assert StepReport(index=0).kv_pct is None
    assert StepReport(index=0, kv_blocks_used=1, kv_blocks_total=0).kv_pct is None


def test_preemption_event_requires_a_positive_count():
    assert not StepReport(index=0, preempted=0).preemption_event
    assert StepReport(index=0, preempted=2).preemption_event


# --- V1 snapshot ------------------------------------------------------------


class FakeBlockPool:
    num_gpu_blocks = 100

    def get_num_free_blocks(self) -> int:
        return 40


class FakeKVCacheManager:
    block_pool = FakeBlockPool()


class FakeScheduler:
    def __init__(self):
        self.running = [1, 2, 3]
        self.waiting = [4]
        self.kv_cache_manager = FakeKVCacheManager()

    def get_request_counts(self) -> tuple[int, int]:
        return len(self.running), len(self.waiting)


class FakeEngineCore:
    def __init__(self):
        self.scheduler = FakeScheduler()


class FakeInprocClient:
    def __init__(self):
        self.engine_core = FakeEngineCore()


class FakeEngine:
    def __init__(self):
        self.engine_core = FakeInprocClient()


def test_snapshot_reads_scheduler_and_kv_state():
    report = snapshot(FakeEngine(), index=7, elapsed_ms=41.0)
    assert report.running == 3
    assert report.waiting == 1
    assert report.kv_blocks_total == 100
    assert report.kv_blocks_used == 60
    assert report.kv_pct == 60.0
    assert report.elapsed_ms == 41.0


def test_snapshot_falls_back_to_the_queue_lengths():
    """A version without get_request_counts still exposes the queues."""

    class OldScheduler:
        running = [1]
        waiting = []

    class Core:
        scheduler = OldScheduler()

    class Engine:
        engine_core = Core()

    report = snapshot(Engine(), index=0)
    assert report.running == 1
    assert report.waiting == 0


def test_snapshot_survives_an_engine_without_the_expected_attributes():
    """Internals move between versions; a missing one must not raise."""

    class Bare:
        pass

    report = snapshot(Bare(), index=0)
    assert report.running is None
    assert report.kv_pct is None


def test_snapshot_takes_prefix_and_preemptions_from_the_tap():
    """V1 drains both counters every step, so they come from the stat logger."""
    sample = EngineSample(ts=0.0, prefix_cache_hit_rate=0.42, preemptions=2)
    report = snapshot(FakeEngine(), index=3, sample=sample)
    assert report.prefix_hit_rate == 0.42
    assert report.preempted == 2


def test_require_scheduler_names_the_knob_to_turn():
    class Bare:
        pass

    with pytest.raises(SchedulerUnreachable) as exc:
        require_scheduler(Bare())
    assert "VLLM_ENABLE_V1_MULTIPROCESSING=0" in str(exc.value)


def test_require_scheduler_returns_the_v1_scheduler():
    engine = FakeEngine()
    assert require_scheduler(engine) is engine.engine_core.engine_core.scheduler
