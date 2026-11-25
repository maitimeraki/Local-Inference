"""Step-through engine anatomist.

Runs the engine one decoding iteration at a time and prints what the scheduler
did on each. This is the mode that justifies importing vLLM instead of
containerising it: nobody running a container has watched `Scheduler.schedule()`
hand out KV-cache blocks.

Targets the **V1** engine, whose scheduler hangs off the engine core:
`engine.engine_core.engine_core.scheduler`. vLLM runs that core in a child
process by default, so the inspector forces `VLLM_ENABLE_V1_MULTIPROCESSING=0`
— in-process, the scheduler is a plain Python object we can read.

`format_step` is pure and lives apart from the engine so the output can be
tested without a GPU.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from inferbench.config import EngineConfig, SamplingConfig
from inferbench.engine_host import (
    SchedulerUnreachable,
    find_scheduler,
    kv_blocks,
    request_counts,
    require_scheduler,
)
from inferbench.instrument import CollectingSink, EngineSample, TapStatLogger, attach

__all__ = [
    "SchedulerUnreachable",
    "StepReport",
    "format_step",
    "require_scheduler",
    "run_inspection",
    "snapshot",
]


@dataclass
class StepReport:
    """One scheduler step, as far as we could read it.

    Every field is optional because vLLM's internals are version-dependent.
    `None` means "this version does not expose it", which is different from
    zero and must not be rendered as zero.
    """

    index: int
    running: int | None = None
    waiting: int | None = None
    tokens_this_step: int | None = None
    kv_blocks_used: int | None = None
    kv_blocks_total: int | None = None
    prefix_hit_rate: float | None = None
    preempted: int | None = None
    finished: int | None = None
    elapsed_ms: float | None = None

    @property
    def kv_pct(self) -> float | None:
        if self.kv_blocks_used is None or not self.kv_blocks_total:
            return None
        return 100.0 * self.kv_blocks_used / self.kv_blocks_total

    @property
    def preemption_event(self) -> bool:
        return bool(self.preempted)


def format_step(r: StepReport) -> str:
    """One line per step, plus a marker line when something was preempted."""
    sched = f"{_n(r.running)} running"
    if r.waiting:
        sched = f"{_n(r.waiting)} new + {sched}"

    parts = [f"step {r.index:04d}", f"sched: {sched}"]
    if r.tokens_this_step is not None:
        parts.append(f"tok {r.tokens_this_step}")
    if r.kv_pct is not None:
        parts.append(f"kv {r.kv_pct:.0f}% ({r.kv_blocks_used}/{r.kv_blocks_total} blk)")
    if r.prefix_hit_rate is not None:
        parts.append(f"prefix {r.prefix_hit_rate * 100:.0f}%")
    if r.elapsed_ms is not None:
        parts.append(f"{r.elapsed_ms:.0f}ms")

    line = " | ".join(parts)
    if r.preemption_event:
        line += f"\n{' ' * 8}↳ preempted {r.preempted} (kv {_pct(r.kv_pct)} full)"
    return line


def _n(value: int | None) -> str:
    return "?" if value is None else str(value)


def _pct(value: float | None) -> str:
    return "?" if value is None else f"{value:.0f}%"


def snapshot(
    engine,
    index: int,
    *,
    elapsed_ms: float | None = None,
    finished: int | None = None,
    sample: EngineSample | None = None,
) -> StepReport:
    """Read scheduler and KV-cache state off a running V1 engine.

    The scheduler is reached through the engine core. Every access below is
    defensive because the internals have moved between vLLM releases. The
    prefix rate and the preemption count are not on the scheduler at all —
    V1 drains them every step — so they come from the stat-logger tap.
    """
    report = StepReport(index=index, elapsed_ms=elapsed_ms, finished=finished)
    sched = find_scheduler(engine)
    if sched is None:
        return report

    report.running, report.waiting = request_counts(sched)
    if report.running is not None:
        # Decode emits one token per running sequence per step; prefill steps
        # move more, which is why this is a lower bound rather than a measure.
        report.tokens_this_step = report.running

    report.kv_blocks_used, report.kv_blocks_total = kv_blocks(sched)

    if sample is not None:
        report.prefix_hit_rate = sample.prefix_cache_hit_rate
        report.preempted = sample.preemptions

    return report


def run_inspection(
    cfg: EngineConfig,
    prompt: str,
    max_tokens: int = 64,
    *,
    sampling: SamplingConfig | None = None,
    step_limit: int = 500,
    emit=print,
) -> list[StepReport]:
    """Drive the V1 engine one step at a time, emitting a line per step.

    Returns the collected reports so a caller can persist them. Raises
    `engine.VllmNotInstalled` when vLLM is unavailable, which is the honest
    outcome on a machine without the engine.
    """
    from inferbench import engine as engine_mod

    # The scheduler is only readable while the engine core is in this process.
    engine_mod.configure_env(force=True)
    vllm = engine_mod.require_vllm()

    from vllm import SamplingParams

    sampling = sampling or SamplingConfig()
    engine_args = vllm.EngineArgs(**cfg.engine_kwargs())
    llm_engine = vllm.LLMEngine.from_engine_args(engine_args)  # type: ignore[attr-defined]

    sink = CollectingSink()
    tap = TapStatLogger(sink)
    if not attach(llm_engine, tap):
        raise RuntimeError(
            "Could not attach the stat-logger tap to the engine: "
            "engine.logger_manager.stat_loggers was not found. The prefix-cache "
            "hit rate and the preemption counts would be missing from every step."
        )
    require_scheduler(llm_engine)

    llm_engine.add_request(
        "inspect-0",
        prompt,
        SamplingParams(
            max_tokens=max_tokens,
            min_tokens=max_tokens,
            ignore_eos=sampling.ignore_eos,
            temperature=sampling.temperature,
            seed=sampling.seed,
        ),
    )

    reports: list[StepReport] = []
    index = 0

    while llm_engine.has_unfinished_requests() and index < step_limit:
        t0 = time.perf_counter()
        outputs = llm_engine.step()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        finished = sum(1 for o in outputs if getattr(o, "finished", False))
        report = snapshot(
            llm_engine,
            index,
            elapsed_ms=elapsed_ms,
            finished=finished,
            sample=sink.samples[-1] if sink.samples else None,
        )
        reports.append(report)
        emit(format_step(report))
        index += 1

    return reports
