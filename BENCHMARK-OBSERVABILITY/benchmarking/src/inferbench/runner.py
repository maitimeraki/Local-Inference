"""Execute experiment cells and persist the results.

One cell = one (workload, engine config, axis point, repeat). The runner fires
its requests through the client, samples the GPU alongside, rolls the numbers
up, and writes a run row that records everything needed to trust the result
later.

The throughput figures here are the ones the brief asks about that cannot come
from a single request: tokens/sec and requests/sec are properties of the batch,
measured against wall-clock for the whole run.
"""

from __future__ import annotations

import subprocess
import time

import httpx

from inferbench import metrics as M
from inferbench.client import BenchmarkClient, run_concurrent
from inferbench.collectors import GpuCollector, gpu_names
from inferbench.config import (
    EngineConfig,
    Experiment,
    PricingConfig,
    SamplingConfig,
    WorkloadProfile,
)
from inferbench.cost import RunCost
from inferbench.store import RunMeta, Store, aggregate_requests, failure_summary, throughput
from inferbench.workloads import PromptFactory, Tokenizer


def git_sha() -> str | None:
    """Short SHA of the working tree, for the run record. Best effort."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def load_tokenizer(model: str) -> Tokenizer:
    """HF tokenizer for prompt construction.

    Comes in with vLLM's dependency tree; there is no separate install.
    """
    from transformers import AutoTokenizer  # type: ignore[import-not-found]

    return AutoTokenizer.from_pretrained(model)


async def execute_batch(
    base_url: str,
    model: str,
    payloads: list[tuple[str, int, int | None]],
    concurrency: int,
    sampling: SamplingConfig,
    timeout_s: float = 300.0,
) -> tuple[list[M.RequestMetrics], float]:
    """Fire every payload at `concurrency` and time the whole batch."""
    started = time.perf_counter()
    async with BenchmarkClient(
        base_url, model, timeout_s=timeout_s, sampling=sampling
    ) as client:
        results = await run_concurrent(client, payloads, concurrency)
    return results, time.perf_counter() - started


def persist_run(
    store: Store,
    meta: RunMeta,
    results: list[M.RequestMetrics],
    duration_s: float,
    pricing: PricingConfig,
    *,
    gpu_samples: list[dict] | None = None,
    pushgateway_url: str | None = None,
) -> str:
    """Write one run: requests, aggregates, cost. Returns the run id."""
    rows = [r.to_row() for r in results]

    summary: dict[str, dict[str, float]] = {}
    summary.update(aggregate_requests(rows))
    summary.update(throughput(rows, duration_s))

    n_ok = sum(1 for r in results if r.ok)
    out_tokens = sum(r.output_tokens or 0 for r in results)
    if results:
        # A first-class metric, not a footnote: an SLO has to be able to point
        # at it, and a run with half its requests failing must not look fast.
        summary["error_rate"] = {"mean": 1.0 - n_ok / len(results)}
    in_tokens = sum(r.input_tokens or 0 for r in results)
    summary.update(
        RunCost(
            duration_s=duration_s,
            output_tokens=out_tokens,
            input_tokens=in_tokens,
            n_requests=len(results),
            n_ok=n_ok,
        ).summary(pricing)
    )

    if gpu_samples:
        peaks = [s["mem_used_bytes"] for s in gpu_samples if s.get("mem_used_bytes")]
        if peaks:
            summary["gpu_mem"] = {"max": float(max(peaks))}

    store.start_run(meta)
    store.add_requests(meta.run_id, results)
    store.save_summary(meta.run_id, summary)
    store.finish_run(meta.run_id, duration_s, status="ok" if n_ok else "failed")

    if pushgateway_url:
        push_to_pushgateway(pushgateway_url, meta, summary, failure_summary(rows), duration_s)

    return meta.run_id


def resolve_cell(
    workloads: dict[str, WorkloadProfile],
    engines: dict[str, EngineConfig],
    experiment: Experiment,
    cell: dict,
) -> tuple[WorkloadProfile, EngineConfig, int]:
    """Resolve one axis point into a concrete workload, engine config and
    concurrency.

    A cell may override the experiment's defaults:
      concurrency   experiment 1 (scaling)
      workload      experiments 2 and 3 (prompt/output length)
      prefix_mode   experiment 4 (prefix caching)
      config        experiments 5 and 6 (settings, and engine)
    """
    workload = workloads[cell.get("workload", experiment.workload)]
    mode = cell.get("prefix_mode")
    if mode is not None and mode != workload.prefix_mode:
        # Switching to `shared` needs a prefix length; derive a sane one when
        # the profile does not carry it, so the axis works on any workload.
        shared = workload.shared_prefix_tokens or min(512, max(1, workload.input_tokens // 2))
        workload = workload.model_copy(
            update={
                "prefix_mode": mode,
                "shared_prefix_tokens": shared if mode == "shared" else 0,
            }
        )
    engine_cfg = engines[cell.get("config", experiment.server)]
    concurrency = int(cell.get("concurrency", 1))
    return workload, engine_cfg, concurrency


async def run_experiment(
    store: Store,
    experiment: Experiment,
    workloads: dict[str, WorkloadProfile],
    engines: dict[str, EngineConfig],
    tokenizer: Tokenizer,
    base_url: str,
    pricing: PricingConfig,
    sampling: SamplingConfig,
    *,
    pushgateway_url: str | None = None,
    progress=print,
) -> list[str]:
    """Run every cell of an experiment, repeats included."""
    run_ids: list[str] = []
    # The epoch counts measured runs across the whole experiment. It salts
    # the prompts, so no run ever asks the server a question an earlier run
    # already asked — the second repeat of a cell would otherwise be measured
    # against the prefix cache the first repeat warmed.
    epoch = 0

    for cell in experiment.cells():
        workload, engine_cfg, concurrency = resolve_cell(workloads, engines, experiment, cell)

        for repeat in range(experiment.repeats):
            prompt_epoch = epoch
            epoch += 1

            factory = PromptFactory(tokenizer, workload, salt=prompt_epoch)
            prompts = factory.build_many(workload.requests)
            payloads = [
                (p.text, workload.output_tokens, p.approx_input_tokens) for p in prompts
            ]

            meta = RunMeta(
                experiment_id=experiment.id,
                workload=workload.name,
                config=engine_cfg.name,
                concurrency=concurrency,
                model=engine_cfg.model,
                dtype=engine_cfg.dtype,
                max_model_len=engine_cfg.max_model_len,
                sampling=sampling.model_dump(),
                engine_flags=engine_cfg.engine_kwargs(),
                gpu=", ".join(gpu_names()) or None,
                git_sha=git_sha(),
                label={
                    **cell,
                    "repeat": repeat,
                    "engine": experiment.engine,
                    # Recorded so a run row says which prompts it used.
                    "prompt_epoch": prompt_epoch,
                },
                notes=workload.purpose,
            )

            collector = GpuCollector()
            with collector:
                # Awaited, not asyncio.run: `run_experiment` is itself a
                # coroutine driven by one asyncio.run in the CLI, so a second
                # asyncio.run here dies with "cannot be called from a running
                # event loop" on the first cell.
                results, duration_s = await execute_batch(
                    base_url,
                    engine_cfg.model,
                    payloads,
                    concurrency,
                    sampling,
                )

            run_id = persist_run(
                store,
                meta,
                results,
                duration_s,
                pricing,
                gpu_samples=collector.as_rows(),
                pushgateway_url=pushgateway_url,
            )
            run_ids.append(run_id)
            n_ok = sum(1 for r in results if r.ok)
            progress(
                f"  {run_id}  {workload.name} c={concurrency:<3} repeat={repeat}  "
                f"{n_ok}/{len(results)} ok  {duration_s:.1f}s"
            )

    return run_ids


# --- pushgateway ------------------------------------------------------------


def _esc(value: object) -> str:
    """Escape a Prometheus label value."""
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def format_push(
    meta: RunMeta, summary: dict[str, dict[str, float]], failures: dict[str, int], duration_s: float
) -> str:
    """Exposition text for one run.

    One `bench_metric` with `metric`/`stat` labels rather than dozens of named
    gauges: a new measurement is then a data change, not a code change.
    """
    base = (
        f'run_id="{_esc(meta.run_id)}",workload="{_esc(meta.workload)}",'
        f'config="{_esc(meta.config)}",concurrency="{_esc(meta.concurrency)}"'
    )
    lines = [
        "# TYPE bench_metric gauge",
        "# TYPE bench_cost_usd_per_million_output_tokens gauge",
        "# TYPE bench_failures_total counter",
        "# TYPE bench_duration_seconds gauge",
    ]

    for metric, stats in sorted(summary.items()):
        if metric == "cost_per_mtok_output":
            # Cost carries no `stat` label, so emitting every stat here would
            # write the same series twice — which Prometheus rejects outright.
            value = stats.get("mean")
            if value is not None:
                lines.append(f'bench_cost_usd_per_million_output_tokens{{{base}}} {value}')
            continue
        for stat, value in sorted(stats.items()):
            lines.append(
                f'bench_metric{{{base},metric="{_esc(metric)}",stat="{_esc(stat)}"}} {value}'
            )

    for reason, count in sorted(failures.items()):
        lines.append(
            f'bench_failures_total{{run_id="{_esc(meta.run_id)}",reason="{_esc(reason)}"}} {count}'
        )

    lines.append(f'bench_duration_seconds{{run_id="{_esc(meta.run_id)}"}} {duration_s}')
    return "\n".join(lines) + "\n"


def push_to_pushgateway(
    url: str,
    meta: RunMeta,
    summary: dict[str, dict[str, float]],
    failures: dict[str, int],
    duration_s: float,
    timeout_s: float = 5.0,
) -> bool:
    """Push one run's metrics. Returns whether the push landed.

    A pushgateway that is down must not fail the benchmark — the sqlite store
    is the record of truth, pushgateway is only how Grafana reads it.
    """
    target = f"{url.rstrip('/')}/metrics/job/inferbench/run_id/{meta.run_id}"
    try:
        resp = httpx.put(
            target,
            content=format_push(meta, summary, failures, duration_s),
            headers={"Content-Type": "text/plain"},
            timeout=timeout_s,
        )
        return resp.status_code < 300
    except httpx.HTTPError:
        return False
