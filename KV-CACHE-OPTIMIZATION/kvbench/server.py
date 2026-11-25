"""The engine driver: one vLLM engine, driven asynchronously, timed.

Both arms go through this module and differ only in the model they load and
the KV page they charge for. The budget is the same for both, so the arm
with the smaller page gets more blocks and serves more sequences -- which is
the question the benchmark asks.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

from . import config, models, weights


@dataclass(frozen=True)
class EngineConfig:
    """Which arm to run, and what it is allowed to spend."""

    arm: str
    rank: int | None = None
    max_model_len: int = config.ENGINE_FLAGS["max_model_len"]
    kv_budget_bytes: int = config.KV_BUDGET_BYTES
    gpu_memory_utilization: float = config.ENGINE_FLAGS["gpu_memory_utilization"]
    block_size: int = 16
    max_num_seqs: int | None = None
    """The engine's batch cap. Fidelity runs pin it to 1: the batch
    composition perturbs the reduction order, and a token-level diff against
    another arm only means something if both arms ran alone."""


@dataclass(frozen=True)
class Request:
    """One generation: the prompt, how much to generate, and how to sample."""

    request_id: str
    prompt: str
    output_tokens: int
    sampling_params: SamplingParams


@dataclass
class LoadResult:
    """The run's timings and rates."""

    prefill_tok_per_s: float = 0.0
    """Prompt tokens divided by the run's wall time."""

    decode_tok_per_s: float = 0.0
    """Output tokens divided by the run's wall time: the aggregate rate
    every stream shared."""

    decode_tok_per_s_per_seq: list[float] = field(default_factory=list)
    """The request's own output rate."""

    ttft_ms: list[float] = field(default_factory=list)
    tpot_ms: list[float] = field(default_factory=list)
    """Time per output token after the first."""

    e2e_ms: list[float] = field(default_factory=list)

    preemptions: int = 0
    """How many requests the engine had to preempt and recompute."""

    resident: int = 0
    """How many requests the engine held in the cache at the same time.

    A request holds its blocks from the moment it is scheduled until it
    finishes, so the overlap of those intervals is the number of requests
    the cache actually held -- the measured answer to "how many requests
    does this cache serve".
    """

    intervals: list[tuple[float, float]] = field(default_factory=list)
    """(scheduled, last token) per request, in engine-core time."""

    wall_s: float = 0.0
    """How long the run took."""

    output_tokens: int = 0
    """How many tokens the run generated in total."""


def page_size_bytes(cfg: EngineConfig) -> int:
    """The bytes one block costs, per layer.

    GQA holds two kv heads of 128 dims, for both the key and the value, in
    bfloat16. The MLA arm holds one latent of r values and no separate
    value.
    """
    if cfg.arm in ("gqa", "gqa_norope"):
        per_token = config.NUM_KV_HEADS * config.HEAD_DIM * 2 * 2
    else:
        per_token = cfg.rank * 2
    return cfg.block_size * per_token

def blocks_for(cfg: EngineConfig) -> int:
    """How many blocks the budget buys.

    The budget covers the whole model, and a block is allocated per layer,
    so the per-layer page is charged once per layer.
    """
    return cfg.kv_budget_bytes // (page_size_bytes(cfg) * config.NUM_LAYERS)


def base_model_dir() -> Path:
    """The pinned model's weights, on disk."""
    if config.MODEL_DIR.is_dir():
        return config.MODEL_DIR
    repo = "models--" + config.MODEL_ID.replace("/", "--")
    snapshots = sorted(
        (Path.home() / ".cache/huggingface/hub" / repo / "snapshots").glob("*")
    )
    if not snapshots:
        raise FileNotFoundError(f"{config.MODEL_ID} is not in the HF cache")
    return snapshots[0]


def mla_model_dir(rank: int) -> Path:
    """The rank-r MLA checkpoint: the base model with its K/V projections
    replaced by the latent stack. Built once, then reused."""
    out = config.MLA_WEIGHTS_DIR / f"r{rank}"
    if (out / "model.safetensors").is_file():
        return out

    base = base_model_dir()
    out.mkdir(parents=True, exist_ok=True)
    built = weights.build_mla_weights(base, rank=rank, out_dir=out)
    built.rename(out / "model.safetensors")
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "generation_config.json",
    ):
        if (base / name).is_file():
            shutil.copy(base / name, out / name)

    model_config = json.loads((base / "config.json").read_text())
    model_config["architectures"] = ["Qwen2MLAForCausalLM"]
    (out / "config.json").write_text(json.dumps(model_config, indent=2))
    return out


def norope_model_dir() -> Path:
    """The base model's weights under a name that does not rotate.

    The weights are the same file -- only the architecture in config.json
    differs, and that is what picks the model class.
    """
    out = config.MLA_WEIGHTS_DIR / "norope"
    if (out / "config.json").is_file():
        return out

    base = base_model_dir()
    out.mkdir(parents=True, exist_ok=True)
    for name in ("model.safetensors", "model.safetensors.index.json"):
        if (base / name).is_file():
            (out / name).symlink_to(base / name)
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "generation_config.json",
    ):
        if (base / name).is_file():
            shutil.copy(base / name, out / name)

    model_config = json.loads((base / "config.json").read_text())
    model_config["architectures"] = ["Qwen2NoRopeForCausalLM"]
    (out / "config.json").write_text(json.dumps(model_config, indent=2))
    return out


def build_engine(cfg: EngineConfig) -> AsyncLLMEngine:
    """The engine for one arm, with the KV budget spent in blocks."""
    kwargs = dict(config.ENGINE_FLAGS)
    kwargs.update(
        max_model_len=cfg.max_model_len,
        # Chunked prefill is off, so a prompt is prefilled in one piece.
        max_num_batched_tokens=cfg.max_model_len,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        block_size=cfg.block_size,
        num_gpu_blocks_override=blocks_for(cfg),
    )
    if cfg.max_num_seqs is not None:
        kwargs["max_num_seqs"] = cfg.max_num_seqs
    if cfg.arm == "mla":
        models.register_plugins()
        kwargs["model"] = str(mla_model_dir(cfg.rank))
        kwargs["hf_overrides"] = {"mla_rank": cfg.rank}
    elif cfg.arm == "gqa_norope":
        models.register_plugins()
        kwargs["model"] = str(norope_model_dir())
    else:
        kwargs["model"] = str(base_model_dir())

    return AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**kwargs))


def make_requests(prompts, output_tokens: int) -> list[Request]:
    """One request per prompt, with distinct ids."""
    params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=output_tokens, n=1)
    return [
        Request(
            request_id=f"req-{i}",
            prompt=prompt,
            output_tokens=output_tokens,
            sampling_params=params,
        )
        for i, prompt in enumerate(prompts)
    ]


async def run_load(engine: AsyncLLMEngine, requests, concurrency: int) -> LoadResult:
    """Run the requests through the engine, `concurrency` at a time.

    The next request starts as soon as one finishes, so the engine stays
    loaded for the whole run.
    """
    result = LoadResult()
    queue = list(requests)
    started = time.perf_counter()
    streamed = 0  # output tokens the engine has produced so far
    prompt_tokens = 0
    preempted: set[str] = set()
    intervals: dict[str, tuple[float, float]] = {}

    async def one(req: Request) -> None:
        nonlocal streamed
        nonlocal prompt_tokens
        t0 = time.perf_counter()
        first = None
        last = 0
        prompt_len = 0
        async for out in engine.generate(
            req.prompt, req.sampling_params, req.request_id
        ):
            now = time.perf_counter()
            if first is None:
                first = now
            tokens = len(out.outputs[0].token_ids)
            streamed += tokens - last
            last = tokens
            prompt_len = len(out.prompt_token_ids)
            metrics = out.metrics
            if metrics is not None:
                if metrics.num_preemptions:
                    preempted.add(req.request_id)
                if metrics.scheduled_ts and metrics.last_token_ts:
                    # the generator yields one output per token, so this
                    # is overwritten until the request's last token
                    intervals[req.request_id] = (
                        metrics.scheduled_ts,
                        metrics.last_token_ts,
                    )
        end = time.perf_counter()

        # the request's own clock: the driver's queue is not the engine's
        # latency, but the engine's own queue is.
        decode_s = max(end - first, 1e-9)
        result.ttft_ms.append((first - t0) * 1e3)
        result.tpot_ms.append((end - first) * 1e3 / max(last - 1, 1))
        result.e2e_ms.append((end - t0) * 1e3)
        result.decode_tok_per_s_per_seq.append(last / decode_s)
        prompt_tokens += prompt_len

    async def worker() -> None:
        while queue:
            await one(queue.pop(0))

    await asyncio.gather(*(worker() for _ in range(concurrency)))
    wall = max(time.perf_counter() - started, 1e-9)
    result.preemptions = len(preempted)
    result.intervals = list(intervals.values())
    result.resident = _peak_overlap(result.intervals)
    # The aggregate rates are the run's: every token the engine produced
    # divided by the run's wall time. A per-request rate cannot be summed
    # into an aggregate, and a running counter over elapsed time is not a
    # rate at all.
    result.wall_s = wall
    result.output_tokens = streamed
    result.decode_tok_per_s = streamed / wall
    result.prefill_tok_per_s = prompt_tokens / wall
    return result


def _peak_overlap(intervals: list[tuple[float, float]]) -> int:
    """How many intervals overlap at the busiest moment."""
    if not intervals:
        return 0
    edges = []
    for start, end in intervals:
        edges.append((start, 1))
        edges.append((end, -1))
    # ends before starts at the same instant: a request that finished frees
    # its blocks for the one that starts
    edges.sort(key=lambda e: (e[0], e[1]))
    peak = current = 0
    for _, delta in edges:
        current += delta
        peak = max(peak, current)
    return peak


async def greedy_outputs(engine: AsyncLLMEngine, prompts, max_tokens: int) -> list[list[int]]:
    """The token ids each prompt generates, greedily."""
    params = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=max_tokens, n=1, seed=0
    )

    async def one(index: int, prompt: str) -> list[int]:
        ids: list[int] = []
        async for out in engine.generate(prompt, params, f"greedy-{index}"):
            ids = list(out.outputs[0].token_ids)
        return ids

    return list(
        await asyncio.gather(*(one(i, p) for i, p in enumerate(prompts)))
    )


CONCURRENCY_LADDER = (1, 2, 4, 8, 16, 32, 64, 128)


def kill_orphan_engines() -> None:
    """Kill any engine core that outlived its driver.

    The engine core runs in its own process. If the driver is killed --
    an interrupt, a crash -- the core survives and keeps the card's
    memory, and the next run cannot start. This finds those by name and
    kills them.
    """
    import subprocess

    try:
        listing = subprocess.run(
            ["ps", "-eo", "pid,args"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except Exception:  # noqa: BLE001
        return

    mine = os.getpid()
    for line in listing.splitlines():
        fields = line.split(None, 1)
        if len(fields) != 2:
            continue
        pid, args = fields
        if "VLLM::EngineCore" not in args:
            continue
        if not pid.isdigit() or int(pid) == mine:
            continue
        try:
            os.kill(int(pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            continue


def prompts_of_length(tokenizer, count: int, tokens: int) -> list[str]:
    """`count` prompts of exactly `tokens` tokens each.

    The measurement is at a fixed context length, so the workload has to
    be: a short prompt would fit many more sequences than the context the
    row claims, and the requests-served number would not mean anything.
    """
    unit = "The quick brown fox jumps over the lazy dog. "
    unit_tokens = len(tokenizer(unit)["input_ids"])
    body = unit * (tokens // unit_tokens + 2)
    body_ids = tokenizer(body)["input_ids"][:tokens]
    text = tokenizer.decode(body_ids)
    return [f"Request {i}. {text}" for i in range(count)]


async def find_requests_served(
    engine: AsyncLLMEngine,
    prompts,
    output_tokens: int,
    ladder=CONCURRENCY_LADDER,
    ceiling: int | None = None,
) -> int:
    """How many requests the cache serves at once.

    The plan's section 4.2: a rung serves if the cache holds every request
    in it at the same time, and the ceiling is the last rung that serves.
    The engine does not refuse the overflow -- it queues it -- so the
    signal is what the engine actually held: the peak number of requests
    whose KV blocks were resident at once.

    `ceiling` is the arithmetic ideal, the most the budget could hold.
    The ladder stops one rung above it: a rung beyond the ideal cannot
    fit whatever the budget says, so running it only measures the queue.

    The prompts are cycled, so a rung of 128 gets 128 requests even if the
    pool is shorter: the pool is a workload, not the limit.
    """
    served = 0
    for rung in ladder:
        if not prompts:
            break
        if ceiling is not None and rung > ceiling:
            break
        rung_prompts = [prompts[i % len(prompts)] for i in range(rung)]
        requests = make_requests(rung_prompts, output_tokens)
        try:
            result = await run_load(engine, requests, rung)
        except Exception:
            break
        if result.preemptions:
            break
        # the cache held fewer than the rung asked for: the overflow was
        # queued, so this rung does not fit
        if result.resident < rung:
            break
        served = rung
    return served
