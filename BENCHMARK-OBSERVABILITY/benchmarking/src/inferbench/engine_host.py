"""The engine host: one place that owns the vLLM engine and knows how to read it.

The OpenAI-compatible server (`inferbench.server`) and the production scheduler
(`infersched`) both need the same four things from the engine — the scheduler
object, the KV-cache block pool, the block size, and the per-request block
counts. Keeping that knowledge here means a vLLM upgrade breaks in one file
instead of three.

Every read is defensive. vLLM's internals move between releases, and a missing
attribute has to yield `None` rather than `0`: "we could not measure this" and
"this was zero" are different facts, and a scheduler that confuses them
over-commits a KV cache it cannot see.

Access paths, verified against vLLM 0.31.x:

  scheduler      engine.engine_core.engine_core.scheduler
  block pool     scheduler.kv_cache_manager.block_pool
  free blocks    block_pool.get_num_free_blocks()
  total blocks   block_pool.num_gpu_blocks, less the block the pool withholds
  block size     scheduler.block_size
  per request    scheduler.kv_cache_manager.coordinator
                   .single_type_managers[*].req_to_blocks[request_id]
  block ids      the same map, read as KVCacheBlock.block_id
"""

from __future__ import annotations

from typing import Any

from inferobs.metric_names import (
    ENGINE_ITERATION_SECONDS,
    ENGINE_KV_CACHE_USAGE,
    ENGINE_PREEMPTIONS_TOTAL,
    ENGINE_PREFIX_CACHE_HIT_RATE,
    ENGINE_REQUESTS_TOTAL,
    ENGINE_RUNNING_REQUESTS,
    ENGINE_SCHEDULED_TOKENS_TOTAL,
    ENGINE_WAITING_REQUESTS,
)
from prometheus_client import Counter, Gauge, Histogram

from inferbench import engine as engine_mod
from inferbench.config import EngineConfig
from inferbench.instrument import EngineSample, TapStatLogger, attach

_METRICS: dict = {}


def metrics() -> dict:
    """Create the Prometheus instruments once per process.

    Guarded because the module can be imported more than once (tests, reloads)
    and prometheus_client raises on duplicate registration.
    """
    if _METRICS:
        return _METRICS
    _METRICS.update(
        running=Gauge(ENGINE_RUNNING_REQUESTS, "Sequences currently decoding"),
        waiting=Gauge(ENGINE_WAITING_REQUESTS, "Requests admitted but not yet scheduled"),
        kv_usage=Gauge(ENGINE_KV_CACHE_USAGE, "Fraction of KV-cache blocks in use"),
        prefix_hit=Gauge(ENGINE_PREFIX_CACHE_HIT_RATE, "Prefix-cache hit rate"),
        iteration=Histogram(
            ENGINE_ITERATION_SECONDS,
            "Duration of one scheduler iteration",
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
        ),
        preemptions=Counter(ENGINE_PREEMPTIONS_TOTAL, "Sequences preempted for memory"),
        scheduled=Counter(ENGINE_SCHEDULED_TOKENS_TOTAL, "Tokens scheduled", ["phase"]),
        requests=Counter(ENGINE_REQUESTS_TOTAL, "Requests handled by the engine", ["status"]),
    )
    return _METRICS


class PrometheusSink:
    """Translates engine samples into Prometheus updates, in-process."""

    def __init__(self) -> None:
        self.m = metrics()
        self.preemptions = 0

    def emit(self, s: EngineSample) -> None:
        if s.running is not None:
            self.m["running"].set(s.running)
        if s.waiting is not None:
            self.m["waiting"].set(s.waiting)
        if s.kv_cache_usage is not None:
            self.m["kv_usage"].set(s.kv_cache_usage)
        if s.prefix_cache_hit_rate is not None:
            self.m["prefix_hit"].set(s.prefix_cache_hit_rate)
        if s.elapsed_ms is not None:
            self.m["iteration"].observe(s.elapsed_ms / 1000.0)
        if s.preemptions:
            self.preemptions += s.preemptions
            self.m["preemptions"].inc(s.preemptions)


class SchedulerUnreachable(RuntimeError):
    """The engine core is not in this process, so its scheduler cannot be read."""


# --- readers ------------------------------------------------------------------


def find_scheduler(engine) -> Any | None:
    """Find the V1 scheduler on a running engine.

    `LLMEngine.engine_core` is an `InprocClient` whose `.engine_core` is the
    `EngineCore` that owns the scheduler. With V1 multiprocessing on, the core
    lives in a child process and nothing in here can be read.
    """
    core = getattr(engine, "engine_core", None)
    for candidate in (core, getattr(core, "engine_core", None), engine):
        scheduler = getattr(candidate, "scheduler", None)
        if scheduler is not None:
            return scheduler
    return None


def require_scheduler(engine) -> Any:
    """Resolve the scheduler, or say plainly why it cannot be read."""
    scheduler = find_scheduler(engine)
    if scheduler is None:
        raise SchedulerUnreachable(
            "Could not reach the V1 scheduler from this process. The engine core "
            "has to run in-process: set VLLM_ENABLE_V1_MULTIPROCESSING=0 before "
            "vLLM is imported (inferbench.engine.configure_env does exactly that)."
        )
    return scheduler


def request_counts(sched) -> tuple[int | None, int | None]:
    """(running, waiting) request counts, as V1 counts them."""
    counts = getattr(sched, "get_request_counts", None)
    if counts is not None:
        running, waiting = counts()
        return running, waiting
    return _len(getattr(sched, "running", None)), _len(getattr(sched, "waiting", None))


def block_pool(sched) -> Any | None:
    """V1's KV-cache block pool."""
    return getattr(getattr(sched, "kv_cache_manager", None), "block_pool", None)


def free_blocks(sched) -> int | None:
    """Blocks still available, as the engine sees them right now."""
    pool = block_pool(sched)
    if pool is None:
        return None
    free = getattr(pool, "get_num_free_blocks", None)
    if free is None:
        return None
    return free() if callable(free) else free


def withheld_blocks(sched) -> int:
    """Blocks the pool permanently withholds from allocation.

    V1's `BlockPool` pops one block off the free queue at construction to be
    the placeholder (`block_id=0`) that the attention backend pads with. Its
    ref count is not maintained and it never returns to the pool, so it is
    capacity nobody can be promised — counting it as free would let the
    scheduler promise a block the engine will never hand out.
    """
    pool = block_pool(sched)
    if pool is None:
        return 0
    return 1 if getattr(pool, "null_block", None) is not None else 0


def total_blocks(sched) -> int | None:
    """Blocks the scheduler may promise: the pool minus what it withholds."""
    pool = block_pool(sched)
    if pool is None:
        return None
    raw = getattr(pool, "num_gpu_blocks", None)
    if not isinstance(raw, int):
        return None
    return raw - withheld_blocks(sched)


def kv_blocks(sched) -> tuple[int | None, int | None]:
    """(blocks_used, blocks_total) from V1's block pool."""
    total = total_blocks(sched)
    free = free_blocks(sched)
    if total is None or free is None:
        return None, None
    return total - free, total


def request_blocks(sched, request_id: str) -> int | None:
    """Blocks currently held by one request, or None if that cannot be read.

    Read through the coordinator's per-type managers, which own the
    `req_to_blocks` map. A request with no entry holds zero blocks.
    """
    coord = getattr(getattr(sched, "kv_cache_manager", None), "coordinator", None)
    if coord is None:
        return None
    # `single_type_managers` is the 0.31 name; `managers` the older one.
    managers = getattr(coord, "single_type_managers", None) or getattr(coord, "managers", None)
    if not managers:
        return None
    for manager in managers:
        blocks = getattr(manager, "req_to_blocks", None)
        if blocks is None:
            continue
        held = blocks.get(request_id)
        if held:
            return len(held)
    return 0


def request_block_ids(sched, request_id: str) -> set[int] | None:
    """The ids of the KV blocks one request holds, or None if unreadable.

    The *ids* matter, not the count. A prefix-cache hit puts the same blocks
    into the hit request's `req_to_blocks`, so two live requests can hold one
    block between them. Summing per-request counts double-counts that; the
    union of ids does not. The reservation ledger's drift metric is only
    meaningful over a union.
    """
    coord = getattr(getattr(sched, "kv_cache_manager", None), "coordinator", None)
    if coord is None:
        return None
    managers = getattr(coord, "single_type_managers", None) or getattr(coord, "managers", None)
    if not managers:
        return None
    ids: set[int] = set()
    for manager in managers:
        blocks = getattr(manager, "req_to_blocks", None)
        if blocks is None:
            continue
        for held in blocks.get(request_id) or ():
            block_id = getattr(held, "block_id", None)
            if isinstance(block_id, int):
                ids.add(block_id)
    return ids


def _vllm_config(engine) -> Any | None:
    """The engine's resolved `VllmConfig`, wherever it is reachable from."""
    core = getattr(engine, "engine_core", None)
    for candidate in (engine, core, getattr(core, "engine_core", None)):
        cfg = getattr(candidate, "vllm_config", None)
        if cfg is not None:
            return cfg
    return None


def block_size(engine) -> int | None:
    """The KV-cache block size, in tokens.

    `scheduler.block_size` is the scheduling granularity; for a single-group
    model it is exactly the KV group block size (16 tokens by default).

    ponytail: assumes one KV group. A hybrid model (Mamba alongside
    attention) has per-group block sizes — read
    `coordinator.single_type_managers[*].block_size` and map request -> group
    when one is deployed.
    """
    sched = find_scheduler(engine)
    size = getattr(sched, "block_size", None) if sched is not None else None
    if isinstance(size, int) and size > 0:
        return size
    cfg = _vllm_config(engine)
    size = getattr(getattr(cfg, "cache_config", None), "block_size", None)
    return size if isinstance(size, int) and size > 0 else None


def max_model_len(engine) -> int | None:
    """The engine's context limit, in tokens."""
    cfg = _vllm_config(engine)
    value = getattr(getattr(cfg, "model_config", None), "max_model_len", None)
    return value if isinstance(value, int) and value > 0 else None


def _len(value) -> int | None:
    try:
        return len(value)
    except TypeError:
        return None


# --- the host -----------------------------------------------------------------


class EngineHost:
    """The engine, the tap on it, and typed access to its internals.

    One instance per process. The engine is not shareable across processes, so
    neither is this.
    """

    def __init__(self, engine: Any, model_name: str | None = None, vllm: Any = None) -> None:
        self.engine = engine
        self.model_name = model_name
        self.vllm = vllm
        self.sink = PrometheusSink()
        self.tap = TapStatLogger(self.sink)
        self.tap_attached = attach(engine, self.tap)

    @classmethod
    def start(cls, cfg: EngineConfig) -> EngineHost:
        """Build the serving engine and attach the tap. Loads the model."""
        vllm = engine_mod.require_vllm()
        engine = engine_mod.build_async_engine(cfg)
        return cls(engine, model_name=cfg.model, vllm=vllm)

    # -- readers ---------------------------------------------------------------

    def scheduler(self) -> Any | None:
        return find_scheduler(self.engine)

    def request_counts(self) -> tuple[int | None, int | None]:
        sched = self.scheduler()
        if sched is None:
            return None, None
        return request_counts(sched)

    def free_blocks(self) -> int | None:
        sched = self.scheduler()
        return None if sched is None else free_blocks(sched)

    def total_blocks(self) -> int | None:
        sched = self.scheduler()
        return None if sched is None else total_blocks(sched)

    def request_blocks(self, request_id: str) -> int | None:
        sched = self.scheduler()
        return None if sched is None else request_blocks(sched, request_id)

    def request_block_ids(self, request_id: str) -> set[int] | None:
        sched = self.scheduler()
        return None if sched is None else request_block_ids(sched, request_id)

    def preemptions(self) -> int:
        """Preemptions the tap has seen since the engine started."""
        return self.sink.preemptions

    def block_size(self) -> int | None:
        return block_size(self.engine)

    def max_model_len(self) -> int | None:
        return max_model_len(self.engine)
