"""Scheduler configuration, read from `config/scheduler.yaml`.

Same directory as the engine and gateway configs, so one place describes the
whole stack.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict

# A typo'd key is dropped by pydantic by default, so the scheduler would
# quietly run with different limits than the file says. Refuse it instead.
_STRICT = ConfigDict(extra="forbid")

# `strict` reserves prompt *and* the worst-case output, so the sum of all
# promises stays within the cache and the engine never has to preempt.
# `prompt_only` reserves the prompt alone and lets the engine preempt if it
# outgrows that.
# `adaptive` reserves the prompt plus the *measured* p95 output, so the
# promise follows what the model actually produces. With no measurement yet
# it falls back to `strict`, so the never-preempt guarantee holds until the
# data says otherwise.
ReservationMode = Literal["strict", "prompt_only", "adaptive"]

# Only used when nothing else says how many sequences the engine can run: an
# engine config without `max_num_seqs` and a scheduler config without an
# explicit ceiling. Small on purpose — over-promising is what preempts.
DEFAULT_MAX_IN_FLIGHT = 32

_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"


def config_dir() -> Path:
    override = os.environ.get("INFERBENCH_CONFIG_DIR")
    return Path(override) if override else _DEFAULT_CONFIG_DIR


class SchedulerConfig(BaseModel):
    """The knobs an operator turns. Everything else is read from the engine."""

    engine: str = "optimized"
    reservation_mode: ReservationMode = "strict"
    # None means "take the engine's max_num_seqs". A hand-copied number here
    # is how the scheduler's ceiling and the engine's capacity drift apart
    # silently, so the engine is the source of truth. Set it only to lower
    # the derived number.
    max_in_flight: int | None = None
    max_in_flight_per_user: int = 4
    max_queue_depth_global: int = 256
    # Same number as config/scheduler.yaml. These two drift apart silently
    # when one changes and the other does not, and the smaller one wins
    # without anybody noticing which file it came from.
    max_queue_depth_per_user: int = 64
    max_reserved_blocks_per_user: int = 512
    host: str = "0.0.0.0"
    port: int = 9001


def resolve_max_in_flight(configured: int | None, engine: Any) -> int:
    """The scheduler's ceiling on concurrent engine requests.
    model_config = _STRICT

    The engine's own `max_num_seqs` is the source of truth: promising more
    concurrent requests than the engine can run is how a queue turns into
    preemption. A configured value may only lower it.
    """
    engine_limit = getattr(getattr(engine, "cfg", None), "max_num_seqs", None)
    if not isinstance(engine_limit, int) or engine_limit < 1:
        engine_limit = None
    if configured is None:
        return engine_limit if engine_limit is not None else DEFAULT_MAX_IN_FLIGHT
    if engine_limit is None:
        return configured
    return min(configured, engine_limit)


def load_scheduler(path: str | Path | None = None) -> SchedulerConfig:
    p = Path(path) if path else config_dir() / "scheduler.yaml"
    if not p.exists():
        raise FileNotFoundError(f"scheduler config not found: {p}")
    cfg = SchedulerConfig(**(yaml.safe_load(p.read_text(encoding="utf-8")) or {}))
    if port := os.environ.get("INFERSCHED_PORT"):
        cfg = cfg.model_copy(update={"port": int(port)})
    return cfg
