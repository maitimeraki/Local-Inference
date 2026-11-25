"""Typed configuration.

All knobs live in YAML under `config/` and are validated here on load. Nothing
else in the codebase reads `os.environ` for configuration — a typo'd key should
fail loudly at startup, not silently produce a different experiment.
"""

from __future__ import annotations

import itertools
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"

# Pydantic ignores unknown fields by default, so a typo'd key is dropped in
# silence and the experiment quietly runs with different settings than the
# file says. This module's own contract is that a typo fails loudly, so
# unknown keys are refused. A key the model does not name but vLLM does goes
# under `extra:`, which is merged into the engine's kwargs verbatim.
_STRICT = ConfigDict(extra="forbid")


def config_dir() -> Path:
    """`config/` at the workspace root, overridable for tests."""
    override = os.environ.get("INFERBENCH_CONFIG_DIR")
    return Path(override) if override else _DEFAULT_CONFIG_DIR


def load_yaml(path: str | Path) -> Any:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


# --- Workloads --------------------------------------------------------------


class WorkloadProfile(BaseModel):
    """One experimental workload category.

    These are categories for controlled experiments, not claims about real
    traffic. Realistic distributions come later, from application traces.
    """
    model_config = _STRICT

    name: str
    input_tokens: int
    output_tokens: int
    purpose: str = ""
    requests: int = 32
    # "unique" = every request gets its own prompt; "shared" = all requests
    # share a common prefix of `shared_prefix_tokens`, which is what makes
    # prefix caching observable.
    prefix_mode: Literal["unique", "shared"] = "unique"
    shared_prefix_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def load_workloads(path: str | Path | None = None) -> dict[str, WorkloadProfile]:
    raw = load_yaml(path or config_dir() / "workloads.yaml")
    profiles = [WorkloadProfile(**p) for p in raw["profiles"]]
    return {p.name: p for p in profiles}


# --- Engine configuration ---------------------------------------------------


class EngineConfig(BaseModel):
    """vLLM EngineArgs expressed as data.

    `extra` is the escape hatch for flags this model does not name yet; it is
    merged verbatim into the EngineArgs kwargs.
    """

    name: str
    model: str
    dtype: str = "auto"
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.80
    enable_prefix_caching: bool = False
    enable_chunked_prefill: bool | None = None
    max_num_seqs: int | None = None
    # How the engine orders requests when it runs out of room. "priority"
    # serves the smallest `priority` number first and preempts the largest
    # one last; "fcfs" is arrival order. It is a named field rather than an
    # `extra` key because a silently-dropped key here is an engine that
    # quietly keeps the old behaviour.
    scheduling_policy: Literal["fcfs", "priority"] = "fcfs"
    enforce_eager: bool = True
    extra: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""

    model_config = _STRICT

    def engine_kwargs(self) -> dict[str, Any]:
        """Flatten to AsyncEngineArgs keyword arguments.

        Keys with `None` are dropped so vLLM applies its own default rather
        than receiving an explicit null.
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "dtype": self.dtype,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "enable_prefix_caching": self.enable_prefix_caching,
            "enforce_eager": self.enforce_eager,
            "scheduling_policy": self.scheduling_policy,
        }
        if self.enable_chunked_prefill is not None:
            kwargs["enable_chunked_prefill"] = self.enable_chunked_prefill
        if self.max_num_seqs is not None:
            kwargs["max_num_seqs"] = self.max_num_seqs
        kwargs.update(self.extra)
        return kwargs


def load_engines(path: str | Path | None = None) -> dict[str, EngineConfig]:
    d = Path(path) if path else config_dir() / "engine"
    out: dict[str, EngineConfig] = {}
    for f in sorted(d.glob("*.yaml")):
        cfg = EngineConfig(**load_yaml(f))
        out[cfg.name] = cfg
    return out


# --- Sampling ---------------------------------------------------------------


class SamplingConfig(BaseModel):
    """Fixed across an entire comparison — this is what makes runs comparable."""

    temperature: float = 0.0
    top_p: float = 1.0
    seed: int | None = 0
    ignore_eos: bool = True


# --- Experiments ------------------------------------------------------------


class Experiment(BaseModel):
    """An experiment is an axis spec; the runner expands the matrix.
    model_config = _STRICT

    Adding experiment 7 is a YAML edit, not a new module.
    """
    model_config = _STRICT

    id: str
    workload: str
    server: str
    axis: dict[str, list[Any]]
    engine: str = "vllm"  # sglang becomes a second value here, not a refactor
    repeats: int = 3
    description: str = ""

    def cells(self) -> list[dict[str, Any]]:
        """Cartesian product of the axis, one dict per cell."""
        if not self.axis:
            return [{}]
        keys = list(self.axis)
        return [
            dict(zip(keys, combo, strict=True))
            for combo in itertools.product(*(self.axis[k] for k in keys))
        ]


def load_experiments(path: str | Path | None = None) -> list[Experiment]:
    raw = load_yaml(path or config_dir() / "experiments.yaml")
    return [Experiment(**e) for e in raw["experiments"]]


# --- Pricing / cost ---------------------------------------------------------


class PricingConfig(BaseModel):
    model_config = _STRICT
    gpu_hourly_usd: float = 0.30
    instance_label: str = "rtx-3050-laptop-6gb"
    cloud_input_per_mtok_usd: float | None = None
    cloud_output_per_mtok_usd: float | None = None
    notes: str = ""


def load_pricing(path: str | Path | None = None) -> PricingConfig:
    return PricingConfig(**load_yaml(path or config_dir() / "pricing.yaml"))
