"""Gateway configuration.

Owned by the gateway, not the benchmark harness: the gateway is a service that
must start without the harness present. `config/gateway.yaml` is the shared
file, loaded here.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

# A typo'd key is dropped by pydantic by default, so the gateway would quietly
# run with different limits than the file says. Refuse it instead.
_STRICT = ConfigDict(extra="forbid")

_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"


def config_dir() -> Path:
    override = os.environ.get("INFERBENCH_CONFIG_DIR")
    return Path(override) if override else _DEFAULT_CONFIG_DIR


class Backend(BaseModel):
    name: str
    base_url: str
    weight: float = 1.0
    healthy: bool = True


class GatewayConfig(BaseModel):
    model_config = _STRICT
    backends: list[Backend] = Field(default_factory=list)
    max_in_flight: int = 32
    request_timeout_s: float = 120.0
    max_retries: int = 1
    cache_enabled: bool = True
    cache_max_entries: int = 2048
    cache_ttl_s: float = 600.0
    strategy: Literal["least_outstanding", "weighted"] = "least_outstanding"
    # Tier 2. Empty means "no scheduler": the gateway proxies to the
    # OpenAI-compatible backends above instead. Set it and the scheduler
    # becomes the only upstream — it is the tier that owns the GPU.
    scheduler_url: str = ""
    scheduler_timeout_s: float = 300.0
    require_user_id: bool = False
    idempotency_ttl_s: float = 900.0
    max_in_flight_per_user: int = 4


def load_gateway(path: str | Path | None = None) -> GatewayConfig:
    p = Path(path) if path else config_dir() / "gateway.yaml"
    if not p.exists():
        raise FileNotFoundError(f"gateway config not found: {p}")
    cfg = GatewayConfig(**yaml.safe_load(p.read_text(encoding="utf-8")))
    return _from_env(cfg)


def _from_env(cfg: GatewayConfig) -> GatewayConfig:
    """Env wins over the file, so a moved port cannot desync the tiers.

    `INFERGW_SCHEDULER_URL` set to an empty string means "no scheduler": the
    gateway proxies straight to the backends, which is the A/B benchmarking
    path. `INFERGW_BACKEND_URL` moves the healthy backend instead.
    """
    if "INFERGW_SCHEDULER_URL" in os.environ:
        cfg = cfg.model_copy(update={"scheduler_url": os.environ["INFERGW_SCHEDULER_URL"]})
    if url := os.environ.get("INFERGW_BACKEND_URL"):
        cfg = cfg.model_copy(
            update={
                "backends": [
                    b.model_copy(update={"base_url": url}) if b.healthy else b
                    for b in cfg.backends
                ]
            }
        )
    return cfg
