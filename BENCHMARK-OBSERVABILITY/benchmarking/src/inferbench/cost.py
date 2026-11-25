"""Cost model.

The brief's actual question is which configuration gives the best balance of
throughput, latency, memory *and cost*. Throughput and latency are measured;
cost is arithmetic on top of them, and the expensive mistake is assuming the
fastest configuration is the cheapest. It usually is not, once idle GPU time is
counted.
"""

from __future__ import annotations

from dataclasses import dataclass

from inferbench.config import PricingConfig

SECONDS_PER_HOUR = 3600.0


def gpu_cost_usd(duration_s: float, gpu_hourly_usd: float) -> float:
    """Straight rental cost of the GPU for the wall-clock duration.

    Charges the whole wall clock, including the tail where the GPU is idle
    waiting for the last request — that idle time is real money.
    """
    return max(duration_s, 0.0) / SECONDS_PER_HOUR * gpu_hourly_usd


def cost_per_million_output_tokens(
    duration_s: float, output_tokens: int, gpu_hourly_usd: float
) -> float | None:
    """USD per 1M generated tokens. None when nothing was generated."""
    if output_tokens <= 0:
        return None
    return gpu_cost_usd(duration_s, gpu_hourly_usd) / (output_tokens / 1_000_000)


def cost_per_1k_requests(
    duration_s: float, n_requests: int, gpu_hourly_usd: float
) -> float | None:
    if n_requests <= 0:
        return None
    return gpu_cost_usd(duration_s, gpu_hourly_usd) / (n_requests / 1000)


def cost_per_successful_request(
    duration_s: float, n_ok: int, gpu_hourly_usd: float
) -> float | None:
    """The honest per-request figure: failed requests still cost GPU time."""
    if n_ok <= 0:
        return None
    return gpu_cost_usd(duration_s, gpu_hourly_usd) / n_ok


def cloud_equivalent_usd(
    pricing: PricingConfig, input_tokens: int, output_tokens: int
) -> float | None:
    """What the same tokens would cost at a hosted per-token rate.

    Only meaningful when `pricing.yaml` carries cloud rates; None otherwise
    rather than a fabricated comparison.
    """
    if pricing.cloud_input_per_mtok_usd is None or pricing.cloud_output_per_mtok_usd is None:
        return None
    return (
        input_tokens / 1_000_000 * pricing.cloud_input_per_mtok_usd
        + output_tokens / 1_000_000 * pricing.cloud_output_per_mtok_usd
    )


@dataclass
class RunCost:
    duration_s: float
    output_tokens: int
    input_tokens: int
    n_requests: int
    n_ok: int

    def summary(self, pricing: PricingConfig) -> dict[str, dict[str, float]]:
        """Cost metrics in the same shape the store expects for other metrics."""
        out: dict[str, dict[str, float]] = {}
        gpu = pricing.gpu_hourly_usd

        def put(name: str, value: float | None) -> None:
            if value is not None:
                out[name] = {"mean": value}

        put(
            "cost_per_mtok_output",
            cost_per_million_output_tokens(self.duration_s, self.output_tokens, gpu),
        )
        put("cost_per_1k_requests", cost_per_1k_requests(self.duration_s, self.n_requests, gpu))
        put("cost_per_success", cost_per_successful_request(self.duration_s, self.n_ok, gpu))
        put("cost_usd", gpu_cost_usd(self.duration_s, gpu))
        put("gpu_seconds", self.duration_s)
        return out
