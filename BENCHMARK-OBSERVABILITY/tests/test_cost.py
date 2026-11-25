"""Cost model arithmetic.

Small numbers, easy to get wrong by a factor of 1000, and the answer feeds the
platform's headline recommendation — so they get tests.
"""

from __future__ import annotations

import pytest
from inferbench.config import PricingConfig
from inferbench.cost import (
    RunCost,
    cloud_equivalent_usd,
    cost_per_1k_requests,
    cost_per_million_output_tokens,
    cost_per_successful_request,
    gpu_cost_usd,
)


def test_gpu_cost_is_linear_in_hours():
    assert gpu_cost_usd(3600, 1.0) == pytest.approx(1.0)
    assert gpu_cost_usd(1800, 1.0) == pytest.approx(0.5)
    assert gpu_cost_usd(0, 1.0) == 0.0


def test_gpu_cost_ignores_negative_duration():
    assert gpu_cost_usd(-5, 1.0) == 0.0


def test_cost_per_million_output_tokens():
    # One hour at $1/h producing 1M tokens costs exactly $1 per Mtok.
    assert cost_per_million_output_tokens(3600, 1_000_000, 1.0) == pytest.approx(1.0)
    # Half a million tokens in the same hour doubles the per-token cost.
    assert cost_per_million_output_tokens(3600, 500_000, 1.0) == pytest.approx(2.0)


def test_cost_per_mtok_is_none_when_nothing_generated():
    """None, not zero: a free result for a broken run is a lie."""
    assert cost_per_million_output_tokens(3600, 0, 1.0) is None


def test_cost_per_1k_requests():
    assert cost_per_1k_requests(3600, 1000, 1.0) == pytest.approx(1.0)
    assert cost_per_1k_requests(3600, 500, 1.0) == pytest.approx(2.0)
    assert cost_per_1k_requests(3600, 0, 1.0) is None


def test_cost_per_successful_request_penalises_failures():
    """Failed requests still rent the GPU; the honest divisor is successes."""
    all_ok = cost_per_successful_request(3600, 1000, 1.0)
    half_ok = cost_per_successful_request(3600, 500, 1.0)
    assert half_ok == pytest.approx(all_ok * 2)
    assert cost_per_successful_request(3600, 0, 1.0) is None


def test_cloud_equivalent_is_none_without_rates():
    pricing = PricingConfig()
    assert cloud_equivalent_usd(pricing, 1_000_000, 1_000_000) is None


def test_cloud_equivalent_with_rates():
    pricing = PricingConfig(cloud_input_per_mtok_usd=3.0, cloud_output_per_mtok_usd=15.0)
    assert cloud_equivalent_usd(pricing, 1_000_000, 1_000_000) == pytest.approx(18.0)


def test_run_cost_summary_shape():
    cost = RunCost(
        duration_s=3600, output_tokens=1_000_000, input_tokens=10, n_requests=1000, n_ok=1000
    )
    summary = cost.summary(PricingConfig(gpu_hourly_usd=1.0))

    assert summary["cost_per_mtok_output"]["mean"] == pytest.approx(1.0)
    assert summary["cost_per_1k_requests"]["mean"] == pytest.approx(1.0)
    assert summary["cost_usd"]["mean"] == pytest.approx(1.0)
    assert summary["gpu_seconds"]["mean"] == pytest.approx(3600.0)


def test_run_cost_summary_omits_undefined_metrics():
    cost = RunCost(duration_s=10, output_tokens=0, input_tokens=0, n_requests=0, n_ok=0)
    summary = cost.summary(PricingConfig())
    assert "cost_per_mtok_output" not in summary
    assert "cost_per_success" not in summary
    assert "cost_usd" in summary
