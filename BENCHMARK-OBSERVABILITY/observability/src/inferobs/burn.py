"""Error-budget burn rate.

An objective like "p95 TTFT < 0.5s" implies a *budget*: at most 5% of requests
are allowed to breach. Burn rate is how fast that budget is being spent:

    burn = observed_breach_fraction / allowed_breach_fraction

  burn < 1  spending slower than budget — fine
  burn == 1 on pace to consume the whole budget over the window
  burn > 1  will exhaust the budget early — this is what should page

A multi-window burn rate (fast 5m/1h + slow 6h/3d) is the production norm;
`burn_rate` computes one window, and `worst_window` picks the highest across the
windows you pass in.
"""

from __future__ import annotations

from dataclasses import dataclass

from inferobs.slo import Comparison

# Budget fractions implied by common objectives.
_TARGET_TO_BUDGET = {0.95: 0.05, 0.99: 0.01, 0.999: 0.001}

_VIOLATES = {
    "<=": lambda a, b: a > b,
    "<": lambda a, b: a >= b,
    ">=": lambda a, b: a < b,
    ">": lambda a, b: a <= b,
}


def budget_fraction(target_ratio: float) -> float:
    """Allowed breach fraction for an objective like 0.95 -> 0.05."""
    if target_ratio in _TARGET_TO_BUDGET:
        return _TARGET_TO_BUDGET[target_ratio]
    if not 0 < target_ratio < 1:
        raise ValueError(f"target_ratio must be in (0, 1), got {target_ratio}")
    return 1.0 - target_ratio


def breach_fraction(values: list[float], threshold: float, comparison: Comparison) -> float:
    """Fraction of `values` that fail the objective. Empty input is 0.0."""
    if not values:
        return 0.0
    violates = _VIOLATES[comparison]
    return sum(1 for v in values if violates(v, threshold)) / len(values)


def burn_rate(
    values: list[float],
    threshold: float,
    comparison: Comparison,
    target_ratio: float = 0.95,
) -> float:
    """Burn rate over one window. 0.0 when nothing breaches."""
    budget = budget_fraction(target_ratio)
    return breach_fraction(values, threshold, comparison) / budget


@dataclass
class WindowBurn:
    window: str
    rate: float


def worst_window(
    windows: dict[str, list[float]],
    threshold: float,
    comparison: Comparison,
    target_ratio: float = 0.95,
) -> WindowBurn:
    """Highest burn rate across named windows — the one that should alert."""
    if not windows:
        return WindowBurn(window="none", rate=0.0)
    burns = [
        WindowBurn(window=name, rate=burn_rate(vals, threshold, comparison, target_ratio))
        for name, vals in windows.items()
    ]
    return max(burns, key=lambda b: b.rate)


def time_to_exhaustion_hours(rate: float, window_hours: float) -> float:
    """How long the budget lasts at this burn rate. `inf` when rate <= 0."""
    if rate <= 0:
        return float("inf")
    return window_hours / rate
