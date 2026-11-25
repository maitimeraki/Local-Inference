"""SLO evaluation and error-budget burn rate."""

from __future__ import annotations

from pathlib import Path

import pytest
from inferobs.burn import (
    WindowBurn,
    breach_fraction,
    budget_fraction,
    burn_rate,
    time_to_exhaustion_hours,
    worst_window,
)
from inferobs.slo import SLOTarget, evaluate, load_slo_targets, summary_from_store_rows

CONFIG = Path(__file__).resolve().parents[1] / "config" / "slo_targets.yaml"


def _target(**kw) -> SLOTarget:
    base = {
        "id": "t",
        "measurement": "ttft",
        "stat": "p95",
        "threshold": 0.5,
        "comparison": "<=",
    }
    return SLOTarget(**{**base, **kw})


# --- loading ----------------------------------------------------------------


def test_real_targets_load():
    targets = load_slo_targets(CONFIG)
    assert len(targets) >= 4
    assert {"ttft_p95_interactive", "error_rate"} <= {t.id for t in targets}


def test_missing_file_is_an_empty_list_not_an_error():
    assert load_slo_targets("nope.yaml") == []


# --- evaluation -------------------------------------------------------------


def test_check_respects_the_comparison_direction():
    assert _target().check(0.4)
    assert _target().check(0.5)
    assert not _target().check(0.6)

    floor = _target(measurement="output_tps", stat="mean", threshold=10.0, comparison=">=")
    assert floor.check(10.0)
    assert not floor.check(9.9)


def test_evaluate_pass_and_fail():
    results = evaluate([_target()], {"ttft": {"p95": 0.3}})
    assert results[0].measured and results[0].compliant
    assert results[0].render().startswith("[PASS]")

    results = evaluate([_target()], {"ttft": {"p95": 9.0}})
    assert results[0].measured and not results[0].compliant
    assert results[0].render().startswith("[FAIL]")


def test_unmeasured_target_is_skipped_never_passed():
    """An SLO nobody measured is unknown. Rendering unknown as PASS is how a
    broken pipeline looks healthy."""
    result = evaluate([_target()], {})[0]
    assert not result.measured
    assert not result.compliant
    assert result.render().startswith("[SKIP]")


def test_evaluate_handles_several_targets():
    targets = [_target(), _target(id="u", measurement="error_rate", stat="mean", threshold=0.01)]
    results = evaluate(targets, {"ttft": {"p95": 0.1}, "error_rate": {"mean": 0.5}})
    assert [r.compliant for r in results] == [True, False]


def test_summary_from_store_rows():
    rows = [
        {"metric": "ttft", "stat": "p95", "value": 0.31},
        {"metric": "e2e", "stat": "p99", "value": 2.0},
    ]
    summary = summary_from_store_rows(rows)
    assert summary["ttft"]["p95"] == 0.31
    assert summary["e2e"]["p99"] == 2.0


# --- burn rate --------------------------------------------------------------


def test_budget_fraction():
    assert budget_fraction(0.95) == pytest.approx(0.05)
    assert budget_fraction(0.99) == pytest.approx(0.01)
    assert budget_fraction(0.9) == pytest.approx(0.1)


def test_budget_fraction_rejects_impossible_targets():
    with pytest.raises(ValueError):
        budget_fraction(1.5)


def test_breach_fraction():
    assert breach_fraction([0.1, 0.2, 0.9], 0.5, "<=") == pytest.approx(1 / 3)
    assert breach_fraction([], 0.5, "<=") == 0.0


def test_burn_rate_against_a_95_percent_objective():
    # 2 of 10 breach; budget is 5%; burn = 0.2 / 0.05 = 4
    values = [0.1] * 8 + [0.9, 0.9]
    assert burn_rate(values, 0.5, "<=", 0.95) == pytest.approx(4.0)


def test_burn_rate_is_zero_when_nothing_breaches():
    assert burn_rate([0.1] * 10, 0.5, "<=") == 0.0


def test_burn_rate_for_a_floor_objective():
    # output_tps must stay above 10; 5 of 10 fall below.
    values = [20.0] * 5 + [5.0] * 5
    assert burn_rate(values, 10.0, ">=", 0.95) == pytest.approx(10.0)


def test_worst_window_picks_the_highest_burn():
    # 5m breaches 2/10 -> 4x burn; 1h breaches 4/10 -> 8x. Both breach, so the
    # test is really "does it pick the worse one".
    windows = {"5m": [0.1] * 8 + [0.9] * 2, "1h": [0.1] * 6 + [0.9] * 4}
    assert worst_window(windows, 0.5, "<=") == WindowBurn(window="1h", rate=8.0)


def test_worst_window_with_no_windows():
    assert worst_window({}, 0.5, "<=") == WindowBurn(window="none", rate=0.0)


def test_time_to_exhaustion():
    assert time_to_exhaustion_hours(2.0, 24.0) == pytest.approx(12.0)
    assert time_to_exhaustion_hours(0.0, 24.0) == float("inf")
