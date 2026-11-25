"""Config loading and the experiment matrix.

The matrix expansion is worth testing: a silently empty axis means an
experiment that runs zero cells and reports success.
"""

from __future__ import annotations

import pytest
from inferbench.config import (
    EngineConfig,
    Experiment,
    WorkloadProfile,
    load_engines,
    load_experiments,
    load_pricing,
    load_workloads,
)
from inferbench.runner import resolve_cell

# The six workload categories from the brief, plus the wiring-check profile.
EXPECTED_WORKLOADS = {
    "smoke",
    "interactive_chat",
    "long_context_analysis",
    "summarization",
    "code_generation",
    "concurrent_chat",
    "repeated_system_prompts",
}


def test_real_workloads_load():
    profiles = load_workloads()
    assert EXPECTED_WORKLOADS <= set(profiles)
    assert profiles["repeated_system_prompts"].prefix_mode == "shared"


def test_real_engine_configs_load():
    engines = load_engines()
    assert {"baseline", "optimized"} <= set(engines)
    assert engines["baseline"].enable_prefix_caching is False
    assert engines["optimized"].enable_prefix_caching is True


def test_real_experiments_load_and_cover_the_brief():
    experiments = {e.id: e for e in load_experiments()}
    for required in (
        "conc_scale",
        "prompt_length",
        "output_length",
        "prefix_cache",
        "baseline_vs_optimized",
        "engine_comparison",
    ):
        assert required in experiments, f"missing experiment {required}"


def test_experiment_axis_expands_to_cartesian_product():
    exp = Experiment(
        id="e",
        workload="w",
        server="s",
        axis={"concurrency": [1, 2], "config": ["a", "b"]},
    )
    cells = exp.cells()
    assert len(cells) == 4
    assert {"concurrency": 1, "config": "a"} in cells
    assert {"concurrency": 2, "config": "b"} in cells


def test_empty_axis_yields_one_cell():
    exp = Experiment(id="e", workload="w", server="s", axis={})
    assert exp.cells() == [{}]


def test_engine_kwargs_drops_unset_optionals():
    """vLLM must apply its own default, not receive an explicit null."""
    cfg = EngineConfig(name="x", model="m")
    kwargs = cfg.engine_kwargs()
    assert "max_num_seqs" not in kwargs
    assert "enable_chunked_prefill" not in kwargs

    cfg2 = EngineConfig(name="x", model="m", max_num_seqs=16)
    assert cfg2.engine_kwargs()["max_num_seqs"] == 16


def test_engine_kwargs_merges_extra():
    cfg = EngineConfig(name="x", model="m", extra={"swap_space": 2})
    assert cfg.engine_kwargs()["swap_space"] == 2


def test_pricing_loads_with_a_usable_basis():
    pricing = load_pricing()
    assert pricing.gpu_hourly_usd > 0


# --- cell resolution --------------------------------------------------------


@pytest.fixture
def matrix():
    workloads = {
        "short": WorkloadProfile(name="short", input_tokens=100, output_tokens=10),
        "long": WorkloadProfile(name="long", input_tokens=900, output_tokens=10),
    }
    engines = {
        "base": EngineConfig(name="base", model="m"),
        "opt": EngineConfig(name="opt", model="m", enable_prefix_caching=True),
    }
    return workloads, engines


def test_resolve_cell_uses_experiment_defaults(matrix):
    workloads, engines = matrix
    exp = Experiment(id="e", workload="short", server="base", axis={})
    wl, eng, conc = resolve_cell(workloads, engines, exp, {})
    assert (wl.name, eng.name, conc) == ("short", "base", 1)


def test_resolve_cell_applies_all_overrides(matrix):
    """Each of experiments 1-6 rides on one of these three overrides."""
    workloads, engines = matrix
    exp = Experiment(id="e", workload="short", server="base", axis={})
    wl, eng, conc = resolve_cell(
        workloads, engines, exp, {"workload": "long", "config": "opt", "concurrency": 8}
    )
    assert (wl.name, eng.name, conc) == ("long", "opt", 8)


def test_resolve_cell_switching_to_shared_derives_a_prefix_length(matrix):
    """Experiment 4 must work on a workload that was not authored for it."""
    workloads, engines = matrix
    exp = Experiment(id="e", workload="short", server="base", axis={})
    wl, _, _ = resolve_cell(workloads, engines, exp, {"prefix_mode": "shared"})
    assert wl.prefix_mode == "shared"
    assert wl.shared_prefix_tokens > 0


def test_resolve_cell_switching_to_unique_clears_the_prefix(matrix):
    workloads, engines = matrix
    exp = Experiment(id="e", workload="short", server="base", axis={})
    wl, _, _ = resolve_cell(workloads, engines, exp, {"prefix_mode": "unique"})
    assert wl.prefix_mode == "unique"
    assert wl.shared_prefix_tokens == 0
