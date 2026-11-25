"""Report rendering, ranking, and the pushgateway payload."""

from __future__ import annotations

from inferbench import metrics as M
from inferbench.report import RunRow, collect_rows, rank, render_markdown, write_report
from inferbench.runner import format_push
from inferbench.store import RunMeta, Store


def _row(config: str, conc: int, ttft_p95: float, out_tps: float, cost: float) -> RunRow:
    return RunRow(
        run_id=f"{config}-{conc}",
        config=config,
        workload="concurrent_chat",
        concurrency=conc,
        repeat=0,
        n_requests=10,
        n_ok=10,
        values={
            "ttft|p95": ttft_p95,
            "output_tps|mean": out_tps,
            "cost_per_mtok_output|mean": cost,
        },
    )


# --- rendering --------------------------------------------------------------


def test_render_markdown_has_headers_and_values():
    text = render_markdown([_row("baseline", 8, 0.301, 42.5, 1.25)])
    assert "| config |" in text
    assert "TTFT p95" in text
    assert "baseline" in text
    assert "0.301" in text


def test_render_markdown_handles_missing_values():
    row = RunRow("r", "baseline", "w", 1, 0, 4, 4, {})
    assert "-" in render_markdown([row])


def test_failure_rate_is_computed_from_counts():
    assert RunRow("r", "c", "w", 1, 0, 4, 3, {}).failure_rate == 0.25
    assert RunRow("r", "c", "w", 1, 0, 0, 0, {}).failure_rate == 0.0


def test_is_clean_requires_every_request_to_have_succeeded():
    assert RunRow("r", "c", "w", 1, 0, 4, 4, {}).is_clean
    assert not RunRow("r", "c", "w", 1, 0, 4, 3, {}).is_clean
    assert not RunRow("r", "c", "w", 1, 0, 0, 0, {}).is_clean


# --- ranking ----------------------------------------------------------------


def test_rank_prefers_the_balanced_configuration():
    rows = [
        _row("fast_expensive", 8, 0.2, 100.0, 2.0),
        _row("slow_cheap", 8, 0.4, 50.0, 1.0),
    ]
    ranked = rank(rows)
    # Higher throughput and lower TTFT outweigh the higher cost here.
    assert ranked[0][0].config == "fast_expensive"
    assert len(ranked) == 2


def test_rank_applies_the_latency_ceiling_first():
    """The fastest configuration is useless if it breaches the interactive
    target."""
    rows = [
        _row("fast_expensive", 8, 0.9, 100.0, 2.0),
        _row("slow_cheap", 8, 0.4, 50.0, 1.0),
    ]
    ranked = rank(rows, max_ttft_p95_s=0.5)
    assert [r.config for r, _ in ranked] == ["slow_cheap"]


def test_rank_returns_empty_when_nothing_qualifies():
    assert rank([_row("a", 8, 5.0, 1.0, 1.0)], max_ttft_p95_s=0.5) == []


def test_rank_of_empty_input():
    assert rank([]) == []


def test_rank_is_stable_when_all_metrics_are_identical():
    rows = [_row("a", 1, 0.5, 10.0, 1.0), _row("b", 1, 0.5, 10.0, 1.0)]
    assert len(rank(rows)) == 2


def test_rank_excludes_runs_with_failed_requests():
    """A run that served nothing has no metrics to compare, and an absent
    metric normalised to 0.5 scores as a middling run rather than a failure."""
    broken = RunRow(
        run_id="broken",
        config="broken",
        workload="concurrent_chat",
        concurrency=8,
        repeat=0,
        n_requests=10,
        n_ok=0,
        values={
            "ttft|p95": 0.1,
            "output_tps|mean": 100.0,
            "cost_per_mtok_output|mean": 0.1,
        },
    )
    ranked = rank([_row("good", 8, 0.3, 50.0, 1.0), broken])
    assert [r.config for r, _ in ranked] == ["good"]


# --- from the store ---------------------------------------------------------


def test_collect_rows_reads_back_what_was_written(tmp_path):
    store = Store(tmp_path / "b.db")
    store.start_run(
        RunMeta(run_id="r1", experiment_id="e", workload="w", config="baseline", concurrency=8)
    )
    store.add_requests(
        "r1", [M.RequestMetrics(request_id="a", ttft_s=0.1, e2e_s=1.0, output_tokens=5)]
    )
    store.save_summary("r1", {"ttft": {"p95": 0.11}})
    store.finish_run("r1", 1.0)

    rows = collect_rows(store, "e")
    assert len(rows) == 1
    assert rows[0].config == "baseline"
    assert rows[0].values["ttft|p95"] == 0.11
    store.close()


def test_write_report_excludes_failed_runs_and_says_so(tmp_path):
    store = Store(tmp_path / "b.db")
    store.start_run(
        RunMeta(run_id="ok", experiment_id="e", workload="w", config="baseline", concurrency=1)
    )
    store.add_requests(
        "ok", [M.RequestMetrics(request_id="a", ttft_s=0.1, e2e_s=1.0, output_tokens=5)]
    )
    store.finish_run("ok", 1.0)
    store.start_run(
        RunMeta(run_id="bad", experiment_id="e", workload="w", config="broken", concurrency=1)
    )
    store.add_requests("bad", [M.RequestMetrics(request_id="b", status="timeout", error="boom")])
    store.finish_run("bad", 1.0)

    text = write_report(store, tmp_path / "r.md", experiment_id="e").read_text(encoding="utf-8")
    store.close()

    assert "1 of 2 runs had failed requests" in text
    assert "**Best balance:** `baseline`" in text
    assert "broken" not in text.split("## Ranking")[1]


# --- pushgateway payload ----------------------------------------------------


def test_format_push_emits_the_expected_series():
    meta = RunMeta(run_id="r1", workload="w", config="baseline", concurrency=8)
    summary = {"ttft": {"p95": 0.31, "mean": 0.12}, "output_tps": {"mean": 42.0}}
    text = format_push(meta, summary, {"timeout": 2}, duration_s=10.0)

    assert '# TYPE bench_metric gauge' in text
    expected = (
        'bench_metric{run_id="r1",workload="w",config="baseline",concurrency="8",'
        'metric="ttft",stat="p95"} 0.31'
    )
    assert expected in text
    assert 'metric="output_tps",stat="mean"' in text
    assert 'bench_failures_total{run_id="r1",reason="timeout"} 2' in text
    assert 'bench_duration_seconds{run_id="r1"} 10.0' in text


def test_format_push_routes_cost_to_its_own_metric():
    meta = RunMeta(run_id="r1", workload="w", config="c", concurrency=1)
    text = format_push(meta, {"cost_per_mtok_output": {"mean": 1.5}}, {}, 1.0)
    assert "bench_cost_usd_per_million_output_tokens{" in text
    assert 'bench_metric{' not in text


def test_format_push_escapes_label_values():
    meta = RunMeta(run_id="r1", workload='we"ird', config="c", concurrency=1)
    text = format_push(meta, {"ttft": {"p95": 0.1}}, {}, 1.0)
    assert 'workload="we\\"ird"' in text
