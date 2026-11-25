"""Persistence and aggregation.

The store is the record of truth — pushgateway and the report are both views of
it. If a row or an aggregate is wrong here, it is wrong everywhere.
"""

from __future__ import annotations

import pytest
from inferbench import metrics as M
from inferbench.store import (
    RunMeta,
    Store,
    aggregate_requests,
    failure_summary,
    throughput,
)


def _store(tmp_path) -> Store:
    return Store(tmp_path / "bench.db")


def _ok(request_id: str = "r1", ttft: float = 0.1, out_tokens: int = 10) -> M.RequestMetrics:
    return M.RequestMetrics(
        request_id=request_id,
        ttft_s=ttft,
        e2e_s=ttft + 1.0,
        tpot_s=0.1,
        itl_mean_s=0.05,
        input_tokens=100,
        output_tokens=out_tokens,
        started_at=1.0,
        finished_at=2.0,
    )


def _failed(request_id: str = "r2") -> M.RequestMetrics:
    return M.RequestMetrics(
        request_id=request_id, status=M.STATUS_TIMEOUT, error="timed out"
    )


def test_run_round_trip(tmp_path):
    store = _store(tmp_path)
    meta = RunMeta(
        run_id="abc123", experiment_id="e1", workload="w", config="baseline", concurrency=8
    )

    store.start_run(meta)
    store.add_requests("abc123", [_ok("r1"), _ok("r2"), _failed("r3")])
    store.save_summary("abc123", {"ttft": {"p95": 0.3}})
    store.finish_run("abc123", duration_s=12.5)

    run = store.runs("e1")[0]
    assert run["run_id"] == "abc123"
    assert run["n_requests"] == 3
    assert run["n_ok"] == 2
    assert run["duration_s"] == 12.5

    assert store.summary_for("abc123")["ttft"]["p95"] == 0.3
    assert len(store.requests_for("abc123")) == 3
    store.close()


def test_meta_serialises_structured_fields(tmp_path):
    store = _store(tmp_path)
    meta = RunMeta(
        run_id="m1",
        sampling={"temperature": 0.0},
        engine_flags={"max_model_len": 4096},
        label={"concurrency": 8, "repeat": 0},
    )
    store.start_run(meta)
    run = store.runs()[0]
    # Stored as JSON text; parseable back.
    import json

    assert json.loads(run["sampling"])["temperature"] == 0.0
    assert json.loads(run["engine_flags"])["max_model_len"] == 4096
    store.close()


def test_aggregate_ignores_nulls_rather_than_zeroing_them():
    rows = [
        {"ttft_s": 0.1},
        {"ttft_s": 0.2},
        {"ttft_s": None},  # a failed request has no TTFT
    ]
    agg = aggregate_requests(rows)
    assert agg["ttft"]["mean"] == pytest.approx(0.15)
    assert agg["ttft"]["max"] == pytest.approx(0.2)


def test_aggregate_omits_metrics_with_no_values():
    agg = aggregate_requests([{"ttft_s": None, "e2e_s": 1.0}])
    assert "ttft" not in agg
    assert agg["e2e"]["mean"] == 1.0


def test_aggregate_empty_input():
    assert aggregate_requests([]) == {}


def test_throughput_counts_only_successful_requests():
    rows = [
        {"ok": True, "status": "ok", "output_tokens": 100, "input_tokens": 10},
        {"ok": False, "status": "timeout", "output_tokens": 0, "input_tokens": 10},
    ]
    rates = throughput(rows, duration_s=10.0)
    assert rates["output_tps"]["mean"] == 10.0
    assert rates["input_tps"]["mean"] == 2.0
    # A failed request returns in microseconds, so counting it would make a
    # run that served nothing look faster than one that actually worked.
    assert rates["request_tps"]["mean"] == 0.1


def test_throughput_guards_zero_duration():
    assert throughput([{"output_tokens": 1}], duration_s=0) == {}


def test_failure_summary_counts_by_status():
    rows = [
        {"ok": True, "status": "ok"},
        {"ok": False, "status": "timeout"},
        {"ok": False, "status": "timeout"},
        {"ok": False, "status": "http_error"},
    ]
    assert failure_summary(rows) == {"timeout": 2, "http_error": 1}


def test_failure_summary_empty_when_all_succeed():
    assert failure_summary([{"ok": True, "status": "ok"}]) == {}


def test_save_summary_replaces_rather_than_accumulates(tmp_path):
    store = _store(tmp_path)
    store.start_run(RunMeta(run_id="s1"))
    store.save_summary("s1", {"ttft": {"p95": 0.5}})
    store.save_summary("s1", {"ttft": {"p95": 0.2}})
    assert store.summary_for("s1")["ttft"]["p95"] == 0.2
    store.close()
