"""The CLI has to be able to fail.

`bench` is the smoke test for the whole wiring: if the gateway points at a
port with nothing on it, every request fails. If that still exits 0, the
failure is invisible until someone reads the report and finds a 0%-success
run ranked second. So `bench` exits non-zero when a request failed, and
`report` exits non-zero when no recorded run qualified for the ranking.
"""

from __future__ import annotations

import inferbench.runner as R
from inferbench import metrics as M
from inferbench.cli import build_parser, cmd_bench, cmd_report
from inferbench.store import RunMeta, Store

EXP = "smoke"


def _record(store: Store, run_id: str, ok: bool) -> None:
    store.start_run(
        RunMeta(
            run_id=run_id,
            experiment_id=EXP,
            workload="smoke",
            config="baseline",
            concurrency=1,
        )
    )
    request = (
        M.RequestMetrics(request_id="r0", ttft_s=0.1, e2e_s=1.0, output_tokens=5)
        if ok
        else M.RequestMetrics(request_id="r0", status="timeout", error="boom")
    )
    store.add_requests(run_id, [request])
    store.finish_run(run_id, 1.0)


def _bench(monkeypatch, tmp_path, ok: bool) -> int:
    async def fake_run_experiment(store, exp, *args, **kwargs):
        _record(store, "run-1", ok)
        return ["run-1"]

    monkeypatch.setattr(R, "load_tokenizer", lambda model: object())
    monkeypatch.setattr(R, "run_experiment", fake_run_experiment)

    args = build_parser().parse_args(
        ["--db", str(tmp_path / "b.db"), "bench", "--experiment", EXP, "--pushgateway", ""]
    )
    return cmd_bench(args)


def test_bench_exits_zero_when_every_request_succeeded(monkeypatch, tmp_path):
    assert _bench(monkeypatch, tmp_path, ok=True) == 0


def test_bench_exits_nonzero_when_a_request_failed(monkeypatch, tmp_path):
    assert _bench(monkeypatch, tmp_path, ok=False) == 1


def _report(tmp_path, ok: bool) -> int:
    db = tmp_path / "b.db"
    store = Store(db)
    _record(store, "run-1", ok)
    store.close()

    args = build_parser().parse_args(
        [
            "--db",
            str(db),
            "report",
            "--experiment",
            EXP,
            "--out",
            str(tmp_path / "report.md"),
        ]
    )
    return cmd_report(args)


def test_report_exits_zero_when_a_run_qualified(tmp_path):
    assert _report(tmp_path, ok=True) == 0


def test_report_exits_nonzero_when_no_run_qualified(tmp_path):
    assert _report(tmp_path, ok=False) == 1
