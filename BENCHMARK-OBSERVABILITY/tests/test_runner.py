"""The runner must survive being driven the way the CLI drives it.

`run_experiment` is a coroutine and `cmd_bench` runs it with a single
`asyncio.run`. If anything inside `run_experiment` calls `asyncio.run` again —
say, to fire the batch — the first cell raises

    RuntimeError: asyncio.run() cannot be called from a running event loop

That failure needs no GPU, no vLLM and no server, so it belongs here rather
than in the untested GPU path.
"""

from __future__ import annotations

import asyncio

import inferbench.runner as R
from inferbench.config import (
    SamplingConfig,
    load_engines,
    load_experiments,
    load_pricing,
    load_workloads,
)


def test_run_experiment_is_drivable_by_one_asyncio_run(monkeypatch, tokenizer):
    experiments = {e.id: e for e in load_experiments()}
    exp = experiments["smoke"]

    fired: list[tuple[str, str, int, int]] = []

    async def fake_execute_batch(
        base_url, model, payloads, concurrency, sampling, timeout_s=300.0
    ):
        fired.append((base_url, model, len(payloads), concurrency))
        return [], 0.0

    monkeypatch.setattr(R, "execute_batch", fake_execute_batch)
    monkeypatch.setattr(R, "persist_run", lambda *a, **k: "run-0001")

    run_ids = asyncio.run(
        R.run_experiment(
            None,  # store — unused, persist_run is stubbed
            exp,
            load_workloads(),
            load_engines(),
            tokenizer,
            "http://localhost:9000",
            load_pricing(),
            SamplingConfig(),
            pushgateway_url=None,
            progress=lambda *a, **k: None,
        )
    )

    assert run_ids == ["run-0001"]
    # one cell in `smoke`, so exactly one batch, fired at the URL we passed in
    assert len(fired) == 1
    assert fired[0][0] == "http://localhost:9000"
    assert fired[0][3] == 1


def test_every_repeat_asks_the_server_a_different_question(monkeypatch, tokenizer):
    """Repeating a cell must not repeat its prompts.

    The engine caches prefixes, so the second repeat of the same cell would
    be measured against a cache the first repeat warmed — the curve the
    dashboard plots would then be a cache curve.
    """
    experiments = {e.id: e for e in load_experiments()}
    exp = experiments["smoke"].model_copy(update={"repeats": 3})

    fired: list[list[str]] = []
    labels: list[dict] = []

    async def fake_execute_batch(
        base_url, model, payloads, concurrency, sampling, timeout_s=300.0
    ):
        fired.append([p[0] for p in payloads])
        return [], 0.0

    def fake_persist(store, meta, *args, **kwargs):
        labels.append(meta.label)
        return "run-0001"

    monkeypatch.setattr(R, "execute_batch", fake_execute_batch)
    monkeypatch.setattr(R, "persist_run", fake_persist)

    asyncio.run(
        R.run_experiment(
            None,  # store — unused, persist_run is stubbed
            exp,
            load_workloads(),
            load_engines(),
            tokenizer,
            "http://localhost:9000",
            load_pricing(),
            SamplingConfig(),
            pushgateway_url=None,
            progress=lambda *a, **k: None,
        )
    )

    assert len(fired) == 3
    assert not set(fired[0]) & set(fired[1])
    assert not set(fired[1]) & set(fired[2])
    # Each run row says which prompts it used, so a reader can check.
    assert [label["prompt_epoch"] for label in labels] == [0, 1, 2]
