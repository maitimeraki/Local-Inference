"""The scheduler's internal API.

The real app and the real dispatcher, a fake engine underneath: everything
that decides admission, accounting and cancellation is exercised without a
GPU. The engine is attached by hand because ``ASGITransport`` does not run
lifespans, and the lifespan is what loads a real engine.
"""

from __future__ import annotations

import asyncio
import json

import httpx
from infersched.config import SchedulerConfig
from infersched.server import build_app
from test_dispatcher import FakeEngine


def _app(**cfg):
    """The app plus a client pointed at it, with a fake engine attached."""
    app = build_app(SchedulerConfig(**cfg))
    engine = FakeEngine()
    app.state.dispatcher.attach_engine(engine, engine.block_size, engine.max_model_len)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://sched.test"
    )
    return app, client, engine


def _submit(**body) -> dict:
    payload = {
        "request_id": "r1",
        "user_id": "u1",
        "prompt": "hello world",
        "max_tokens": 8,
    }
    payload.update(body)
    return payload


def _frames(text: str) -> list[dict]:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if not payload or payload == "[DONE]":
            continue
        out.append(json.loads(payload))
    return out


# --- admission ---------------------------------------------------------------


def test_a_submit_admits_the_request_and_reserves_its_blocks():
    async def main():
        app, client, engine = _app()
        async with client:
            resp = await client.post("/internal/v1/submit", json=_submit())
        return resp.json(), engine

    result, engine = asyncio.run(main())
    assert result["decision"] == "admit"
    assert result["reserved_blocks"] > 0
    assert len(engine.submitted) == 1


def test_a_duplicate_submit_does_not_start_a_second_engine_request():
    async def main():
        app, client, engine = _app()
        async with client:
            first = await client.post("/internal/v1/submit", json=_submit())
            second = await client.post("/internal/v1/submit", json=_submit())
        return first.json(), second.json(), engine

    first, second, engine = asyncio.run(main())
    assert first["decision"] == "admit"
    assert second["decision"] == "reject"
    assert second["reason"] == "duplicate"
    assert len(engine.submitted) == 1


def test_a_request_that_does_not_fit_is_held():
    async def main():
        app, client, engine = _app(max_in_flight=1)
        async with client:
            await client.post("/internal/v1/submit", json=_submit(request_id="r1"))
            second = await client.post(
                "/internal/v1/submit", json=_submit(request_id="r2")
            )
            state = (await client.get("/internal/v1/state")).json()
        return second.json(), state, engine

    result, state, engine = asyncio.run(main())
    assert result["decision"] == "hold"
    assert result["reason"] == "capacity"
    assert state["held"] == 1
    assert state["inflight"] == 1
    assert "r2" not in engine.submitted


def test_a_held_request_is_dispatched_when_a_slot_frees():
    async def main():
        app, client, engine = _app(max_in_flight=1)
        async with client:
            await client.post("/internal/v1/submit", json=_submit(request_id="r1"))
            await client.post("/internal/v1/submit", json=_submit(request_id="r2"))
            engine.finish("r1", "first")
            state = (await client.get("/internal/v1/state")).json()
        return state, engine

    state, engine = asyncio.run(main())
    assert state["held"] == 0
    assert state["inflight"] == 1
    assert "r2" in engine.submitted


def test_submit_before_an_engine_is_attached_is_refused():
    async def main():
        app = build_app(SchedulerConfig())
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://sched.test"
        ) as client:
            resp = await client.post("/internal/v1/submit", json=_submit())
            health = (await client.get("/health")).json()
        return resp, health

    resp, health = asyncio.run(main())
    assert resp.status_code == 503
    assert resp.json() == {"decision": "reject", "reason": "engine_not_ready"}
    assert health["ready"] is False


# --- accounting --------------------------------------------------------------


def test_the_state_reports_a_consistent_ledger():
    async def main():
        app, client, engine = _app()
        async with client:
            await client.post("/internal/v1/submit", json=_submit())
            # The engine has granted three blocks; the ledger has to agree.
            engine.hold("r1", {1, 2, 3})
            app.state.dispatcher.reconcile()
            state = (await client.get("/internal/v1/state")).json()
        return state

    state = asyncio.run(main())
    assert state["total_blocks"] == 1000
    assert state["free_blocks"] == 997
    assert state["allocated_blocks"] == 3
    assert state["reserved_blocks"] > 0
    assert state["drift_blocks"] == 0
    assert state["per_user"]["u1"]["inflight"] == 1


def test_abort_releases_exactly_the_reserved_blocks():
    async def main():
        app, client, engine = _app()
        async with client:
            admitted = (await client.post("/internal/v1/submit", json=_submit())).json()
            aborted = (
                await client.post("/internal/v1/abort", json={"request_id": "r1"})
            ).json()
            state = (await client.get("/internal/v1/state")).json()
        return admitted, aborted, state, engine

    admitted, aborted, state, engine = asyncio.run(main())
    assert aborted["released_blocks"] == admitted["reserved_blocks"]
    assert state["reserved_blocks"] == 0
    assert state["inflight"] == 0
    assert engine.submitted == {}


def test_aborting_a_held_request_takes_it_out_of_the_queue():
    async def main():
        app, client, engine = _app(max_in_flight=1)
        async with client:
            await client.post("/internal/v1/submit", json=_submit(request_id="r1"))
            await client.post("/internal/v1/submit", json=_submit(request_id="r2"))
            await client.post("/internal/v1/abort", json={"request_id": "r2"})
            state = (await client.get("/internal/v1/state")).json()
        return state

    state = asyncio.run(main())
    assert state["held"] == 0
    assert state["inflight"] == 1


# --- the stream --------------------------------------------------------------


def test_the_stream_carries_the_output():
    async def main():
        app, client, engine = _app()
        async with client:
            await client.post("/internal/v1/submit", json=_submit())
            engine.finish("r1", "hello there")
            resp = await client.get("/internal/v1/stream/r1")
        return resp.text

    frames = _frames(asyncio.run(main()))
    deltas = "".join(f.get("delta", "") for f in frames)
    assert deltas == "hello there"
    assert frames[-1]["done"] is True
    assert frames[-1]["finish_reason"] == "stop"


def test_an_unknown_request_id_is_a_404():
    async def main():
        app, client, engine = _app()
        async with client:
            stream = await client.get("/internal/v1/stream/nope")
            state = await client.get("/internal/v1/requests/nope")
        return stream, state

    stream, state = asyncio.run(main())
    assert stream.status_code == 404
    assert state.status_code == 404
