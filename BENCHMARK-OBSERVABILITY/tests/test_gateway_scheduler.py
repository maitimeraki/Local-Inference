"""The gateway talking to the scheduler.

One fake scheduler behind one ``MockTransport``, the real gateway app on the
other side of it. What these tests pin down is the contract between the two
tiers: what the gateway forwards, what it does with each decision, and the
promise that a duplicate submit never becomes a second engine request.
"""

from __future__ import annotations

import asyncio
import json

import httpx
from infergw.app import build_app
from infergw.config import Backend, GatewayConfig
from prometheus_client import REGISTRY

DEFAULT_FRAMES = [
    {"delta": "hello"},
    {
        "done": True,
        "finish_reason": "stop",
        "prompt_tokens": 7,
        "completion_tokens": 1,
        "hold_seconds": 0.25,
    },
]


def _frame(payload: dict) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


class FakeScheduler:
    """The scheduler tier, as far as the gateway can see it."""

    def __init__(
        self,
        *,
        decision: str = "admit",
        reason: str = "fit",
        reserved_blocks: int = 11,
        demand: int = 11,
        hold_seq: int = 0,
        stream_status: int = 200,
    ) -> None:
        self.decision = decision
        self.reason = reason
        self.reserved_blocks = reserved_blocks
        self.demand = demand
        self.hold_seq = hold_seq
        self.stream_status = stream_status
        self.frames = list(DEFAULT_FRAMES)
        self.state: dict = {
            "ready": True,
            "inflight": 1,
            "held": 2,
            "free_blocks": 3662,
            "total_blocks": 3662,
        }
        self.submits: list[dict] = []
        self.aborts: list[dict] = []
        # The stream blocks here until the test lets it go.
        self.gate = asyncio.Event()
        self.gate.set()
        self.stream_started = asyncio.Event()

    def block(self) -> None:
        """Hold the stream open until `release()`."""
        self.gate.clear()

    def release(self) -> None:
        self.gate.set()

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/internal/v1/submit":
            self.submits.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "decision": self.decision,
                    "reason": self.reason,
                    "reserved_blocks": self.reserved_blocks,
                    "demand": self.demand,
                    "hold_seq": self.hold_seq,
                },
            )
        if path == "/internal/v1/abort":
            self.aborts.append(json.loads(request.content))
            return httpx.Response(200, json={"released_blocks": 0})
        if path == "/internal/v1/state":
            return httpx.Response(200, json=dict(self.state))
        if path.startswith("/internal/v1/stream/"):
            return self._stream(path.rsplit("/", 1)[-1])
        raise AssertionError(f"unexpected scheduler call: {path}")

    def _stream(self, request_id: str) -> httpx.Response:
        if self.stream_status != 200:
            return httpx.Response(self.stream_status, json={"error": "boom"})

        async def body():
            self.stream_started.set()
            yield _frame({"request_id": request_id, "decision": self.decision, "state": "running"})
            await self.gate.wait()
            for f in self.frames:
                yield _frame(f)

        return httpx.Response(
            200, content=body(), headers={"content-type": "text/event-stream"}
        )


def _gateway(sched: FakeScheduler, **cfg_kw):
    """The app plus a client pointed at it. Caller closes both."""
    cfg = GatewayConfig(
        backends=[Backend(name="b1", base_url="http://engine.test")],
        scheduler_url="http://sched.test",
        require_user_id=True,
        **cfg_kw,
    )
    app = build_app(cfg)
    app.state.client = httpx.AsyncClient(
        transport=httpx.MockTransport(sched.handler), timeout=5.0
    )
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gw.test"
    )
    return app, client


def _counter(name: str) -> float:
    return REGISTRY.get_sample_value(name) or 0.0


# --- forwarding --------------------------------------------------------------


def test_a_request_is_forwarded_to_the_scheduler_with_the_decision_headers():
    sched = FakeScheduler()

    async def main():
        app, client = _gateway(sched)
        async with client:
            resp = await client.post(
                "/v1/completions",
                json={"prompt": "hello", "max_tokens": 8},
                headers={"x-user-id": "u1"},
            )
        await app.state.client.aclose()
        return resp

    resp = asyncio.run(main())
    assert resp.status_code == 200
    assert resp.headers["x-scheduler-decision"] == "admit"
    assert resp.headers["x-scheduler-reason"] == "fit"
    assert resp.headers["x-kv-blocks-reserved"] == "11"
    assert resp.headers["x-kv-blocks-demand"] == "11"
    assert resp.json()["choices"][0]["text"] == "hello"

    assert len(sched.submits) == 1
    submitted = sched.submits[0]
    assert submitted["prompt"] == "hello"
    assert submitted["user_id"] == "u1"
    assert submitted["max_tokens"] == 8


def test_a_held_request_is_waited_on_not_refused():
    """A hold is the scheduler saying "not yet" — the caller waits for it."""
    sched = FakeScheduler(decision="hold", reason="capacity", reserved_blocks=0, hold_seq=3)
    sched.block()

    async def main():
        app, client = _gateway(sched)
        async with client:
            task = asyncio.create_task(
                client.post(
                    "/v1/completions",
                    json={"prompt": "hello", "max_tokens": 8},
                    headers={"x-user-id": "u1"},
                )
            )
            await asyncio.wait_for(sched.stream_started.wait(), timeout=5)
            # Still waiting: a hold did not turn into a 429.
            assert not task.done()
            sched.release()
            resp = await task
        await app.state.client.aclose()
        return resp

    resp = asyncio.run(main())
    assert resp.status_code == 200
    assert resp.headers["x-scheduler-decision"] == "hold"
    assert resp.headers["x-scheduler-reason"] == "capacity"
    assert resp.headers["x-scheduler-hold-time"] == "0.250000"
    # A held request reserved nothing, but it still needs blocks: the demand
    # is what separates "the cache is busy" from "you asked for too much".
    assert resp.headers["x-kv-blocks-reserved"] == "0"
    assert resp.headers["x-kv-blocks-demand"] == "11"


def test_a_rejected_request_becomes_a_429():
    sched = FakeScheduler(decision="reject", reason="queue_full")

    async def main():
        app, client = _gateway(sched)
        async with client:
            resp = await client.post(
                "/v1/completions",
                json={"prompt": "hello", "max_tokens": 8},
                headers={"x-user-id": "u1"},
            )
        await app.state.client.aclose()
        return resp

    resp = asyncio.run(main())
    assert resp.status_code == 429
    assert resp.json()["reason"] == "queue_full"
    assert resp.headers["x-scheduler-reason"] == "queue_full"
    assert resp.headers["x-scheduler-decision"] == "reject"
    # Refused, but not because it needed nothing.
    assert resp.headers["x-kv-blocks-reserved"] == "0"
    assert resp.headers["x-kv-blocks-demand"] == "11"


def test_an_engine_that_is_not_ready_is_a_503():
    sched = FakeScheduler(decision="reject", reason="engine_not_ready")

    async def main():
        app, client = _gateway(sched)
        async with client:
            resp = await client.post(
                "/v1/completions",
                json={"prompt": "hello", "max_tokens": 8},
                headers={"x-user-id": "u1"},
            )
        await app.state.client.aclose()
        return resp

    resp = asyncio.run(main())
    assert resp.status_code == 503
    assert resp.json()["reason"] == "engine_not_ready"


# --- fairness and idempotency ------------------------------------------------


def test_a_duplicate_submit_is_refused_and_starts_no_second_engine_request():
    sched = FakeScheduler()
    sched.block()

    async def main():
        app, client = _gateway(sched, cache_enabled=False)
        async with client:
            first = asyncio.create_task(
                client.post(
                    "/v1/completions",
                    json={"prompt": "hello", "max_tokens": 8},
                    headers={"x-user-id": "u1", "x-request-id": "req-1",
                             "idempotency-key": "k1"},
                )
            )
            await asyncio.wait_for(sched.stream_started.wait(), timeout=5)
            second = await client.post(
                "/v1/completions",
                json={"prompt": "hello", "max_tokens": 8},
                headers={"x-user-id": "u1", "x-request-id": "req-2",
                         "idempotency-key": "k1"},
            )
            sched.release()
            await first
        await app.state.client.aclose()
        return second

    second = asyncio.run(main())
    assert second.status_code == 409
    assert second.headers["x-idempotent-replay"] == "true"
    assert second.headers["x-request-id"] == "req-1"
    assert second.json()["reason"] == "duplicate"
    # One engine request, not two.
    assert len(sched.submits) == 1


def test_two_users_may_use_the_same_idempotency_key():
    """The key is scoped to the user, so this is not a collision."""
    sched = FakeScheduler()

    async def main():
        app, client = _gateway(sched, cache_enabled=False)
        async with client:
            a = await client.post(
                "/v1/completions",
                json={"prompt": "a", "max_tokens": 8},
                headers={"x-user-id": "u1", "idempotency-key": "k1"},
            )
            b = await client.post(
                "/v1/completions",
                json={"prompt": "b", "max_tokens": 8},
                headers={"x-user-id": "u2", "idempotency-key": "k1"},
            )
        await app.state.client.aclose()
        return a, b

    a, b = asyncio.run(main())
    assert a.status_code == 200
    assert b.status_code == 200
    assert len(sched.submits) == 2


def test_the_gateway_refuses_a_user_that_is_over_its_own_limit():
    sched = FakeScheduler()
    sched.block()

    async def main():
        app, client = _gateway(sched, max_in_flight_per_user=1)
        async with client:
            first = asyncio.create_task(
                client.post(
                    "/v1/completions",
                    json={"prompt": "a", "max_tokens": 8},
                    headers={"x-user-id": "u1"},
                )
            )
            await asyncio.wait_for(sched.stream_started.wait(), timeout=5)
            second = await client.post(
                "/v1/completions",
                json={"prompt": "b", "max_tokens": 8},
                headers={"x-user-id": "u1"},
            )
            sched.release()
            await first
        await app.state.client.aclose()
        return second

    second = asyncio.run(main())
    assert second.status_code == 429
    assert second.json()["reason"] == "user_concurrency"


def test_a_request_without_a_user_id_is_refused():
    sched = FakeScheduler()

    async def main():
        app, client = _gateway(sched)
        async with client:
            resp = await client.post("/v1/completions", json={"prompt": "hello"})
        await app.state.client.aclose()
        return resp

    resp = asyncio.run(main())
    assert resp.status_code == 400
    assert resp.json()["reason"] == "missing_user_id"
    assert sched.submits == []


# --- client-visible latency ----------------------------------------------------


def test_the_gateway_measures_ttft_and_tpot_as_the_client_sees_them():
    """The latency the user feels, measured where the user feels it.

    The benchmark client streams to the gateway, so a gateway-side TTFT is
    the same measurement the benchmark-comparison panel shows — one clock,
    one definition, and the queueing and the scheduler's hold inside it.
    """
    sched = FakeScheduler()
    sched.frames = [
        {"delta": "a"},
        {"delta": "b"},
        {"delta": "c"},
        {"done": True, "finish_reason": "stop", "prompt_tokens": 7, "completion_tokens": 3},
    ]

    async def main():
        app, client = _gateway(sched)
        async with client:
            resp = await client.post(
                "/v1/completions",
                json={"prompt": "hello", "max_tokens": 8, "stream": True},
                headers={"x-user-id": "u1"},
            )
        await app.state.client.aclose()
        return resp

    ttft_before = _counter("gate_ttft_seconds_count")
    tpot_before = _counter("gate_tpot_seconds_count")
    resp = asyncio.run(main())
    assert resp.status_code == 200
    assert _counter("gate_ttft_seconds_count") - ttft_before == 1
    # Three output tokens means two inter-token gaps, so TPOT is defined.
    assert _counter("gate_tpot_seconds_count") - tpot_before == 1


def test_a_single_token_request_has_no_tpot():
    """One token has no gap after the first, so there is no per-token cost."""
    sched = FakeScheduler()

    async def main():
        app, client = _gateway(sched)
        async with client:
            resp = await client.post(
                "/v1/completions",
                json={"prompt": "hello", "max_tokens": 8, "stream": True},
                headers={"x-user-id": "u1"},
            )
        await app.state.client.aclose()
        return resp

    tpot_before = _counter("gate_tpot_seconds_count")
    ttft_before = _counter("gate_ttft_seconds_count")
    resp = asyncio.run(main())
    assert resp.status_code == 200
    assert _counter("gate_tpot_seconds_count") == tpot_before
    assert _counter("gate_ttft_seconds_count") - ttft_before == 1


def test_a_shed_request_is_not_counted_as_a_latency_sample():
    """A request the scheduler dropped never finished; its timing is a lie."""
    sched = FakeScheduler()
    sched.frames = [{"done": True, "finish_reason": "capacity_breach"}]

    async def main():
        app, client = _gateway(sched)
        async with client:
            resp = await client.post(
                "/v1/completions",
                json={"prompt": "hello", "max_tokens": 8, "stream": True},
                headers={"x-user-id": "u1"},
            )
        await app.state.client.aclose()
        return resp

    ttft_before = _counter("gate_ttft_seconds_count")
    resp = asyncio.run(main())
    assert resp.status_code == 200
    assert _counter("gate_ttft_seconds_count") == ttft_before
    # The status line was already sent, so the error arrives as a frame —
    # but it is a retryable one, not an empty completion.
    assert "capacity_breach" in resp.text


def test_a_shed_request_is_a_retryable_503():
    """The buffered path can still send a status line, so it sends the right one."""
    sched = FakeScheduler()
    sched.frames = [{"done": True, "finish_reason": "capacity_breach"}]

    async def main():
        app, client = _gateway(sched)
        async with client:
            resp = await client.post(
                "/v1/completions",
                json={"prompt": "hello", "max_tokens": 8},
                headers={"x-user-id": "u1"},
            )
        await app.state.client.aclose()
        return resp

    resp = asyncio.run(main())
    assert resp.status_code == 503
    assert resp.headers["retry-after"] == "1"
    assert resp.json()["reason"] == "capacity_breach"


# --- accounting --------------------------------------------------------------


def test_a_cache_miss_is_counted():
    sched = FakeScheduler()

    async def main():
        app, client = _gateway(sched)
        before = _counter("gate_cache_misses_total")
        async with client:
            await client.post(
                "/v1/completions",
                json={"prompt": "hello", "max_tokens": 8},
                headers={"x-user-id": "u1"},
            )
        await app.state.client.aclose()
        return _counter("gate_cache_misses_total") - before

    assert asyncio.run(main()) == 1


def test_a_broken_stream_aborts_the_request_so_the_reservation_does_not_leak():
    sched = FakeScheduler(stream_status=500)

    async def main():
        app, client = _gateway(sched)
        async with client:
            resp = await client.post(
                "/v1/completions",
                json={"prompt": "hello", "max_tokens": 8},
                headers={"x-user-id": "u1"},
            )
        await app.state.client.aclose()
        return resp

    resp = asyncio.run(main())
    assert resp.status_code == 502
    assert len(sched.aborts) == 1
    assert sched.aborts[0]["request_id"] == sched.submits[0]["request_id"]


def test_health_reports_the_scheduler_state():
    sched = FakeScheduler()

    async def main():
        app, client = _gateway(sched)
        async with client:
            resp = await client.get("/health")
        await app.state.client.aclose()
        return resp

    resp = asyncio.run(main())
    payload = resp.json()
    assert payload["scheduler"]["ready"] is True
    assert payload["scheduler"]["held"] == 2
    assert payload["scheduler"]["total_blocks"] == 3662


def test_health_is_degraded_when_the_scheduler_cannot_serve():
    sched = FakeScheduler()
    sched.state = {"ready": False, "inflight": 0, "held": 0}

    async def main():
        app, client = _gateway(sched)
        async with client:
            resp = await client.get("/health")
        await app.state.client.aclose()
        return resp

    resp = asyncio.run(main())
    assert resp.json()["status"] == "degraded"
