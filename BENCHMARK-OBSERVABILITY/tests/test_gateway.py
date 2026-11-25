"""Gateway routing, admission, caching, and both proxy paths.

The app tests drive the real ASGI app through ``httpx.ASGITransport`` with a
``MockTransport`` upstream, so no engine and no socket are involved.
``app.state.client`` is normally built in the lifespan and ASGITransport does
not run lifespans, so the tests set it directly.
"""

from __future__ import annotations

import asyncio
import time

import httpx
import pytest
from infergw.admission import Admission
from infergw.app import build_app
from infergw.cache import CachedCompletion, ExactCache, cache_key, text_from_sse
from infergw.config import Backend, GatewayConfig
from infergw.router import NoHealthyBackend, Router

SSE = (
    b'data: {"choices":[{"text":"hel"}]}\n\n'
    b'data: {"choices":[{"text":"lo"}]}\n\n'
    b'data: [DONE]\n\n'
)


def _cfg(**kw) -> GatewayConfig:
    backends = kw.pop("backends", [Backend(name="b1", base_url="http://engine.test")])
    return GatewayConfig(backends=backends, **kw)


def _two(**kw) -> Router:
    return Router(
        _cfg(
            backends=[
                Backend(name="b1", base_url="http://a.test"),
                Backend(name="b2", base_url="http://b.test"),
            ],
            **kw,
        )
    )


# --- routing ----------------------------------------------------------------


def test_ties_go_to_declaration_order_so_a_run_is_reproducible():
    assert _two().pick().name == "b1"


def test_least_outstanding_skips_the_busy_backend():
    r = _two()
    r.acquire("b1")
    assert r.pick().name == "b2"
    r.acquire("b2")
    r.release("b1")
    assert r.pick().name == "b1"


def test_weighted_split_is_deterministic_and_proportional():
    r = _two(strategy="weighted")
    r.config.backends[0].weight = 0.7
    r.config.backends[1].weight = 0.3
    picked = [r.pick().name for _ in range(10)]
    assert picked.count("b1") == 7
    assert picked.count("b2") == 3
    # Same sequence on a second pass: a wobbling split is not comparable run to run.
    assert [r.pick().name for _ in range(10)] == picked


def test_unhealthy_backends_are_skipped():
    r = _two()
    r.mark_unhealthy("b1")
    assert [r.pick().name for _ in range(3)] == ["b2", "b2", "b2"]


def test_no_healthy_backend_raises():
    r = _two()
    r.mark_unhealthy("b1")
    r.mark_unhealthy("b2")
    with pytest.raises(NoHealthyBackend):
        r.pick()


def test_outstanding_never_goes_negative():
    r = _two()
    r.release("b1")
    assert r.outstanding("b1") == 0


# --- admission --------------------------------------------------------------


def test_admission_refuses_rather_than_waits():
    a = Admission(limit=1)
    assert a.try_acquire("r1")
    assert a.saturated
    assert not a.try_acquire("r2")  # refuse, do not block
    a.release("r1")
    assert not a.saturated
    assert a.try_acquire("r3")


def test_a_waiting_request_moves_to_running_when_the_scheduler_starts_it():
    """Two counters, because the gateway must not count a held request
    against the running ceiling the scheduler reports."""
    a = Admission(limit=2)
    assert a.try_acquire("r1")
    assert a.try_acquire("r2")
    assert (a.running, a.waiting) == (0, 2)

    a.mark_running("r1")

    assert (a.running, a.waiting) == (1, 1)
    assert a.in_flight == 2
    a.release("r1")
    assert (a.running, a.waiting) == (0, 1)


def test_admission_rejects_a_useless_limit():
    with pytest.raises(ValueError):
        Admission(0)


# --- cache ------------------------------------------------------------------


def test_key_ignores_cosmetic_fields():
    assert cache_key({"prompt": "x", "max_tokens": 5, "user": "abc"}) == cache_key(
        {"prompt": "x", "max_tokens": 5}
    )


def test_key_separates_different_sampling():
    base = {"prompt": "x", "max_tokens": 5}
    assert cache_key(base) != cache_key({**base, "max_tokens": 6})
    assert cache_key(base) != cache_key({**base, "temperature": 0.9})
    assert cache_key(base) != cache_key({**base, "seed": 1})


def test_key_ignores_stream_so_both_shapes_share_an_entry():
    base = {"prompt": "x", "max_tokens": 5}
    assert cache_key(base) == cache_key({**base, "stream": True})


def test_cache_hit_and_miss_counters():
    c = ExactCache()
    assert c.get("absent") is None
    c.put("k", CachedCompletion("hi", 1, 1, "stop", 0.5, time.time() + 60))
    assert c.get("k").text == "hi"
    stats = c.stats()
    assert stats["hits"] == 1 and stats["misses"] == 1
    assert stats["hit_rate"] == pytest.approx(0.5)


def test_expired_entry_is_a_miss_not_a_stale_answer():
    c = ExactCache()
    c.put("k", CachedCompletion("old", 1, 1, "stop", 0.5, time.time() - 1))
    assert c.get("k") is None
    assert c.stats()["expirations"] == 1


def test_lru_eviction_drops_the_least_recently_used():
    c = ExactCache(max_entries=2)
    for k in ("a", "b"):
        c.put(k, CachedCompletion(k, 1, 1, "stop", 0.1, time.time() + 60))
    c.get("a")  # touch a so b becomes the coldest
    c.put("c", CachedCompletion("c", 1, 1, "stop", 0.1, time.time() + 60))
    assert c.get("b") is None
    assert c.get("a") is not None
    assert c.stats()["evictions"] == 1


def test_text_from_sse_concatenates_and_counts_chunks():
    text, tokens, finish = text_from_sse([SSE])
    assert text == "hello"
    assert tokens == 2
    assert finish == "stop"


def test_text_from_sse_reads_finish_reason_and_survives_junk():
    body = (
        b": keep-alive comment\n\n"
        b"data: {not json}\n\n"
        b"data: [DONE]\n\n"
        b'data: {"choices":[{"text":"z","finish_reason":"length"}]}\n\n'
    )
    text, tokens, finish = text_from_sse([body])
    assert (text, tokens, finish) == ("z", 1, "length")


# --- app --------------------------------------------------------------------


def _gateway(handler, **cfg_kw):
    """Build the app plus a client pointed at it. Caller closes both."""
    app = build_app(_cfg(**cfg_kw))
    app.state.client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), timeout=5.0
    )
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gw.test"
    )
    return app, client


def _json_handler(calls: list, text: str = "hi"):
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(
            200,
            json={
                "id": "cmpl-1",
                "choices": [{"text": text, "index": 0, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            },
        )

    return handler


def _sse_handler(calls: list):
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        # `content=<bytes>` is pre-read by httpx, so `aiter_raw()` would raise
        # StreamConsumed. An async iterator is what a real engine hands back.
        async def body():
            yield SSE

        return httpx.Response(
            200, content=body(), headers={"content-type": "text/event-stream"}
        )

    return handler


def test_non_streaming_request_is_proxied_and_timestamped():
    calls: list = []

    async def main():
        app, client = _gateway(_json_handler(calls))
        async with client:
            resp = await client.post(
                "/v1/completions", json={"prompt": "hello", "max_tokens": 4}
            )
        await app.state.client.aclose()
        return resp

    resp = asyncio.run(main())
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["text"] == "hi"
    # The queue time is the one latency only the gateway can see.
    assert float(resp.headers["x-gateway-queue-time"]) >= 0.0
    assert calls == ["http://engine.test/v1/completions"]


def test_streaming_bytes_pass_through_unmodified():
    calls: list = []

    async def main():
        app, client = _gateway(_sse_handler(calls))
        async with client:
            resp = await client.post(
                "/v1/completions", json={"prompt": "hello", "stream": True}
            )
        await app.state.client.aclose()
        return resp

    resp = asyncio.run(main())
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert resp.content == SSE


def test_second_identical_request_is_served_from_cache():
    calls: list = []

    async def main():
        app, client = _gateway(_json_handler(calls))
        body = {"prompt": "hello", "max_tokens": 4, "temperature": 0.0}
        async with client:
            first = await client.post("/v1/completions", json=body)
            second = await client.post("/v1/completions", json=body)
        await app.state.client.aclose()
        return first, second

    first, second = asyncio.run(main())
    assert calls == ["http://engine.test/v1/completions"]  # upstream hit once
    assert "x-gateway-cache" not in first.headers
    assert second.headers["x-gateway-cache"] == "hit"
    assert second.json()["choices"][0]["text"] == first.json()["choices"][0]["text"]


def test_cached_entry_is_not_reused_for_a_different_request():
    calls: list = []

    async def main():
        app, client = _gateway(_json_handler(calls))
        async with client:
            await client.post("/v1/completions", json={"prompt": "a", "max_tokens": 4})
            await client.post("/v1/completions", json={"prompt": "a", "max_tokens": 5})
        await app.state.client.aclose()

    asyncio.run(main())
    assert len(calls) == 2


def test_streaming_request_releases_its_admission_slot():
    """Starlette drains the stream generator after the handler returns, so the
    release has to happen inside the generator. Missing it 429s the gateway
    forever after `max_in_flight` streams."""
    calls: list = []

    async def main():
        app, client = _gateway(_sse_handler(calls), max_in_flight=1)
        async with client:
            for i in range(3):
                # Distinct prompts so each is a cache miss and has to pass
                # through admission. Three identical ones would all hit the
                # cache and prove nothing.
                resp = await client.post(
                    "/v1/completions", json={"prompt": f"x{i}", "stream": True}
                )
                assert resp.status_code == 200, resp.text
            health = await client.get("/health")
        await app.state.client.aclose()
        return health

    health = asyncio.run(main())
    assert health.json()["in_flight"] == 0
    assert len(calls) == 3


def test_streaming_request_can_be_replayed_from_cache_as_sse():
    calls: list = []

    async def main():
        app, client = _gateway(_sse_handler(calls))
        body = {"prompt": "same", "stream": True, "temperature": 0.0}
        async with client:
            first = await client.post("/v1/completions", json=body)
            second = await client.post("/v1/completions", json=body)
        await app.state.client.aclose()
        return first, second

    first, second = asyncio.run(main())
    assert len(calls) == 1  # upstream hit once
    assert second.headers["x-gateway-cache"] == "hit"
    # The replay is re-synthesised, not byte-identical (the original chunking is
    # gone), so assert on what a client would actually read — parsed with the
    # same SSE parser the cache wrote through.
    assert second.text.endswith("data: [DONE]\n\n")
    assert text_from_sse([first.content])[0] == "hello"
    replay_text, replay_tokens, replay_finish = text_from_sse([second.content])
    assert replay_text == "hello"
    assert replay_tokens == 1
    assert replay_finish == "stop"


def test_saturation_sheds_exactly_one_request():
    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.2)  # hold the single slot open
        return httpx.Response(200, json={"choices": [{"text": "x", "finish_reason": "stop"}]})

    async def main():
        app, client = _gateway(handler, max_in_flight=1)
        async with client:
            responses = await asyncio.gather(
                client.post("/v1/completions", json={"prompt": "a", "max_tokens": 1}),
                client.post("/v1/completions", json={"prompt": "b", "max_tokens": 1}),
            )
        await app.state.client.aclose()
        return responses

    # Which one wins is scheduling, not contract; that exactly one is refused is.
    responses = asyncio.run(main())
    assert sorted(r.status_code for r in responses) == [200, 429]
    shed = next(r for r in responses if r.status_code == 429)
    assert shed.json()["error"] == "gateway saturated"
    assert shed.headers["retry-after"] == "1"


def test_health_reports_backends_and_no_healthy_backend_is_503():
    async def main():
        cfg = _cfg(backends=[Backend(name="dead", base_url="http://x.test", healthy=False)])
        app = build_app(cfg)
        app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(_json_handler([])))
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gw.test"
        )
        async with client:
            health = await client.get("/health")
            post = await client.post("/v1/completions", json={"prompt": "x"})
        await app.state.client.aclose()
        return health, post

    health, post = asyncio.run(main())
    assert health.json()["status"] == "degraded"
    assert health.json()["backends"][0]["name"] == "dead"
    assert post.status_code == 503


def test_cache_can_be_disabled_entirely():
    calls: list = []

    async def main():
        app, client = _gateway(_json_handler(calls), cache_enabled=False)
        async with client:
            body = {"prompt": "same", "max_tokens": 4}
            await client.post("/v1/completions", json=body)
            await client.post("/v1/completions", json=body)
            stats = await client.get("/gateway/cache")
        await app.state.client.aclose()
        return stats

    stats = asyncio.run(main())
    assert stats.json() == {"enabled": False}
    assert len(calls) == 2


def test_invalid_json_body_is_a_400_not_a_crash():
    async def main():
        app, client = _gateway(_json_handler([]))
        async with client:
            resp = await client.post(
                "/v1/completions",
                content=b"{not json",
                headers={"content-type": "application/json"},
            )
        await app.state.client.aclose()
        return resp

    assert asyncio.run(main()).status_code == 400
