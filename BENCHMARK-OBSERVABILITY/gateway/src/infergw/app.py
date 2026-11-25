"""The gateway ASGI app.

Order of operations per request, and why:

  1. cache lookup      — a hit costs a dict lookup; do it before spending
                         admission budget on work that needs no engine.
  2. idempotency       — a duplicate submit must not start a second engine
                         request, so it is refused before anything is admitted.
  3. admission control — refuse (429) rather than queue unboundedly, so
                         overload is visible instead of absorbed.
  4. scheduler slot    — with a scheduler configured, the request goes to the
                         tier that owns the GPU, which decides admit / hold /
                         reject against the KV-block ledger.
  5. route             — without a scheduler, pick a backend; weighted mode
                         splits traffic for A/B.
  6. proxy             — stream through untouched. Buffering here would
                         silently destroy TTFT, which is the metric half this
                         platform exists to measure.

A *hold* is waited on, not refused. The scheduler holds the request until a
batch slot frees, and the gateway relays the stream when it runs: the caller
asked for a completion, and the scheduler is the thing that decides whether
it can be served. Only a *reject* becomes a 429.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from inferobs import tracing
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from infergw import cache as cache_mod
from infergw.admission import Admission
from infergw.config import GatewayConfig, load_gateway
from infergw.metrics import metrics
from infergw.registry import UserRegistry
from infergw.router import NoHealthyBackend, Router
from infergw.scheduler_client import SchedulerClient, SchedulerUnavailable


def build_app(
    cfg: GatewayConfig | None = None, transport: httpx.AsyncBaseTransport | None = None
) -> FastAPI:
    """Build the app. `transport` is injectable so tests can run without a server."""
    cfg = cfg or load_gateway()
    m = metrics()
    router = Router(cfg)
    admission = Admission(cfg.max_in_flight)
    registry = UserRegistry(cfg.idempotency_ttl_s)
    scheduler = (
        SchedulerClient(cfg.scheduler_url, cfg.scheduler_timeout_s) if cfg.scheduler_url else None
    )
    cache = (
        cache_mod.ExactCache(cfg.cache_max_entries, cfg.cache_ttl_s)
        if cfg.cache_enabled
        else None
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.client = httpx.AsyncClient(
            timeout=cfg.request_timeout_s, transport=transport
        )
        tracing.setup_tracing("infergw")
        adopt = asyncio.create_task(_adopt_scheduler_limits(app.state.client))
        try:
            yield
        finally:
            adopt.cancel()
            await app.state.client.aclose()
            tracing.shutdown()

    app = FastAPI(title="infergw", lifespan=lifespan)

    async def _adopt_scheduler_limits(client: httpx.AsyncClient) -> None:
        """Take the gateway's ceilings from the scheduler.

        Hand-copied numbers drift: the gateway's limit and the scheduler's
        disagree, and the gateway then refuses work the scheduler was willing
        to hold — while the engine still has free slots. The scheduler's total
        capacity is what it will run plus what it will hold, so that is the
        gateway's limit.

        The scheduler may still be loading its weights when the gateway
        starts, so this keeps trying for a while: a gateway left on its own
        numbers is exactly the drift this removes.
        """
        if scheduler is None:
            return
        deadline = time.monotonic() + 120.0
        while True:
            try:
                state = await scheduler.state(client)
            except SchedulerUnavailable:
                if time.monotonic() > deadline:
                    return
                await asyncio.sleep(1.0)
                continue
            limits = state.get("limits") or {}
            running = int(limits.get("max_in_flight") or 0)
            queue = int(limits.get("max_queue_depth_global") or 0)
            if running > 0:
                admission.set_limit(running + queue)
            per_user = int(limits.get("max_in_flight_per_user") or 0)
            if per_user > 0:
                cfg.max_in_flight_per_user = per_user + int(
                    limits.get("max_queue_depth_per_user") or 0
                )
            return

    def _sync_gauges() -> None:
        """The gauges follow the counters instead of being incremented by hand."""
        m["in_flight"].set(admission.running)
        m["waiting"].set(admission.waiting)

    @app.get("/health")
    async def health(request: Request) -> dict:
        payload = {
            "status": "ok" if router.healthy() else "degraded",
            "backends": [
                {"name": b.name, "url": b.base_url, "healthy": b.healthy,
                 "outstanding": router.outstanding(b.name)}
                for b in cfg.backends
            ],
            "in_flight": admission.in_flight,
        }
        if scheduler is not None:
            # A gateway that says "ok" while the tier holding the GPU cannot
            # serve is worse than one that says "degraded" — the caller would
            # keep sending work that has nowhere to go.
            try:
                state = await scheduler.state(request.app.state.client)
            except SchedulerUnavailable as exc:
                payload["status"] = "degraded"
                payload["scheduler"] = {"ready": False, "error": str(exc)}
            else:
                ready = bool(state.get("ready"))
                if not ready:
                    payload["status"] = "degraded"
                payload["scheduler"] = {
                    "ready": ready,
                    "inflight": state.get("inflight"),
                    "held": state.get("held"),
                    "free_blocks": state.get("free_blocks"),
                    "total_blocks": state.get("total_blocks"),
                }
        return payload

    @app.get("/metrics")
    async def prometheus() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/gateway/cache")
    async def cache_stats() -> dict:
        return cache.stats() if cache is not None else {"enabled": False}

    @app.post("/v1/completions")
    async def completions(request: Request):  # noqa: C901 - the flow is the doc
        accepted = time.perf_counter()
        try:
            body = await request.json()
        except (json.JSONDecodeError, ValueError):
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)

        user_id = request.headers.get("x-user-id") or ""
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        idem_key = request.headers.get("idempotency-key") or None

        with tracing.span("gateway.request", prompt_chars=len(body.get("prompt") or "")):
            trace_id = tracing.current_trace_id() or uuid.uuid4().hex
            if cfg.require_user_id and not user_id:
                return JSONResponse(
                    {"error": "X-User-Id header is required", "reason": "missing_user_id"},
                    status_code=400,
                    headers={"x-request-id": request_id},
                )

            key = cache_mod.cache_key(body) if cache is not None else None

            if cache is not None and key is not None:
                hit = cache.get(key)
                if hit is not None:
                    m["cache_hits"].inc()
                    m["cache_saved"].observe(hit.observed_duration_s)
                    m["requests"].labels(backend="cache", status="ok").inc()
                    return _replay(hit, body, m)
                m["cache_misses"].inc()

            # A duplicate submit must not start a second engine request, so
            # the check comes before anything is admitted or forwarded.
            if idem_key:
                original = registry.lookup(user_id, idem_key)
                if original is not None:
                    m["idempotent_replays"].inc()
                    m["requests"].labels(backend="none", status="conflict").inc()
                    return JSONResponse(
                        {"error": "duplicate request", "reason": "duplicate",
                         "request_id": original},
                        status_code=409,
                        headers={
                            "x-request-id": original,
                            "x-idempotent-replay": "true",
                            "x-user-id": user_id,
                            "x-trace-id": trace_id,
                        },
                    )

            if not admission.try_acquire(request_id):
                m["rejected"].labels(reason="global_limit").inc()
                m["requests"].labels(backend="none", status="rejected").inc()
                return JSONResponse(
                    {"error": "gateway saturated", "reason": "global_limit",
                     "in_flight": admission.in_flight},
                    status_code=429,
                    headers={"retry-after": "1"},
                )

            _sync_gauges()
            held: dict = {"backend": None}
            released = False

            def release() -> None:
                nonlocal released
                if released:
                    return
                released = True
                if held["backend"] is not None:
                    router.release(held["backend"].name)
                    held["backend"] = None
                admission.release(request_id)
                registry.release(user_id, request_id)
                _sync_gauges()

            limit = _user_limit(cfg, body)
            if user_id and registry.in_flight(user_id) >= limit:
                release()
                m["rejected"].labels(reason="user_concurrency").inc()
                m["requests"].labels(backend="none", status="rejected").inc()
                return JSONResponse(
                    {"error": "per-user concurrency limit reached", "reason": "user_concurrency",
                     "in_flight": registry.in_flight(user_id)},
                    status_code=429,
                    headers={"retry-after": "1", "x-user-id": user_id, "x-request-id": request_id},
                )
            registry.acquire(user_id, request_id, idem_key)

            queue_s = time.perf_counter() - accepted
            m["queue"].observe(queue_s)
            headers = {
                "x-gateway-queue-time": f"{queue_s:.6f}",
                "x-trace-id": trace_id,
                "x-request-id": request_id,
                "x-user-id": user_id,
            }
            client: httpx.AsyncClient = request.app.state.client

            if scheduler is not None:
                return await _to_scheduler(
                    client, body, headers, key, request_id, user_id, idem_key,
                    trace_id, accepted, release,
                )

            try:
                backend = router.pick()
            except NoHealthyBackend as exc:
                release()
                m["requests"].labels(backend="none", status="unavailable").inc()
                return JSONResponse({"error": str(exc)}, status_code=503, headers=headers)
            held["backend"] = backend
            router.acquire(backend.name)
            # No scheduler in this path: the backend slot is the engine slot,
            # so the request is running from here on.
            admission.mark_running(request_id)
            _sync_gauges()

            if body.get("stream"):
                return StreamingResponse(
                    _proxy_stream(
                        client, cfg, router, held, body, headers, key, cache, m,
                        accepted, release,
                    ),
                    media_type="text/event-stream",
                    headers=headers,
                )
            return await _proxy_json(
                client, cfg, router, held, body, headers, key, cache, m, accepted, release
            )

    # --- the scheduler tier -----------------------------------------------------

    async def _to_scheduler(
        client, body, headers, key, request_id, user_id, idem_key, trace_id, accepted, release
    ) -> Response:
        """Hand a request to the scheduler and relay what it decides.

        The scheduler owns the KV-block ledger and the hold queue, so it is
        the one that says admit, hold or reject. The gateway's job here is to
        translate: OpenAI's body and SSE in, the decision headers out.
        """
        prompt = body.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            release()
            m["requests"].labels(backend="scheduler", status="rejected").inc()
            return JSONResponse(
                {"error": "prompt is required"}, status_code=400, headers=headers
            )

        try:
            result = await scheduler.submit(
                client,
                request_id=request_id,
                user_id=user_id,
                prompt=prompt,
                max_tokens=int(body.get("max_tokens") or 64),
                temperature=float(body.get("temperature") or 0.0),
                top_p=float(body["top_p"]) if body.get("top_p") is not None else 1.0,
                seed=body.get("seed"),
                priority=_int_field(body, "priority", 10),
                min_tokens=_int_field(body, "min_tokens", 0),
                ignore_eos=bool(body.get("ignore_eos")),
                idempotency_key=idem_key,
                deadline_ts=time.time() + cfg.request_timeout_s,
                trace_id=trace_id,
            )
        except (SchedulerUnavailable, ValueError, TypeError) as exc:
            release()
            m["requests"].labels(backend="scheduler", status="unavailable").inc()
            return JSONResponse(
                {"error": str(exc), "reason": "engine_error"}, status_code=503, headers=headers
            )

        if result.decision == "reject":
            release()
            m["rejected"].labels(reason=result.reason or "rejected").inc()
            m["requests"].labels(backend="scheduler", status="rejected").inc()
            ready = result.reason == "engine_not_ready"
            payload = {
                "error": result.reason or "rejected",
                "reason": result.reason,
                "in_flight": admission.in_flight,
            }
            out = {
                **headers,
                "x-scheduler-decision": "reject",
                "x-scheduler-reason": result.reason or "",
                # A rejected request reserved nothing, but it still asked for
                # blocks: that number is why it was refused, and a client
                # that cannot see it cannot tell "too big" from "too busy".
                "x-kv-blocks-reserved": "0",
                "x-kv-blocks-demand": str(result.demand),
            }
            if not ready:
                out["retry-after"] = "1"
            return JSONResponse(payload, status_code=503 if ready else 429, headers=out)

        if result.decision == "admit":
            m["admitted"].inc()
            # The engine has it: it counts against the running ceiling, not
            # the waiting one.
            admission.mark_running(request_id)
            _sync_gauges()
        else:
            m["held"].labels(reason=result.reason or "capacity").inc()
            m["user_queue_depth"].observe(result.hold_seq)

        headers = {
            **headers,
            "x-scheduler-decision": result.decision,
            # Which ceiling bound decided this: fit, capacity, kv_capacity,
            # user_reserved, user_concurrency. Without it a held request is
            # just "held" and the client cannot tell a busy cache from a
            # busy user.
            "x-scheduler-reason": result.reason or "",
            "x-kv-blocks-reserved": str(result.reserved_blocks),
            # What the request asked for, before any ceiling was applied. The
            # gap between this and the reservation is the scheduler's
            # over-promise: it is why a held request can still be too big.
            "x-kv-blocks-demand": str(result.demand),
        }

        if body.get("stream"):
            return StreamingResponse(
                _relay_stream(client, request_id, body.get("model"), headers, m, accepted, release),
                media_type="text/event-stream",
                headers=headers,
            )
        return await _relay_json(
            client, request_id, body.get("model"), headers, key, cache, m, accepted, release
        )

    async def _relay_stream(client, request_id, model, headers, m, accepted, release):
        """The scheduler's frames, in OpenAI's shape.

        A held request waits here: the scheduler decides when it runs, and the
        caller sees one continuous stream either way.
        """
        started = time.perf_counter()
        base = {"object": "text_completion", "created": int(time.time()), "model": model}
        completed = False
        status = "ok"
        # The client-visible latency markers. `accepted` is the request's
        # send instant, so first-token minus accepted is the TTFT a user
        # actually feels: queueing, the scheduler's hold and prefill all
        # inside it.
        first_delta_at: float | None = None
        last_delta_at: float | None = None
        try:
            async for frame in scheduler.stream(client, request_id):
                if frame.get("state") == "running":
                    # The scheduler started a held request. It counts against
                    # the running ceiling now, not the waiting one.
                    admission.mark_running(request_id)
                    _sync_gauges()
                if frame.get("done"):
                    finish = frame.get("finish_reason") or "stop"
                    if finish == "capacity_breach":
                        # The scheduler shed this request to keep the pool
                        # alive. The status line is already sent, so the
                        # client gets the error as a frame — but it gets a
                        # *retryable* one, not an empty completion.
                        status = "capacity_breach"
                        m["rejected"].labels(reason="capacity_breach").inc()
                        yield _sse({
                            "error": "capacity_breach",
                            "reason": "capacity_breach",
                            "request_id": request_id,
                        })
                        yield "data: [DONE]\n\n"
                        completed = True
                        break
                    usage = _usage(frame)
                    if first_delta_at is not None and last_delta_at is not None:
                        m["ttft"].observe(first_delta_at - accepted)
                        output_tokens = usage["completion_tokens"]
                        if output_tokens > 1:
                            # vLLM's TPOT: the per-token cost after the
                            # first, which excludes prefill.
                            m["tpot"].observe(
                                (last_delta_at - first_delta_at) / (output_tokens - 1)
                            )
                    yield _sse({
                        "id": f"cmpl-{request_id}", **base,
                        "choices": [{"text": "", "index": 0,
                                     "finish_reason": finish}],
                    })
                    yield _sse({
                        "id": f"cmpl-{request_id}", **base, "choices": [],
                        "usage": usage,
                    })
                    yield "data: [DONE]\n\n"
                    completed = True
                    hold = frame.get("hold_seconds")
                    if hold is not None:
                        m["hold_seconds"].observe(float(hold))
                    break
                delta = frame.get("delta") or ""
                if delta:
                    now = time.perf_counter()
                    if first_delta_at is None:
                        first_delta_at = now
                    last_delta_at = now
                    yield _sse({
                        "id": f"cmpl-{request_id}", **base,
                        "choices": [{"text": delta, "index": 0, "finish_reason": None}],
                    })
            m["requests"].labels(backend="scheduler", status=status).inc()
        except SchedulerUnavailable as exc:
            m["requests"].labels(backend="scheduler", status="connect_error").inc()
            yield _sse({"error": str(exc)})
        finally:
            if not completed:
                # The caller went away, or the stream broke. The slot and the
                # reservation have to go back, or the next request waits for
                # capacity nobody is using.
                await _abort(client, request_id, "client_disconnect")
            m["upstream"].labels(backend="scheduler").observe(time.perf_counter() - started)
            m["duration"].labels(backend="scheduler").observe(time.perf_counter() - accepted)
            release()

    async def _relay_json(client, request_id, model, headers, key, cache, m, accepted, release):
        """The same relay, buffered into one JSON body."""
        started = time.perf_counter()
        parts: list[str] = []
        finish_reason = "stop"
        frame_done: dict = {}
        completed = False
        try:
            async for frame in scheduler.stream(client, request_id):
                if frame.get("state") == "running":
                    # The scheduler started a held request. It counts against
                    # the running ceiling now, not the waiting one.
                    admission.mark_running(request_id)
                    _sync_gauges()
                if frame.get("done"):
                    frame_done = frame
                    finish_reason = frame.get("finish_reason") or "stop"
                    completed = True
                    break
                delta = frame.get("delta") or ""
                if delta:
                    parts.append(delta)
            if finish_reason == "capacity_breach":
                # The scheduler shed this request to keep the pool alive.
                # The client can retry: this is a capacity signal, not a
                # broken request. Nothing goes in the cache — the text is
                # partial at best.
                m["requests"].labels(backend="scheduler", status="capacity_breach").inc()
                m["rejected"].labels(reason="capacity_breach").inc()
                return JSONResponse(
                    {"error": "capacity_breach", "reason": "capacity_breach",
                     "request_id": request_id},
                    status_code=503,
                    headers={**headers, "retry-after": "1"},
                )
            text = "".join(parts)
            usage = _usage(frame_done)
            hold = frame_done.get("hold_seconds")
            if hold is not None:
                m["hold_seconds"].observe(float(hold))
            if cache is not None and key is not None and text:
                cache.put(
                    key,
                    cache_mod.CachedCompletion(
                        text=text,
                        prompt_tokens=usage["prompt_tokens"],
                        completion_tokens=usage["completion_tokens"],
                        finish_reason=finish_reason,
                        observed_duration_s=time.perf_counter() - accepted,
                        expires_at=time.time() + cache.ttl_s,
                    ),
                )
            m["requests"].labels(backend="scheduler", status="ok").inc()
            out = dict(headers)
            if hold is not None:
                out["x-scheduler-hold-time"] = f"{float(hold):.6f}"
            return JSONResponse(
                {
                    "id": f"cmpl-{request_id}",
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{"text": text, "index": 0, "finish_reason": finish_reason}],
                    "usage": usage,
                },
                headers=out,
            )
        except SchedulerUnavailable as exc:
            m["requests"].labels(backend="scheduler", status="connect_error").inc()
            return JSONResponse({"error": str(exc)}, status_code=502, headers=headers)
        finally:
            if not completed:
                await _abort(client, request_id, "client_disconnect")
            m["upstream"].labels(backend="scheduler").observe(time.perf_counter() - started)
            m["duration"].labels(backend="scheduler").observe(time.perf_counter() - accepted)
            release()

    async def _abort(client, request_id, reason) -> None:
        """Give a request's slot and reservation back.

        The abort has to outlive the cancellation that is tearing the request
        down, so it runs in its own task: awaited inline it would be cancelled
        before it ever reached the scheduler, and the reservation would leak.
        """
        try:
            task = asyncio.create_task(scheduler.abort(client, request_id, reason))
        except RuntimeError:
            return
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            raise
        except Exception:
            pass

    return app


def _sse(payload: dict) -> str:
    """One OpenAI-shaped SSE frame."""
    return f"data: {json.dumps(payload)}\n\n"


def _usage(frame: dict) -> dict:
    """Token usage out of the scheduler's final frame."""
    prompt_tokens = int(frame.get("prompt_tokens") or 0)
    completion_tokens = int(frame.get("completion_tokens") or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _int_field(body: dict, name: str, default: int) -> int:
    """An integer body field, with the default only when it is absent.

    `int(body.get(name) or default)` would turn a legitimate `0` into the
    default, and `priority=0` is the *highest* importance there is.
    """
    value = body.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _user_limit(cfg: GatewayConfig, body: dict) -> int:
    """The concurrency ceiling for this user, after clamping.

    A client cannot raise its own ceiling by asking for a bigger one: the
    requested limit is clamped to the server's.
    """
    requested = body.get("concurrency_limit")
    try:
        requested = int(requested) if requested is not None else cfg.max_in_flight_per_user
    except (TypeError, ValueError):
        requested = cfg.max_in_flight_per_user
    return max(1, min(requested, cfg.max_in_flight_per_user))


def _replay(hit: cache_mod.CachedCompletion, body: dict, m: dict) -> Response:
    """Serve a cache hit in the shape the caller asked for."""
    if body.get("stream"):
        created = int(time.time())
        base = {"object": "text_completion", "created": created, "model": body.get("model")}
        frames = [
            {"id": "cmpl-cache", **base,
             "choices": [{"text": hit.text, "index": 0, "finish_reason": None}]},
            {"id": "cmpl-cache", **base,
             "choices": [{"text": "", "index": 0, "finish_reason": hit.finish_reason}]},
            {"id": "cmpl-cache", **base, "choices": [],
             "usage": {
                 "prompt_tokens": hit.prompt_tokens,
                 "completion_tokens": hit.completion_tokens,
                 "total_tokens": hit.prompt_tokens + hit.completion_tokens,
             }},
        ]
        body_text = "".join(f"data: {json.dumps(f)}\n\n" for f in frames) + "data: [DONE]\n\n"
        return Response(
            content=body_text,
            media_type="text/event-stream",
            headers={"x-gateway-cache": "hit", "x-gateway-queue-time": "0"},
        )

    return JSONResponse(
        {
            "id": "cmpl-cache",
            "object": "text_completion",
            "created": int(time.time()),
            "model": body.get("model"),
            "choices": [{"text": hit.text, "index": 0, "finish_reason": hit.finish_reason}],
            "usage": {
                "prompt_tokens": hit.prompt_tokens,
                "completion_tokens": hit.completion_tokens,
                "total_tokens": hit.prompt_tokens + hit.completion_tokens,
            },
        },
        headers={"x-gateway-cache": "hit"},
    )


async def _proxy_stream(
    client, cfg, router, held, body, headers, key, cache, m, accepted, release
):
    """Stream upstream bytes through untouched, capturing a cacheable copy.

    No buffering: bytes are yielded the moment they arrive. The cost of
    accumulating a copy for the cache is paid only when caching is on.

    `release` is mandatory here: Starlette drains this generator *after* the
    handler returns, so nothing else will hand the admission slot back.
    """
    started = time.perf_counter()
    captured: list[bytes] = []
    try:
        for attempt in range(cfg.max_retries + 1):
            backend = held["backend"]
            try:
                async with client.stream(
                    "POST", f"{backend.base_url.rstrip('/')}/v1/completions", json=body
                ) as resp:
                    if resp.status_code >= 500 and attempt < cfg.max_retries:
                        router.release(backend.name)
                        held["backend"] = router.pick()
                        router.acquire(held["backend"].name)
                        continue
                    if resp.status_code >= 400:
                        raw = await resp.aread()
                        m["requests"].labels(
                            backend=backend.name, status=f"http_{resp.status_code}"
                        ).inc()
                        yield raw
                        return
                    async for chunk in resp.aiter_raw():
                        if cache is not None:
                            captured.append(chunk)
                        yield chunk
                    m["requests"].labels(backend=backend.name, status="ok").inc()
                break
            except httpx.HTTPError as exc:
                if attempt < cfg.max_retries:
                    router.release(held["backend"].name)
                    held["backend"] = router.pick()
                    router.acquire(held["backend"].name)
                    continue
                m["requests"].labels(backend=held["backend"].name, status="connect_error").inc()
                yield f"data: {json.dumps({'error': str(exc)})}\n\n".encode()
                return

        if cache is not None and key is not None and captured:
            text, tokens, finish = cache_mod.text_from_sse(captured)
            if text:
                cache.put(
                    key,
                    cache_mod.CachedCompletion(
                        text=text,
                        prompt_tokens=0,  # not carried in the stream body
                        completion_tokens=tokens,
                        finish_reason=finish,
                        observed_duration_s=time.perf_counter() - accepted,
                        expires_at=time.time() + cache.ttl_s,
                    ),
                )
    finally:
        elapsed = time.perf_counter() - started
        backend_name = held["backend"].name if held["backend"] else "none"
        m["upstream"].labels(backend=backend_name).observe(elapsed)
        m["duration"].labels(backend=backend_name).observe(time.perf_counter() - accepted)
        release()


async def _proxy_json(
    client, cfg, router, held, body, headers, key, cache, m, accepted, release
) -> Response:
    """Non-streaming proxy, with the same retry rule as the streaming path."""
    started = time.perf_counter()
    try:
        for attempt in range(cfg.max_retries + 1):
            backend = held["backend"]
            try:
                resp = await client.post(
                    f"{backend.base_url.rstrip('/')}/v1/completions", json=body
                )
                if resp.status_code >= 500 and attempt < cfg.max_retries:
                    router.release(backend.name)
                    held["backend"] = router.pick()
                    router.acquire(held["backend"].name)
                    continue
                status = "ok" if resp.status_code < 400 else f"http_{resp.status_code}"
                m["requests"].labels(backend=backend.name, status=status).inc()
                if cache is not None and key is not None and resp.status_code == 200:
                    _maybe_cache(cache, key, resp, m, accepted)
                return Response(
                    content=resp.content,
                    status_code=resp.status_code,
                    media_type=resp.headers.get("content-type", "application/json"),
                    headers=headers,
                )
            except httpx.HTTPError as exc:
                if attempt < cfg.max_retries:
                    router.release(held["backend"].name)
                    held["backend"] = router.pick()
                    router.acquire(held["backend"].name)
                    continue
                m["requests"].labels(backend=held["backend"].name, status="connect_error").inc()
                return JSONResponse({"error": str(exc)}, status_code=502, headers=headers)
        return JSONResponse({"error": "unreachable"}, status_code=500)
    finally:
        m["upstream"].labels(backend=(held["backend"].name if held["backend"] else "none")).observe(
            time.perf_counter() - started
        )
        m["duration"].labels(backend=(held["backend"].name if held["backend"] else "none")).observe(
            time.perf_counter() - accepted
        )
        release()


def _maybe_cache(cache, key: str, resp: httpx.Response, m: dict, accepted: float) -> None:
    try:
        data = resp.json()
    except ValueError:
        return
    choices = data.get("choices") or [{}]
    text = choices[0].get("text") or ""
    if not text:
        return
    usage = data.get("usage") or {}
    cache.put(
        key,
        cache_mod.CachedCompletion(
            text=text,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            finish_reason=choices[0].get("finish_reason") or "stop",
            observed_duration_s=time.perf_counter() - accepted,
            expires_at=time.time() + cache.ttl_s,
        ),
    )


def serve(host: str = "0.0.0.0", port: int = 9000, cfg: GatewayConfig | None = None) -> None:
    import uvicorn

    cfg = cfg or load_gateway()
    uvicorn.run(build_app(cfg), host=host, port=port, log_level="info")


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(prog="infergw", description="vLLM request router")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=9000)
    args = p.parse_args()
    serve(args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
