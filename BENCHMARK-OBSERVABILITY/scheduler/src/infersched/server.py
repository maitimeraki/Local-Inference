"""The scheduler tier's HTTP surface.

One process holds the GPU and the engine; the gateway talks to this over
loopback. The contract is §12.2 of the design doc: submit decides fast, the
stream carries the output.

Nothing here reads the engine's mutable state. The engine thread owns the
block pool, and `req_to_blocks` is a dict that the engine mutates while it
runs — reading it from the event loop would race. Everything the HTTP layer
reports comes from the ledger, which the engine thread refreshes on every
step.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from inferbench.config import load_engines
from inferbench.step_engine import StepEngine
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from infersched.config import SchedulerConfig, load_scheduler
from infersched.dispatcher import RUNNING, Dispatcher, ScheduledRequest

log = logging.getLogger(__name__)


class SubmitBody(BaseModel):
    """A request, flattened. The gateway fills this in from the OpenAI body."""

    request_id: str
    user_id: str
    prompt: str
    max_tokens: int = 64
    temperature: float = 0.7
    top_p: float = 1.0
    seed: int | None = None
    # Lower number, higher importance. The engine serves the smallest first
    # and preempts the largest last, so this is the lever the scheduler uses
    # to decide who survives a capacity breach.
    priority: int = 10
    min_tokens: int = 0
    ignore_eos: bool = False
    idempotency_key: str | None = None
    deadline_ts: float | None = None
    trace_id: str | None = None


class AbortBody(BaseModel):
    request_id: str
    reason: str = "client_abort"


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def build_app(cfg: SchedulerConfig) -> FastAPI:
    """The scheduler app. The engine is loaded in the lifespan."""
    dispatcher = Dispatcher(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        dispatcher.loop = asyncio.get_running_loop()
        engines = load_engines()
        if cfg.engine not in engines:
            raise RuntimeError(f"unknown engine config: {cfg.engine!r}")

        engine = StepEngine(engines[cfg.engine], on_step=lambda _e: dispatcher.reconcile())
        engine.start()
        dispatcher.attach_engine(engine, engine.block_size(), engine.max_model_len())
        app.state.engine = engine
        try:
            yield
        finally:
            # The engine core holds the GPU. Stopping it here is what keeps a
            # restart from inheriting a dead child's VRAM.
            engine.stop()

    app = FastAPI(title="infersched", version="0.1.0", lifespan=lifespan)
    app.state.dispatcher = dispatcher

    @app.get("/health")
    async def health() -> dict:
        return {
            "status": "ok" if dispatcher.ready else "loading",
            "engine": cfg.engine,
            "ready": dispatcher.ready,
        }

    @app.get("/ready")
    async def ready() -> Response:
        """Readiness, as an HTTP status.

        `/health` answers 200 while the weights are still loading, because the
        process is alive and that is what a liveness probe asks. A readiness
        probe needs the other half: it must fail until there is an engine for
        the ledger to account for, or a load balancer will send traffic to a
        scheduler that can only reject it.
        """
        if not dispatcher.ready:
            return Response(
                json.dumps({"ready": False, "engine": cfg.engine}),
                status_code=503,
                media_type="application/json",
            )
        return Response(
            json.dumps({"ready": True, "engine": cfg.engine}),
            media_type="application/json",
        )

    @app.get("/metrics")
    async def prometheus() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/internal/v1/submit")
    async def submit(body: SubmitBody):
        if not dispatcher.ready:
            return Response(
                json.dumps({"decision": "reject", "reason": "engine_not_ready"}),
                status_code=503,
                media_type="application/json",
            )

        req = ScheduledRequest(
            request_id=body.request_id,
            user_id=body.user_id,
            prompt=body.prompt,
            prompt_tokens=dispatcher.engine.count_tokens(body.prompt),
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            top_p=body.top_p,
            seed=body.seed,
            priority=body.priority,
            min_tokens=body.min_tokens,
            ignore_eos=body.ignore_eos,
            events=asyncio.Queue(),
            done=asyncio.Event(),
        )
        decision = dispatcher.submit(req)
        return {
            "decision": decision.decision,
            "reason": decision.reason,
            "reserved_blocks": decision.reserved_blocks,
            "demand": decision.demand,
            "hold_seq": req.hold_seq,
        }

    @app.post("/internal/v1/abort")
    async def abort(body: AbortBody):
        released = dispatcher.abort(body.request_id)
        return {"released_blocks": released}

    @app.get("/internal/v1/stream/{request_id}")
    async def stream(request_id: str):
        req = dispatcher.get(request_id)
        if req is None:
            raise HTTPException(status_code=404, detail="unknown request")
        return StreamingResponse(_events(req), media_type="text/event-stream")

    @app.get("/internal/v1/state")
    async def state() -> dict:
        ledger = dispatcher.ledger
        per_user = {
            user_id: {
                "inflight": counts["inflight"],
                "queued": counts["queued"],
                "reserved_blocks": ledger.reserved_by_user(user_id),
            }
            for user_id, counts in dispatcher.registry.snapshot().items()
        }
        return {
            "ready": dispatcher.ready,
            "inflight": dispatcher.inflight_count(),
            "held": dispatcher.held_count(),
            "free_blocks": ledger.free_blocks,
            "total_blocks": ledger.total_blocks,
            "reserved_blocks": ledger.reserved_blocks,
            "allocated_blocks": ledger.allocated_blocks,
            "drift_blocks": ledger.drift,
            "per_user": per_user,
            # The ceilings the scheduler is actually enforcing. The gateway
            # reads them so its own admission limit is derived from the
            # scheduler instead of hand-copied and left to drift.
            "limits": {
                "max_in_flight": dispatcher.max_in_flight,
                "max_in_flight_per_user": cfg.max_in_flight_per_user,
                "max_queue_depth_global": cfg.max_queue_depth_global,
                "max_queue_depth_per_user": cfg.max_queue_depth_per_user,
            },
        }

    @app.get("/internal/v1/requests/{request_id}")
    async def request_state(request_id: str):
        req = dispatcher.get(request_id)
        if req is None:
            raise HTTPException(status_code=404, detail="unknown request")
        return {
            "request_id": req.request_id,
            "user_id": req.user_id,
            "state": req.state,
            "decision": req.decision,
            "reason": req.reason,
            "reserved_blocks": req.reserved_blocks,
            "admitted_ts": req.admitted_ts,
            "hold_seconds": req.held_seconds,
            "completion_tokens": req.completion_tokens,
            "error": req.error,
        }

    return app


async def _events(req: ScheduledRequest):
    """Relay one request's output as SSE.

    The text is taken from the request, not from the event queue: the queue
    is a wakeup signal, and a client that attaches late still gets
    everything.
    """
    sent = 0
    state = req.state
    yield _sse({"request_id": req.request_id, "decision": req.decision, "state": state})
    while True:
        text = req.text
        if len(text) > sent:
            yield _sse({"delta": text[sent:]})
            sent = len(text)
        if req.dispatched and state != RUNNING:
            # It reached the engine. The caller has to be told: until it is,
            # a request that was held still looks like it is queued, and the
            # client counts it against the wrong ceiling.
            state = RUNNING
            yield _sse({"request_id": req.request_id, "state": RUNNING})
        if req.done is not None and req.done.is_set():
            break
        try:
            await asyncio.wait_for(req.events.get(), timeout=0.5)  # type: ignore[union-attr]
        except asyncio.TimeoutError:
            continue

    text = req.text
    if len(text) > sent:
        yield _sse({"delta": text[sent:]})
    yield _sse(
        {
            "done": True,
            "request_id": req.request_id,
            "state": req.state,
            "finish_reason": req.finish_reason,
            "error": req.error,
            "prompt_tokens": req.actual_prompt_tokens,
            "completion_tokens": req.completion_tokens,
            "hold_seconds": req.held_seconds,
        }
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    cfg = load_scheduler()
    uvicorn.run(build_app(cfg), host=cfg.host, port=cfg.port)


if __name__ == "__main__":
    main()
