"""OpenAI-compatible server we own, wrapped around `AsyncLLMEngine`.

Not `vllm serve`. Running the endpoint ourselves means the engine object is
reachable, so the stat-logger tap can push scheduler and KV-cache state straight
into Prometheus without a scrape or a second process.

Serves the sliver of the OpenAI API a benchmark needs: `/v1/completions`
(streaming and not), `/v1/models`, `/health`, `/metrics`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from inferbench.config import EngineConfig, SamplingConfig
from inferbench.engine_host import EngineHost, metrics

# --- request/response models ------------------------------------------------


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str = ""
    max_tokens: int = 16
    min_tokens: int | None = None
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int | None = None
    ignore_eos: bool = False
    stream: bool = False
    stream_options: dict | None = None

    def sampling_params(self, vllm, defaults: SamplingConfig):
        max_tokens = self.max_tokens
        # A benchmark pins length; a human caller does not.
        min_tokens = self.min_tokens
        if min_tokens is None and self.ignore_eos:
            min_tokens = max_tokens
        return vllm.SamplingParams(
            max_tokens=max_tokens,
            min_tokens=min_tokens or 0,
            ignore_eos=self.ignore_eos or defaults.ignore_eos,
            temperature=self.temperature,
            top_p=self.top_p,
            seed=self.seed if self.seed is not None else defaults.seed,
        )


# --- app --------------------------------------------------------------------


def build_app(cfg: EngineConfig, model_name: str | None = None) -> FastAPI:
    """Build the ASGI app. The engine loads in the lifespan, not at import."""
    model_name = model_name or cfg.model
    defaults = SamplingConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        host = EngineHost.start(cfg)
        app.state.host = host
        app.state.vllm = host.vllm
        app.state.engine = host.engine
        app.state.sink = host.sink
        app.state.tap = host.tap
        app.state.tap_attached = host.tap_attached
        if not host.tap_attached:
            logging.getLogger(__name__).warning(
                "stat-logger tap did not attach: engine.logger_manager.stat_loggers "
                "was not found. The engine_* Prometheus metrics will stay at zero."
            )
        try:
            yield
        finally:
            # The engine core is a child process (AsyncLLM always forks), so
            # killing the parent orphans it and leaves ~5 GB of VRAM pinned.
            # Shutting the engine down explicitly reaps the child.
            host.engine.shutdown()

    app = FastAPI(title="inferbench vLLM server", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "model": model_name}

    @app.get("/v1/models")
    async def models() -> dict:
        return {
            "object": "list",
            "data": [{"id": model_name, "object": "model", "created": int(time.time())}],
        }

    @app.get("/metrics")
    async def prometheus() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest):
        engine = getattr(app.state, "engine", None)
        if engine is None:
            return JSONResponse({"error": "engine not ready"}, status_code=503)

        vllm = app.state.vllm
        request_id = f"cmpl-{uuid.uuid4().hex[:12]}"
        params = req.sampling_params(vllm, defaults)
        m = metrics()

        if not req.stream:
            return await _complete(engine, req, params, model_name, request_id, m)

        async def events():
            prev = ""
            in_tokens = None
            out_tokens = 0
            finish = "stop"
            created = int(time.time())
            try:
                async for out in engine.generate(req.prompt, params, request_id):
                    if (pt := getattr(out, "prompt_token_ids", None)) is not None:
                        in_tokens = len(pt)
                    if out.outputs:
                        text = out.outputs[0].text
                        delta = text[len(prev) :]
                        prev = text
                        out_tokens = len(out.outputs[0].token_ids)
                        if delta:
                            yield _sse(
                                {
                                    "id": request_id,
                                    "object": "text_completion",
                                    "created": created,
                                    "model": model_name,
                                    "choices": [
                                        {"text": delta, "index": 0, "finish_reason": None}
                                    ],
                                }
                            )
                        if out.finished:
                            finish = out.outputs[0].finish_reason or "stop"
            except asyncio.CancelledError:
                m["requests"].labels(status="cancelled").inc()
                raise
            except Exception as exc:  # surface, do not swallow
                m["requests"].labels(status="error").inc()
                yield _sse({"error": str(exc)})
                yield "data: [DONE]\n\n"
                return

            m["requests"].labels(status="ok").inc()
            yield _sse(
                {
                    "id": request_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model_name,
                    "choices": [{"text": "", "index": 0, "finish_reason": finish}],
                }
            )
            yield _sse(
                {
                    "id": request_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model_name,
                    "choices": [],
                    "usage": _usage(in_tokens, out_tokens),
                }
            )
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


async def _complete(engine, req, params, model_name, request_id, m) -> JSONResponse:
    """Non-streaming path: drain the generator and return one body."""
    created = int(time.time())
    in_tokens: int | None = None
    text, finish, out_tokens = "", "stop", 0
    try:
        async for out in engine.generate(req.prompt, params, request_id):
            if (pt := getattr(out, "prompt_token_ids", None)) is not None:
                in_tokens = len(pt)
            if out.outputs:
                text = out.outputs[0].text
                out_tokens = len(out.outputs[0].token_ids)
                if out.finished:
                    finish = out.outputs[0].finish_reason or "stop"
    except Exception as exc:
        m["requests"].labels(status="error").inc()
        return JSONResponse({"error": str(exc)}, status_code=500)

    m["requests"].labels(status="ok").inc()
    return JSONResponse(
        {
            "id": request_id,
            "object": "text_completion",
            "created": created,
            "model": model_name,
            "choices": [{"text": text, "index": 0, "finish_reason": finish}],
            "usage": _usage(in_tokens, out_tokens),
        }
    )


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _usage(input_tokens: int | None, output_tokens: int) -> dict:
    total = (input_tokens or 0) + output_tokens
    return {
        "prompt_tokens": input_tokens or 0,
        "completion_tokens": output_tokens,
        "total_tokens": total,
    }


def serve(cfg: EngineConfig, host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the app. Single worker: the engine is not shareable across processes."""
    import uvicorn

    uvicorn.run(build_app(cfg), host=host, port=port, log_level="info")
