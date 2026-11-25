"""Async streaming client against an OpenAI-compatible endpoint.

Deliberately not `vllm bench serve`: writing the SSE loop is how TTFT and ITL
become numbers you understand rather than numbers you trust. The harness
cross-checks against `vllm bench serve` separately, which is the point of the
phase-2 gate.

Timestamps come from `perf_counter_ns` — monotonic, so a clock adjustment
mid-run cannot produce a negative latency.
"""

from __future__ import annotations

import asyncio
import time

import httpx

from inferbench import metrics as M
from inferbench.config import SamplingConfig


class BenchmarkClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout_s: float = 120.0,
        sampling: SamplingConfig | None = None,
        connect_timeout_s: float = 10.0,
        user_id: str = "bench",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.sampling = sampling or SamplingConfig()
        # The gateway's tier-2 path needs to know the caller: it caps each
        # user's share of the GPU and scopes idempotency keys by user.
        self.user_id = user_id
        self._timeout = httpx.Timeout(timeout_s, connect=connect_timeout_s)
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> BenchmarkClient:
        self._client = httpx.AsyncClient(timeout=self._timeout)
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _payload(self, prompt: str, max_tokens: int) -> dict:
        """A deterministic request.

        `min_tokens == max_tokens` with `ignore_eos` pins the output length, so
        run-to-run variance is engine behaviour and not a variable-length
        answer. Without this, "identical workloads" is not true.
        """
        return {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "min_tokens": max_tokens,
            "ignore_eos": self.sampling.ignore_eos,
            "temperature": self.sampling.temperature,
            "top_p": self.sampling.top_p,
            "seed": self.sampling.seed,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

    async def complete(
        self,
        prompt: str,
        max_tokens: int,
        request_id: str,
        input_tokens: int | None = None,
        user_id: str | None = None,
    ) -> M.RequestMetrics:
        """One streaming request. Never raises for a server-side failure —
        failures come back as a classified status so a run reports honestly."""
        if self._client is None:
            raise RuntimeError("use BenchmarkClient as an async context manager")

        url = f"{self.base_url}/v1/completions"
        timed_chunks: list[tuple[int, int]] = []
        usage: dict | None = None
        queue_time_s: float | None = None
        trace_id: str | None = None
        status, error = M.STATUS_OK, None

        start_ns = time.perf_counter_ns()
        try:
            payload = self._payload(prompt, max_tokens)
            async with self._client.stream(
                "POST", url, json=payload, headers={"x-user-id": user_id or self.user_id}
            ) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode("utf-8", "replace")[:200]
                    end_ns = time.perf_counter_ns()
                    return M.build_request_metrics(
                        request_id,
                        start_ns,
                        end_ns,
                        [],
                        input_tokens=input_tokens,
                        status=M.STATUS_HTTP,
                        error=f"HTTP {resp.status_code}: {body}",
                    )

                if (raw_q := resp.headers.get("x-gateway-queue-time")) is not None:
                    queue_time_s = _safe_float(raw_q)
                trace_id = resp.headers.get("x-trace-id")

                async for line in resp.aiter_lines():
                    chunk = M.parse_sse_data(line.encode("utf-8"))
                    if chunk is None:
                        continue
                    now = time.perf_counter_ns()
                    k = M.chunk_token_count(chunk)
                    if k:
                        timed_chunks.append((now, k))
                    if (u := M.chunk_usage(chunk)) is not None:
                        usage = u

        except httpx.TimeoutException:
            return self._failed(request_id, start_ns, M.STATUS_TIMEOUT, "request timed out")
        except httpx.HTTPError as exc:
            return self._failed(request_id, start_ns, M.STATUS_CONNECT, str(exc))

        end_ns = time.perf_counter_ns()
        in_tok, out_tok = M.usage_counts(usage)
        if in_tok is None:
            in_tok = input_tokens
        if not timed_chunks:
            status, error = M.STATUS_TRUNCATED, "stream closed with no tokens"

        return M.build_request_metrics(
            request_id,
            start_ns,
            end_ns,
            timed_chunks,
            input_tokens=in_tok,
            output_tokens=out_tok,
            status=status,
            error=error,
            queue_time_s=queue_time_s,
            trace_id=trace_id,
        )

    def _failed(self, request_id: str, start_ns: int, status: str, error: str) -> M.RequestMetrics:
        return M.build_request_metrics(
            request_id,
            start_ns,
            time.perf_counter_ns(),
            [],
            status=status,
            error=error,
        )


def _safe_float(raw: str) -> float | None:
    try:
        return float(raw)
    except ValueError:
        return None


async def run_concurrent(
    client: BenchmarkClient,
    prompts: list[tuple[str, int, int | None]],  # (text, max_tokens, approx_input_tokens)
    concurrency: int,
) -> list[M.RequestMetrics]:
    """Fire every prompt, at most `concurrency` in flight.

    A gate rather than a fixed number of workers: each slot immediately takes
    the next queued prompt, so a slow request cannot idle its slot.

    Each slot is a different virtual user. The scheduler caps one user's
    share of the GPU, so a single user for the whole batch would measure that
    cap instead of the concurrency axis — every point above the cap would look
    like the cap. A slot is handed back when its request finishes, so the
    batch runs as `concurrency` independent closed-loop clients.
    """
    slots: asyncio.Queue[int] = asyncio.Queue()
    for i in range(max(1, concurrency)):
        slots.put_nowait(i)
    results: list[M.RequestMetrics | None] = [None] * len(prompts)

    async def one(idx: int, text: str, max_tokens: int, input_tokens: int | None) -> None:
        slot = await slots.get()
        try:
            results[idx] = await client.complete(
                text,
                max_tokens,
                request_id=f"r{idx:05d}",
                input_tokens=input_tokens,
                user_id=f"{client.user_id}-u{slot}",
            )
        finally:
            slots.put_nowait(slot)

    await asyncio.gather(
        *(one(i, t, mt, it) for i, (t, mt, it) in enumerate(prompts))
    )
    return [r for r in results if r is not None]
