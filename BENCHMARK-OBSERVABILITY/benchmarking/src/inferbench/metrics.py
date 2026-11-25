"""Latency math and SSE parsing.

This is the module that decides whether every number the platform publishes is
true, so it is pure functions over timestamps and byte streams — no HTTP, no
vLLM, no clock reads. The client feeds it timings; the store consumes its
output. Everything here is testable without a GPU.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field

# --- Percentiles ------------------------------------------------------------


def percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolation percentile, matching `numpy.percentile` default.

    The exact method matters: nearest-rank and interpolated percentiles differ
    by several percent on small samples, and a benchmark that disagrees with
    the obvious reference implementation is a benchmark nobody trusts.
    """
    if not values:
        return None
    if not 0 <= q <= 100:
        raise ValueError(f"q must be in [0, 100], got {q}")
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    rank = (q / 100.0) * (len(xs) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return xs[int(rank)]
    return xs[lo] + (xs[hi] - xs[lo]) * (rank - lo)


def summarize(values: list[float]) -> dict[str, float]:
    """mean/p50/p95/p99/min/max. Empty input yields an empty dict, not zeros.

    Returning zeros for "no data" is how a dashboard ends up reporting a
    perfect score for a run that produced nothing.
    """
    if not values:
        return {}
    xs = sorted(values)
    return {
        "mean": sum(xs) / len(xs),
        "p50": percentile(xs, 50) or 0.0,
        "p95": percentile(xs, 95) or 0.0,
        "p99": percentile(xs, 99) or 0.0,
        "min": xs[0],
        "max": xs[-1],
    }


# --- Per-request record -----------------------------------------------------

# Status vocabulary. "ok" is the only success; everything else is classified
# rather than swallowed, so a run cannot silently report fewer requests than it
# sent.
STATUS_OK = "ok"
STATUS_HTTP = "http_error"
STATUS_TIMEOUT = "timeout"
STATUS_CONNECT = "connect_error"
STATUS_PARSE = "parse_error"
STATUS_TRUNCATED = "truncated"


@dataclass
class RequestMetrics:
    request_id: str
    status: str = STATUS_OK
    error: str | None = None
    ttft_s: float | None = None
    e2e_s: float | None = None
    tpot_s: float | None = None
    itl_mean_s: float | None = None
    itl_p95_s: float | None = None
    itls: list[float] = field(default_factory=list)
    input_tokens: int | None = None
    output_tokens: int | None = None
    queue_time_s: float | None = None
    trace_id: str | None = None
    started_at: float = 0.0
    finished_at: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK

    def to_row(self) -> dict:
        """Flat record for the store. `itls` is summarised, not stored raw."""
        d = asdict(self)
        d.pop("itls")
        d["ok"] = self.ok
        return d


def build_request_metrics(
    request_id: str,
    start_ns: int,
    end_ns: int,
    timed_chunks: list[tuple[int, int]],
    *,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    status: str = STATUS_OK,
    error: str | None = None,
    queue_time_s: float | None = None,
    trace_id: str | None = None,
) -> RequestMetrics:
    """Derive the latency fields from raw chunk timings.

    `timed_chunks` is `(monotonic_ns, tokens_in_this_chunk)` in arrival order.

    Definitions used (and the reason each is what it is):
      TTFT  first token's arrival minus request send. The number the user feels.
      ITL   gaps between successive tokens. When a chunk carries k>1 tokens the
            gap is split evenly across them, since the wire only gives us the
            chunk boundary.
      TPOT  (e2e - ttft) / (output_tokens - 1) — vLLM's definition, i.e. the
            per-token cost *after* the first, which excludes prefill.
      e2e   last token's arrival minus request send.
    """
    started = start_ns / 1e9
    finished = end_ns / 1e9
    content = [(t, k) for t, k in timed_chunks if k > 0]

    ttft_s: float | None = None
    e2e_s = (end_ns - start_ns) / 1e9
    itls: list[float] = []

    if content:
        ttft_s = (content[0][0] - start_ns) / 1e9
        e2e_s = (content[-1][0] - start_ns) / 1e9
        prev_t = content[0][0]
        for t, k in content[1:]:
            gap = (t - prev_t) / 1e9
            itls.extend([gap / k] * k)
            prev_t = t

    if output_tokens is None:
        output_tokens = sum(k for _, k in content) or None

    tpot_s: float | None = None
    if ttft_s is not None and output_tokens is not None and output_tokens > 1:
        tpot_s = (e2e_s - ttft_s) / (output_tokens - 1)

    return RequestMetrics(
        request_id=request_id,
        status=status,
        error=error,
        ttft_s=ttft_s,
        e2e_s=e2e_s,
        tpot_s=tpot_s,
        itl_mean_s=(sum(itls) / len(itls)) if itls else None,
        itl_p95_s=percentile(itls, 95),
        itls=itls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        queue_time_s=queue_time_s,
        trace_id=trace_id,
        started_at=started,
        finished_at=finished,
    )


# --- SSE parsing ------------------------------------------------------------


def parse_sse_data(line: bytes) -> dict | None:
    """Parse one SSE line. Returns None for comments, blanks, and `[DONE]`."""
    if not line:
        return None
    line = line.strip()
    if not line.startswith(b"data:"):
        return None
    payload = line[5:].strip()
    if not payload or payload == b"[DONE]":
        return None
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def chunk_text(chunk: dict) -> str:
    """Text of this chunk, for either /v1/completions or /v1/chat/completions."""
    choices = chunk.get("choices") or []
    if not choices:
        return ""
    c0 = choices[0]
    if "text" in c0:
        return c0.get("text") or ""
    delta = c0.get("delta") or {}
    return delta.get("content") or ""


def chunk_token_count(chunk: dict) -> int:
    """Tokens contributed by this chunk.

    vLLM streams one token per chunk, so a text-bearing chunk counts as one.
    Anything more precise comes from `usage` at the end of the stream.
    """
    return 1 if chunk_text(chunk) else 0


def chunk_usage(chunk: dict) -> dict | None:
    """The `usage` object, present on the final chunk when stream_options
    requests it."""
    usage = chunk.get("usage")
    return usage if isinstance(usage, dict) else None


def chunk_finish_reason(chunk: dict) -> str | None:
    choices = chunk.get("choices") or []
    if not choices:
        return None
    return choices[0].get("finish_reason")


def usage_counts(usage: dict | None) -> tuple[int | None, int | None]:
    """(input_tokens, output_tokens) from a usage object."""
    if not usage:
        return None, None
    return usage.get("prompt_tokens"), usage.get("completion_tokens")
