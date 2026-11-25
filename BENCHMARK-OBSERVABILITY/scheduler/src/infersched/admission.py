"""The admission decision: admit, hold, or reject.

Pure on purpose. Every input is in `AdmissionInput` and nothing is read
from the engine, the clock, or the filesystem, so the whole truth table is
testable without a GPU.

Order matters: the rejects that cannot wait come first, then the holds. A
request that can never be admitted — too long for the context, or bigger
than its user's whole block budget — must be rejected now, not held
forever in a queue that will never drain.

Everything else waits. A user over their concurrency cap is not an error:
the request joins that user's queue and runs when the user's slot frees.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

ADMIT = "admit"
HOLD = "hold"
REJECT = "reject"


@dataclass(frozen=True)
class AdmissionInput:
    """Everything the decision depends on, and nothing else."""

    request_id: str
    user_id: str
    prompt_tokens: int
    max_tokens: int
    block_size: int
    max_model_len: int | None = None
    total_blocks: int | None = None
    reserved_blocks: int = 0
    inflight: int = 0
    user_inflight: int = 0
    user_reserved_blocks: int = 0
    user_queued: int = 0
    queued_total: int = 0
    max_in_flight: int = 32
    max_in_flight_per_user: int = 4
    max_queue_depth_global: int = 256
    max_queue_depth_per_user: int = 16
    max_reserved_blocks_per_user: int = 512
    reservation_mode: str = "strict"
    # The engine's free blocks, as the engine last reported them. The
    # promise says what we have agreed to hand out; this says what is
    # physically there. They are not the same number, and only one of them
    # is a fact.
    free_blocks: int | None = None
    # Blocks the running requests still need for the output they have left.
    in_flight_remaining_blocks: int = 0
    # Measured p95 output length for this user, for `adaptive` reservation.
    expected_output_tokens: int | None = None


@dataclass(frozen=True)
class Decision:
    decision: str
    reason: str = ""
    reserved_blocks: int = 0
    # What this request needs. Carried out of every branch, including the
    # holds and the rejects: a held request still needs blocks, and a client
    # that cannot see the number cannot tell a busy cache from a busy user.
    demand: int = 0

    @property
    def admitted(self) -> bool:
        return self.decision == ADMIT


class OutputLengths:
    """Rolling p95 of the output tokens requests actually produced.

    The measurement `adaptive` reservation needs. Kept per user, because one
    user's workload is not another's, and bounded, because a long-lived
    process must not grow without limit.
    """

    def __init__(self, window: int = 256, min_samples: int = 20) -> None:
        self._window = window
        self._min_samples = min_samples
        self._by_user: dict[str, list[int]] = {}

    def observe(self, user_id: str, tokens: int) -> None:
        samples = self._by_user.setdefault(user_id, [])
        samples.append(tokens)
        if len(samples) > self._window:
            del samples[: len(samples) - self._window]

    def p95(self, user_id: str) -> int | None:
        """The p95 output length for this user, or None if it is not known yet.

        None until there are enough samples to mean anything: a p95 over one
        request is that request, and reserving against it would be a promise
        built from a guess.
        """
        samples = self._by_user.get(user_id)
        if not samples or len(samples) < self._min_samples:
            return None
        ordered = sorted(samples)
        return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def block_demand(
    prompt_tokens: int,
    max_tokens: int,
    block_size: int,
    mode: str = "strict",
    expected_output_tokens: int | None = None,
) -> tuple[int, int]:
    """(prompt_blocks, reserved_blocks) for one request.

    `prompt_blocks` is what the prompt alone needs. `reserved_blocks` is what
    we promise: in `strict` mode the prompt plus the worst-case output, so
    that the sum of all promises fits inside the cache.

    `adaptive` promises the prompt plus the measured p95 output, which is
    what the workload actually costs. With no measurement yet it falls back
    to `strict`: an unmeasured promise is no promise at all.
    """
    prompt_blocks = math.ceil(prompt_tokens / block_size)
    output_blocks = math.ceil(max_tokens / block_size)
    if mode == "prompt_only":
        return prompt_blocks, prompt_blocks
    if mode == "adaptive" and expected_output_tokens is not None:
        p95 = min(expected_output_tokens, max_tokens)
        return prompt_blocks, prompt_blocks + math.ceil(p95 / block_size)
    return prompt_blocks, prompt_blocks + output_blocks


def decide(inp: AdmissionInput) -> Decision:
    """Decide one request."""
    _, demand = block_demand(
        inp.prompt_tokens,
        inp.max_tokens,
        inp.block_size,
        inp.reservation_mode,
        inp.expected_output_tokens,
    )

    if inp.max_model_len is not None and inp.prompt_tokens + inp.max_tokens > inp.max_model_len:
        return Decision(REJECT, "too_long", demand=demand)

    if demand > inp.max_reserved_blocks_per_user:
        # It can never fit in this user's budget, however long we wait.
        return Decision(REJECT, "user_reserved", demand=demand)
    if inp.user_queued >= inp.max_queue_depth_per_user:
        return Decision(REJECT, "user_queue_full", demand=demand)
    if inp.queued_total >= inp.max_queue_depth_global:
        return Decision(REJECT, "queue_full", demand=demand)

    if inp.user_reserved_blocks + demand > inp.max_reserved_blocks_per_user:
        # It fits, but not while the user's other requests hold their blocks.
        return Decision(HOLD, "user_reserved", demand=demand)
    if inp.user_inflight >= inp.max_in_flight_per_user:
        # Over the user's own cap: wait for one of the user's slots.
        return Decision(HOLD, "user_concurrency", demand=demand)
    if inp.inflight >= inp.max_in_flight:
        return Decision(HOLD, "capacity", demand=demand)
    if inp.total_blocks is not None and inp.reserved_blocks + demand > inp.total_blocks:
        return Decision(HOLD, "kv_capacity", demand=demand)
    if (
        inp.free_blocks is not None
        and inp.free_blocks - demand < inp.in_flight_remaining_blocks
    ):
        # The promise fits and the pool does not. Admitting now is what makes
        # vLLM preempt, silently, mid-generation: the client waits and the
        # tokens are recomputed. Hold instead, and the pool keeps the room
        # the running requests need to finish.
        return Decision(HOLD, "physical_capacity", demand=demand)

    return Decision(ADMIT, "fit", reserved_blocks=demand, demand=demand)
