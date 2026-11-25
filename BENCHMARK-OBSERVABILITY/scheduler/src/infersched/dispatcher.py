"""The dispatcher: the only thing that decides what the engine runs.

One writer, one lock. Every path that could start an engine request goes
through here, so two concurrent admits cannot both believe they had room.
The lock is held across the decision *and* the reservation, which is what
makes the invariant hold: at no point can two requests see the same free
capacity.

The lock is reentrant because the engine thread finishes a request and
immediately pumps the hold queue, from inside a callback that the dispatcher
itself is holding the lock for.
"""

from __future__ import annotations

import asyncio
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from infersched.admission import (
    ADMIT,
    HOLD,
    REJECT,
    AdmissionInput,
    Decision,
    OutputLengths,
    decide,
)
from infersched.config import SchedulerConfig, resolve_max_in_flight
from infersched.ledger import ReservationLedger
from infersched.metrics import metrics
from infersched.registry import UserRegistry
from infersched.state import reconcile

HELD = "held"
RUNNING = "running"
DONE = "done"
ABORTED = "aborted"
FAILED = "failed"

TERMINAL = (DONE, ABORTED, FAILED)

# How many finished requests to keep around for inspection.
_HISTORY = 1024


@dataclass
class ScheduledRequest:
    """One request, from submission to outcome."""

    request_id: str
    user_id: str
    prompt: str
    prompt_tokens: int
    max_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int | None = None
    # Lower number, higher importance. The same convention vLLM uses under
    # its priority scheduling policy: the engine serves the smallest number
    # first and preempts the largest one last.
    priority: int = 10
    min_tokens: int = 0
    ignore_eos: bool = False
    state: str = HELD
    decision: str = ""
    reason: str = ""
    reserved_blocks: int = 0
    hold_seq: int = 0
    # Set once the request has reached the engine. The stream needs to know
    # it even if the request finished before anybody looked: "it ran" is not
    # the same fact as "it is running now".
    dispatched: bool = False
    submitted_ts: float = field(default_factory=time.monotonic)
    admitted_ts: float = 0.0
    held_ts: float = 0.0
    held_seconds: float = 0.0
    text: str = ""
    actual_prompt_tokens: int | None = None
    completion_tokens: int = 0
    finish_reason: str = ""
    error: str | None = None
    events: asyncio.Queue | None = None
    done: asyncio.Event | None = None


class Dispatcher:
    """Admits, holds, and dispatches requests against a KV cache."""

    def __init__(
        self,
        cfg: SchedulerConfig,
        *,
        ledger: ReservationLedger | None = None,
        registry: UserRegistry | None = None,
        engine: Any = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self.cfg = cfg
        self.ledger = ledger or ReservationLedger()
        self.registry = registry or UserRegistry()
        self.engine = engine
        self.loop = loop
        self.m = metrics()

        self.block_size: int | None = None
        self.max_model_len: int | None = None
        # The scheduler's ceiling on concurrent engine requests. Resolved
        # against the engine when it attaches, because the engine is the only
        # one that knows how many sequences it can actually run.
        self.max_in_flight = resolve_max_in_flight(cfg.max_in_flight, engine)

        self._lock = threading.RLock()
        self._requests: dict[str, ScheduledRequest] = {}
        # Measured output lengths, for `adaptive` reservation.
        self._output_lengths = OutputLengths()
        # Users with a non-empty wait queue, in round-robin order.
        self._rr: list[str] = []
        self._rr_cursor: str | None = None
        self._preemptions_seen = 0
        # When the last slot became free. The dispatch latency is measured
        # from here, not from when the request was submitted: the hold time
        # belongs in `hold_seconds`, and a metric that mixes the two cannot
        # tell "the event path is broken" from "the cache is busy".
        self._slot_freed_ts: float | None = None

    # -- lifecycle -----------------------------------------------------------

    def attach_engine(self, engine: Any, block_size: int | None, max_model_len: int | None) -> None:
        with self._lock:
            self.engine = engine
            self.block_size = block_size
            self.max_model_len = max_model_len
            self.max_in_flight = resolve_max_in_flight(self.cfg.max_in_flight, engine)

    @property
    def ready(self) -> bool:
        return self.engine is not None and self.block_size is not None

    def get(self, request_id: str) -> ScheduledRequest | None:
        with self._lock:
            return self._requests.get(request_id)

    def inflight_count(self) -> int:
        return self.registry.total_inflight()

    def held_count(self) -> int:
        return self.registry.total_queued()

    # -- admission -----------------------------------------------------------

    def submit(self, req: ScheduledRequest) -> Decision:
        """Admit or hold a request. The only path that starts engine work."""
        with self._lock:
            if not self.ready:
                return Decision(REJECT, "engine_not_ready")
            if req.request_id in self._requests:
                # A duplicate submit must not start a second engine request.
                return Decision(REJECT, "duplicate")

            decision = decide(self._admission_input(req))
            req.decision = decision.decision
            req.reason = decision.reason
            self._requests[req.request_id] = req
            self.m["dispatch"].labels(decision=decision.decision).inc()

            if decision.decision == REJECT:
                req.state = FAILED
                req.error = decision.reason
                self.m["requests"].labels(status="rejected").inc()
                self._mark_done(req)
                return decision

            if decision.decision == HOLD:
                self._enqueue(req)
                return decision

            self._start(req, decision.reserved_blocks)
            return decision

    def abort(self, request_id: str, reason: str = "abort") -> int:
        """Cancel a request wherever it is. Returns the blocks released.

        `reason` is what the client is told. The scheduler uses
        `capacity_breach` when it sheds a request to keep the pool alive,
        so the client can tell "retry this" from "you cancelled this".
        """
        with self._lock:
            req = self._requests.get(request_id)
            if req is None or req.state in TERMINAL:
                return 0

            released = 0
            was_held = req.state == HELD
            if was_held:
                self.registry.cancel_queued(req.user_id, request_id)
                self._prune_users()
            else:
                released = self.ledger.release(request_id)
                self.registry.finish(req.user_id, request_id)

            # The outcome is decided here, so the engine's abort event must
            # not decide it again: the state is set *before* the engine is
            # told, or the finish handler would release the ledger a second
            # time and count the request under a reason nobody asked for.
            req.state = ABORTED
            req.finish_reason = reason
            self.m["requests"].labels(status="aborted").inc()

            if not was_held and self.engine is not None:
                self.engine.abort(request_id)

            self._mark_done(req)
            self._slot_freed_ts = time.monotonic()
            self.pump()
            return released

    # -- dispatch ------------------------------------------------------------

    def pump(self) -> None:
        """Dispatch held requests, round-robin across users, while there is room.

        The cursor names the user to try next and only moves past a user that
        was *served*. A user whose request does not fit keeps its place at the
        head of its own queue and is tried again on the next pump, so nobody
        is starved and nobody gets two turns in a row.
        """
        with self._lock:
            if not self.ready:
                return
            self._prune_users()
            order = list(self._rr)
            if not order:
                return
            start = order.index(self._rr_cursor) if self._rr_cursor in order else 0

            for turn in range(len(order)):
                user_id = order[(start + turn) % len(order)]
                request_id = self.registry.dequeue(user_id)
                if request_id is None:
                    continue
                req = self._requests.get(request_id)
                if req is None:
                    continue
                decision = decide(self._admission_input(req))
                if decision.decision != ADMIT:
                    self.registry.requeue_front(user_id, request_id)
                    if decision.reason == "capacity":
                        # The in-flight ceiling binds for everyone; there is no
                        # point asking anybody else.
                        break
                    # A KV shortfall might still let a smaller request in.
                    continue
                self._start(req, decision.reserved_blocks)
                self._rr_cursor = order[(start + turn + 1) % len(order)]

            self._prune_users()

    def _enqueue(self, req: ScheduledRequest) -> None:
        self.registry.enqueue(req.user_id, req.request_id)
        self._ensure_user(req.user_id)
        req.state = HELD
        req.held_ts = time.monotonic()
        req.hold_seq = self.registry.queued_count(req.user_id)
        self.m["hold_depth"].set(self.held_count())

    def _start(self, req: ScheduledRequest, reserved_blocks: int) -> None:
        self.ledger.reserve(req.request_id, req.user_id, reserved_blocks)
        self.registry.admit(req.user_id, req.request_id)
        req.reserved_blocks = reserved_blocks
        req.admitted_ts = time.monotonic()
        req.state = RUNNING
        req.dispatched = True
        if req.held_ts:
            req.held_seconds = req.admitted_ts - req.held_ts
            self.m["hold_seconds"].observe(req.held_seconds)
            if self._slot_freed_ts is not None:
                self.m["dispatch_latency"].observe(req.admitted_ts - self._slot_freed_ts)
        else:
            # Never held: the only thing between the submit and the engine is
            # the admission decision itself.
            self.m["dispatch_latency"].observe(req.admitted_ts - req.submitted_ts)
        self.m["inflight"].set(self.inflight_count())
        self.m["hold_depth"].set(self.held_count())

        params = self.engine.make_params(
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            seed=req.seed,
            min_tokens=req.min_tokens,
            ignore_eos=req.ignore_eos,
        )
        self.engine.submit(
            req.request_id,
            req.prompt,
            params,
            on_event=lambda event, r=req: self._on_event(r, event),
            priority=req.priority,
        )
        # Wake the request's stream: it is holding a request it thinks is
        # still queued, and the client is counting it as waiting.
        self._publish(req, None)

    def _ensure_user(self, user_id: str) -> None:
        if user_id not in self._rr:
            self._rr.append(user_id)

    def _prune_users(self) -> None:
        """Drop users whose wait queue is empty."""
        self._rr = [u for u in self._rr if self.registry.state(u).queued]
        if self._rr_cursor not in self._rr:
            self._rr_cursor = self._rr[0] if self._rr else None

    # -- engine events -------------------------------------------------------

    def _on_event(self, req: ScheduledRequest, event: Any) -> None:
        if event.delta:
            req.text += event.delta
        if event.prompt_tokens is not None:
            req.actual_prompt_tokens = event.prompt_tokens
        req.completion_tokens = event.completion_tokens
        self._publish(req, event)
        if event.finished:
            self._finish(req, event)

    def _finish(self, req: ScheduledRequest, event: Any) -> None:
        with self._lock:
            if req.state in TERMINAL:
                return
            self.ledger.release(req.request_id)
            self.registry.finish(req.user_id, req.request_id)

            if event.error:
                req.state = FAILED
                req.error = event.error
                self.m["requests"].labels(status="error").inc()
            elif event.finish_reason == "abort":
                req.state = ABORTED
                req.finish_reason = event.finish_reason or "abort"
                self.m["requests"].labels(status="aborted").inc()
            else:
                req.state = DONE
                req.finish_reason = event.finish_reason
                self.m["requests"].labels(status="completed").inc()
                # What this request really cost. `adaptive` reservation
                # promises against this number, so a completed request has to
                # report it, or the promise stays a guess.
                self._output_lengths.observe(req.user_id, req.completion_tokens)

            self._mark_done(req)
            self._prune()
            # A slot freed: give it to whoever has been waiting longest.
            self._slot_freed_ts = time.monotonic()
            self.pump()

    def _publish(self, req: ScheduledRequest, event: Any) -> None:
        if req.events is None:
            return
        if self.loop is None:
            req.events.put_nowait(event)
            return
        try:
            self.loop.call_soon_threadsafe(self._put, req.events, event)
        except RuntimeError:
            pass  # the loop is gone; the request is over anyway

    @staticmethod
    def _put(queue: asyncio.Queue, event: Any) -> None:
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            # The consumer is behind. The full text is on the request itself,
            # so only the live stream degrades, not the result.
            pass

    def _mark_done(self, req: ScheduledRequest) -> None:
        if req.done is None:
            return
        if self.loop is None:
            req.done.set()
            return
        try:
            self.loop.call_soon_threadsafe(req.done.set)
        except RuntimeError:
            pass

    # -- reconciliation ------------------------------------------------------

    def reconcile(self) -> int | None:
        """Refresh the ledger from the engine. Returns the drift."""
        with self._lock:
            if self.engine is None:
                return None
            live = [r.request_id for r in self._requests.values() if r.state == RUNNING]
            drift = reconcile(self.engine, self.ledger, live)

            m = self.m
            m["free_blocks"].set(self.ledger.free_blocks or 0)
            m["total_blocks"].set(self.ledger.total_blocks or 0)
            m["reserved_blocks"].set(self.ledger.reserved_blocks)
            m["allocated_blocks"].set(self.ledger.allocated_blocks)
            if drift is not None:
                m["drift"].set(drift)
            m["inflight"].set(self.inflight_count())
            m["hold_depth"].set(self.held_count())

            preemptions = self.engine.preemptions()
            delta = preemptions - self._preemptions_seen
            if delta > 0:
                m["preemptions"].inc(delta)
                self._preemptions_seen = preemptions
                self._shed_preempted(delta)
            return drift

    def _shed_preempted(self, count: int) -> None:
        """Turn the engine's preemption into a visible, retryable failure.

        vLLM preempted, which means the pool ran out while the request was
        being generated: it will recompute that request from scratch and the
        client sees nothing but latency. The scheduler knows which of its
        requests matters least, so it sheds that one instead. The client gets
        a 503 and retries; the pool gets the blocks back now.
        """
        for _ in range(count):
            victim = self._pick_victim()
            if victim is None:
                return
            self.m["capacity_breaches"].inc()
            self.abort(victim, reason="capacity_breach")

    def _pick_victim(self) -> str | None:
        """The running request that matters least.

        The same rule vLLM uses under its priority policy: the largest
        priority number goes, and among equals the one that arrived last.
        """
        running = [r for r in self._requests.values() if r.state == RUNNING]
        if not running:
            return None
        return max(running, key=lambda r: (r.priority, r.admitted_ts)).request_id

    def _in_flight_remaining_blocks(self) -> int:
        """Blocks the running requests still need for the output they have left.

        The worst case for each: every running request generates its full
        `max_tokens`. That is what `strict` promised for them, so this is the
        room the pool has to keep free until they finish.
        """
        block_size = self.block_size or 1
        total = 0
        for req in self._requests.values():
            if req.state != RUNNING:
                continue
            left = max(0, req.max_tokens - req.completion_tokens)
            total += math.ceil(left / block_size)
        return total

    # -- bookkeeping ---------------------------------------------------------

    def _prune(self) -> None:
        """Keep the request history bounded."""
        if len(self._requests) <= _HISTORY:
            return
        for request_id, req in list(self._requests.items()):
            if req.state in TERMINAL:
                del self._requests[request_id]
                if len(self._requests) <= _HISTORY:
                    return

    def _admission_input(self, req: ScheduledRequest) -> AdmissionInput:
        return AdmissionInput(
            request_id=req.request_id,
            user_id=req.user_id,
            prompt_tokens=req.prompt_tokens,
            max_tokens=req.max_tokens,
            block_size=self.block_size or 1,
            max_model_len=self.max_model_len,
            total_blocks=self.ledger.total_blocks,
            reserved_blocks=self.ledger.reserved_blocks,
            inflight=self.inflight_count(),
            user_inflight=self.registry.inflight_count(req.user_id),
            user_reserved_blocks=self.ledger.reserved_by_user(req.user_id),
            user_queued=self.registry.queued_count(req.user_id),
            queued_total=self.held_count(),
            max_in_flight=self.max_in_flight,
            max_in_flight_per_user=self.cfg.max_in_flight_per_user,
            max_queue_depth_global=self.cfg.max_queue_depth_global,
            max_queue_depth_per_user=self.cfg.max_queue_depth_per_user,
            max_reserved_blocks_per_user=self.cfg.max_reserved_blocks_per_user,
            reservation_mode=self.cfg.reservation_mode,
            free_blocks=self.ledger.free_blocks,
            in_flight_remaining_blocks=self._in_flight_remaining_blocks(),
            expected_output_tokens=self._output_lengths.p95(req.user_id),
        )
