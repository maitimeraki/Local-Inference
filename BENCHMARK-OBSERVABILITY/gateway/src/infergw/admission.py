"""Admission control.

A capacity ceiling with an explicit rejection, rather than letting the engine
absorb unbounded load. This is a deliberate choice for two reasons:

  * it makes queueing *observable* — an overloaded gateway sheds load and
    reports it, instead of hiding the overload inside the engine's own queue;
  * it protects the engine from a thundering herd, which on a 6 GB card is the
    difference between slow and OOM.

Rejections are 429s, counted. They appear on the dashboard as a real signal,
not an error to be suppressed.
"""

from __future__ import annotations


class Admission:
    """Non-blocking in-flight limiter.

    Deliberately not an `asyncio.Semaphore`: we want to *refuse* work, not wait
    for it, and a semaphore has no non-blocking acquire. The counter needs no
    lock because `try_acquire` contains no `await`, so nothing can interleave
    between the check and the increment.

    Two counters, because the scheduler has two: the requests it is running
    and the requests it is holding. Adding them together makes the gateway's
    number disagree with the scheduler's, and refusing work at the gateway
    while the engine still has free slots is exactly the bug this splits.

    A request counts as *waiting* until the scheduler says it is running:
    the outcome is not known when the slot is taken, and waiting is the
    pessimistic assumption for a ceiling.
    """

    def __init__(self, limit: int) -> None:
        if limit < 1:
            raise ValueError("max_in_flight must be >= 1")
        self.limit = limit
        self._running = 0
        self._waiting = 0
        self._states: dict[str, str] = {}

    @property
    def in_flight(self) -> int:
        """Everything the gateway is relaying: running plus waiting."""
        return self._running + self._waiting

    @property
    def running(self) -> int:
        return self._running

    @property
    def waiting(self) -> int:
        return self._waiting

    @property
    def saturated(self) -> bool:
        return self.in_flight >= self.limit

    def set_limit(self, limit: int) -> None:
        """Adopt a ceiling learned from the scheduler at startup."""
        if limit >= 1:
            self.limit = limit

    def try_acquire(self, request_id: str) -> bool:
        if self.in_flight >= self.limit:
            return False
        self._waiting += 1
        self._states[request_id] = "waiting"
        return True

    def mark_running(self, request_id: str) -> None:
        """The scheduler started this request."""
        if self._states.get(request_id) == "waiting":
            self._waiting -= 1
            self._running += 1
            self._states[request_id] = "running"

    def release(self, request_id: str) -> None:
        state = self._states.pop(request_id, None)
        if state == "running":
            self._running = max(0, self._running - 1)
        elif state == "waiting":
            self._waiting = max(0, self._waiting - 1)
