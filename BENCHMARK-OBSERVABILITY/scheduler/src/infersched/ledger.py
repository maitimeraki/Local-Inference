"""The reservation ledger.

Two numbers, kept apart on purpose:

  * `reserved` — what the ledger has committed to the requests it admitted.
    This is what admission control adds up.
  * `allocated` — the KV blocks the engine has actually granted to those
    requests, counted as the *union* of their block ids. A prefix-cache hit
    shares blocks between two live requests, so summing per-request counts
    double-counts and the drift metric would never read zero.

`reserved` is not a static worst case. It is what the engine has actually
granted, plus what the live requests may still add. A request that has been
handed its blocks has stopped promising them, and a prefix cache hit costs
what the engine says it costs rather than what the request asked for.

`drift` is what the engine holds minus what the ledger accounts for. In
steady state it is exactly zero. Anything else means the engine is holding
blocks nobody claimed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Reservation:
    """One admitted request's claim on the KV cache."""

    request_id: str
    user_id: str
    needed_blocks: int
    block_ids: set[int] = field(default_factory=set)
    admitted_ts: float = field(default_factory=time.monotonic)

    @property
    def actual_blocks(self) -> int:
        return len(self.block_ids)

    @property
    def outstanding_blocks(self) -> int:
        """Blocks this request may still take from the pool.

        `needed` is the worst case it was promised; `actual` is what the
        engine has already handed it. The difference is what it may still
        add. Never negative: a request that outgrew its promise is the
        pool's problem, not the promise's.
        """
        return max(0, self.needed_blocks - self.actual_blocks)


class ReservationLedger:
    """What we promised, and what the engine actually gave us."""

    def __init__(self) -> None:
        self._reservations: dict[str, Reservation] = {}
        self.total_blocks: int | None = None
        self.free_blocks: int | None = None

    # -- engine truth ---------------------------------------------------------

    def observe(self, *, free_blocks: int | None = None, total_blocks: int | None = None) -> None:
        if free_blocks is not None:
            self.free_blocks = free_blocks
        if total_blocks is not None:
            self.total_blocks = total_blocks

    def observe_blocks(self, request_id: str, block_ids: set[int]) -> None:
        res = self._reservations.get(request_id)
        if res is not None:
            res.block_ids = set(block_ids)
            # The promise has to cover what the engine actually took. A
            # request that outgrew its reservation must stop the ledger from
            # handing the same blocks to somebody else, or the pool is
            # oversold and the engine preempts to cover the difference.
            res.needed_blocks = max(res.needed_blocks, res.actual_blocks)

    # -- ledger truth ---------------------------------------------------------

    def reserve(self, request_id: str, user_id: str, blocks: int) -> Reservation:
        if request_id in self._reservations:
            raise ValueError(f"request {request_id} is already reserved")
        res = Reservation(request_id=request_id, user_id=user_id, needed_blocks=blocks)
        self._reservations[request_id] = res
        return res

    def release(self, request_id: str) -> int:
        """Drop a reservation. Returns the blocks it had promised, 0 if unknown."""
        res = self._reservations.pop(request_id, None)
        return 0 if res is None else res.needed_blocks

    def get(self, request_id: str) -> Reservation | None:
        return self._reservations.get(request_id)

    @property
    def reserved_blocks(self) -> int:
        """Blocks the ledger has committed to the live requests.

        Two parts, and both of them are needed. The blocks the engine has
        already granted, counted as a union so a shared prefix counts once.
        Plus what the live requests may still add, which is what they were
        promised minus what they have already taken.

        Promising every request its whole worst case instead over-promises
        by the size of the prefix cache: 28 requests sharing one 2000-token
        prefix cost 125 blocks between them, not 28 x 129. That over-promise
        is not a rounding error — it swallows the whole pool and stops the
        scheduler from running work the engine has room for.
        """
        ids: set[int] = set()
        outstanding = 0
        for res in self._reservations.values():
            ids |= res.block_ids
            outstanding += res.outstanding_blocks
        return len(ids) + outstanding

    @property
    def allocated_blocks(self) -> int:
        """Blocks the engine granted to tracked requests, shared blocks counted once."""
        ids: set[int] = set()
        for res in self._reservations.values():
            ids |= res.block_ids
        return len(ids)

    def reserved_by_user(self, user_id: str) -> int:
        """The same commitment, counted for one user."""
        ids: set[int] = set()
        outstanding = 0
        for res in self._reservations.values():
            if res.user_id != user_id:
                continue
            ids |= res.block_ids
            outstanding += res.outstanding_blocks
        return len(ids) + outstanding

    @property
    def headroom(self) -> int | None:
        """Blocks still promiseable.

        `total - reserved`, not `free - reserved`. Free counts the blocks the
        engine has not handed out; reserved counts the blocks we have already
        promised. Subtracting one from the other double-counts the promises
        that have not been granted yet, and under-promises capacity.
        """
        if self.total_blocks is None:
            return None
        return self.total_blocks - self.reserved_blocks

    @property
    def drift(self) -> int | None:
        """Blocks the engine holds that the ledger does not account for."""
        if self.total_blocks is None or self.free_blocks is None:
            return None
        return (self.total_blocks - self.free_blocks) - self.allocated_blocks
