"""The reservation ledger.

The ledger is the scheduler's memory of what it promised the engine. Two
things can go wrong and both are silent: promising more than the cache
holds (the engine preempts), and promising less than the engine took (the
ledger hands out capacity that is already gone). The drift metric is the
one that catches the second.
"""

from __future__ import annotations

from infersched.ledger import ReservationLedger


def test_reserve_and_release():
    ledger = ReservationLedger()
    ledger.observe(free_blocks=100, total_blocks=100)

    ledger.reserve("r1", "u1", 10)
    ledger.reserve("r2", "u2", 5)

    assert ledger.reserved_blocks == 15
    assert ledger.reserved_by_user("u1") == 10
    assert ledger.reserved_by_user("u2") == 5

    assert ledger.release("r1") == 10
    assert ledger.reserved_blocks == 5
    # Releasing twice is not an error: the abort path and the finish path can
    # both reach the same request.
    assert ledger.release("r1") == 0


def test_duplicate_reserve_is_rejected():
    ledger = ReservationLedger()
    ledger.reserve("r1", "u1", 10)
    try:
        ledger.reserve("r1", "u1", 10)
    except ValueError:
        return
    raise AssertionError("a second reservation for the same request must fail")


def test_allocated_counts_shared_blocks_once():
    """A prefix-cache hit puts the same block in two live requests.

    Summing per-request counts would claim two blocks where the engine holds
    one, and the drift metric could never read zero.
    """
    ledger = ReservationLedger()
    ledger.reserve("r1", "u1", 4)
    ledger.reserve("r2", "u1", 4)

    ledger.observe_blocks("r1", {1, 2, 3})
    ledger.observe_blocks("r2", {3, 4})  # block 3 is shared with r1

    assert ledger.allocated_blocks == 4
    # 4 granted, plus 1 still owed by r1 and 2 still owed by r2. Block 3 is
    # shared, so it is owed by nobody twice.
    assert ledger.reserved_blocks == 7


def test_a_granted_request_stops_promising_its_blocks():
    """The promise tracks what the engine actually took.

    Two requests share one prefix, so the pool holds four blocks between
    them, not eight. A ledger that kept promising eight would tell
    admission control that the pool is full while the engine has room for
    more work — the pool's real cost is four blocks plus what is still
    outstanding, and that is what the ledger has to say.
    """
    ledger = ReservationLedger()
    ledger.observe(free_blocks=96, total_blocks=100)
    ledger.reserve("r1", "u1", 4)
    ledger.reserve("r2", "u1", 4)

    # Neither has been scheduled yet, so both still owe their full promise.
    assert ledger.reserved_blocks == 8
    assert ledger.headroom == 92

    # The engine hands both of them the same four blocks (a prefix hit).
    ledger.observe_blocks("r1", {1, 2, 3, 4})
    ledger.observe_blocks("r2", {1, 2, 3, 4})

    assert ledger.allocated_blocks == 4
    assert ledger.reserved_blocks == 4
    # The pool is not full. 96 blocks are still free and 96 are still
    # promiseable, so the next request fits.
    assert ledger.headroom == 96


def test_a_request_that_outgrew_its_promise_is_not_under_promised():
    """The promise never falls below what the engine actually took."""
    ledger = ReservationLedger()
    ledger.reserve("r1", "u1", 2)
    ledger.observe_blocks("r1", {1, 2, 3, 4})

    # r1 was promised 2 and took 4: it owes nothing more, but the ledger
    # must not claim it owes less than it holds.
    assert ledger.reserved_blocks == 4
    assert ledger.get("r1").outstanding_blocks == 0


def test_drift_is_zero_when_the_ledger_accounts_for_everything():
    ledger = ReservationLedger()
    ledger.reserve("r1", "u1", 3)
    ledger.observe_blocks("r1", {1, 2, 3})

    # The engine holds 3 of 100 blocks, and the ledger accounts for 3.
    ledger.observe(free_blocks=97, total_blocks=100)
    assert ledger.drift == 0


def test_drift_is_non_zero_when_the_engine_holds_unclaimed_blocks():
    ledger = ReservationLedger()
    ledger.reserve("r1", "u1", 3)
    ledger.observe_blocks("r1", {1, 2, 3})

    # The engine holds 5 blocks but the ledger only knows about 3.
    ledger.observe(free_blocks=95, total_blocks=100)
    assert ledger.drift == 2


def test_drift_is_none_before_the_engine_has_been_read():
    ledger = ReservationLedger()
    assert ledger.drift is None
    assert ledger.headroom is None


def test_headroom_is_total_minus_reserved():
    """Not `free - reserved`.

    Free counts blocks the engine has not handed out; reserved counts the
    blocks we have already promised. Subtracting one from the other
    double-counts promises that have not been granted yet, and
    under-promises capacity.
    """
    ledger = ReservationLedger()
    ledger.observe(free_blocks=100, total_blocks=100)
    ledger.reserve("r1", "u1", 30)

    # The engine has granted nothing yet: it still has 100 free blocks, but
    # 30 of them are spoken for.
    assert ledger.headroom == 70
    assert ledger.free_blocks == 100
