"""The admission decision.

Pure function, whole truth table. No engine, no clock, no GPU — which is
the only reason the rules can be checked at all.
"""

from __future__ import annotations

from infersched.admission import ADMIT, HOLD, REJECT, AdmissionInput, block_demand, decide


def inp(**overrides) -> AdmissionInput:
    base = {
        "request_id": "r1",
        "user_id": "u1",
        "prompt_tokens": 100,
        "max_tokens": 50,
        "block_size": 16,
        "total_blocks": 100,
    }
    base.update(overrides)
    return AdmissionInput(**base)


def test_fits_and_is_admitted():
    d = decide(inp())
    assert d.decision == ADMIT
    assert d.reason == "fit"
    # 100 prompt tokens = 7 blocks of 16, 50 output tokens = 4 blocks.
    assert d.reserved_blocks == 11
    assert d.admitted


def test_too_long_is_rejected():
    d = decide(inp(max_model_len=120))
    assert d.decision == REJECT
    assert d.reason == "too_long"


def test_user_concurrency_cap_holds():
    """Over the user's own cap the request waits — it is not an error.

    A client may send as many requests as it likes; the scheduler runs the
    user's share and queues the rest.
    """
    d = decide(inp(user_inflight=4, max_in_flight_per_user=4))
    assert d.decision == HOLD
    assert d.reason == "user_concurrency"


def test_user_reserved_cap_holds_until_the_blocks_free():
    d = decide(inp(user_reserved_blocks=510, max_reserved_blocks_per_user=512))
    assert d.decision == HOLD
    assert d.reason == "user_reserved"


def test_a_request_bigger_than_the_users_whole_budget_is_rejected():
    """It can never fit, so waiting would hold it forever."""
    d = decide(inp(max_reserved_blocks_per_user=4))
    assert d.decision == REJECT
    assert d.reason == "user_reserved"


def test_user_queue_depth_cap_rejects():
    d = decide(inp(user_queued=16, max_queue_depth_per_user=16))
    assert d.decision == REJECT
    assert d.reason == "user_queue_full"


def test_global_queue_depth_cap_rejects():
    d = decide(inp(queued_total=256, max_queue_depth_global=256))
    assert d.decision == REJECT
    assert d.reason == "queue_full"


def test_in_flight_ceiling_holds_rather_than_rejects():
    """Over the ceiling the request waits — it is not an error."""
    d = decide(inp(inflight=32, max_in_flight=32))
    assert d.decision == HOLD
    assert d.reason == "capacity"


def test_kv_capacity_holds():
    d = decide(inp(total_blocks=100, reserved_blocks=95))
    assert d.decision == HOLD
    assert d.reason == "kv_capacity"


def test_kv_capacity_uses_reserved_not_free():
    """The promise, not the engine's current free count, is what binds.

    A request whose blocks are promised but not yet granted is still spoken
    for; counting it again would let the ledger promise more blocks than
    the cache has.
    """
    d = decide(inp(total_blocks=100, reserved_blocks=85))
    assert d.decision == ADMIT

    d = decide(inp(total_blocks=100, reserved_blocks=95))
    assert d.decision == HOLD


def test_rejects_are_checked_before_holds():
    """A full queue is refused even though the request could otherwise wait.

    The order matters: if the hold came first, a request could sit past the
    queue bound and the queue would grow without limit.
    """
    d = decide(
        inp(
            inflight=32,
            max_in_flight=32,
            user_queued=16,
            max_queue_depth_per_user=16,
        )
    )
    assert d.decision == REJECT
    assert d.reason == "user_queue_full"


def test_block_demand_strict_reserves_the_worst_case_output():
    prompt_blocks, reserved = block_demand(100, 50, 16, "strict")
    assert prompt_blocks == 7
    assert reserved == 11


def test_block_demand_prompt_only_reserves_the_prompt():
    prompt_blocks, reserved = block_demand(100, 50, 16, "prompt_only")
    assert prompt_blocks == 7
    assert reserved == 7


def test_block_demand_rounds_up():
    _, reserved = block_demand(1, 1, 16, "strict")
    assert reserved == 2


def test_prompt_only_mode_admits_what_strict_would_hold():
    strict = decide(inp(total_blocks=12, reserved_blocks=5, reservation_mode="strict"))
    assert strict.decision == HOLD

    loose = decide(inp(total_blocks=12, reserved_blocks=5, reservation_mode="prompt_only"))
    assert loose.decision == ADMIT


def test_physical_capacity_holds_when_the_pool_cannot_cover_the_run():
    """The promise fits; the engine's free blocks do not.

    Admitting here is what makes vLLM preempt mid-generation: the client
    waits and the tokens are recomputed. Holding instead keeps the room the
    running requests need to finish.
    """
    d = decide(inp(free_blocks=20, in_flight_remaining_blocks=15))
    assert d.decision == HOLD
    assert d.reason == "physical_capacity"


def test_adaptive_reservation_promises_the_measured_p95_output():
    """The workload's real cost, not the worst case the client asked for."""
    d = decide(inp(reservation_mode="adaptive", expected_output_tokens=40))
    # 100 prompt tokens = 7 blocks; a measured 40-token output = 3 blocks.
    assert d.reserved_blocks == 10


def test_adaptive_falls_back_to_strict_with_no_measurement():
    """An unmeasured promise is no promise at all."""
    d = decide(inp(reservation_mode="adaptive", expected_output_tokens=None))
    assert d.reserved_blocks == 11


def test_every_branch_carries_the_demand():
    """A held or refused request still needs blocks.

    The number is what separates "the cache is busy" from "you asked for
    too much" — and it is what the loadgen prints next to a reserved count
    of 0, so that a held request does not look like one that needed nothing.
    """
    cases = {
        "admit": inp(),
        "reject too_long": inp(max_model_len=120),
        "reject user_reserved": inp(max_reserved_blocks_per_user=4),
        "reject user_queue_full": inp(user_queued=16, max_queue_depth_per_user=16),
        "reject queue_full": inp(queued_total=256, max_queue_depth_global=256),
        "hold user_reserved": inp(user_reserved_blocks=510, max_reserved_blocks_per_user=512),
        "hold user_concurrency": inp(user_inflight=4, max_in_flight_per_user=4),
        "hold capacity": inp(inflight=32, max_in_flight=32),
        "hold kv_capacity": inp(total_blocks=100, reserved_blocks=95),
        "hold physical_capacity": inp(free_blocks=20, in_flight_remaining_blocks=15),
    }
    for label, case in cases.items():
        d = decide(case)
        assert d.demand == 11, f"{label} ({d.reason}) lost the demand"
