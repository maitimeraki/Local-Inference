"""Per-user state: who is running, who is waiting."""

from __future__ import annotations

from infersched.registry import UserRegistry


def test_queue_is_fifo():
    reg = UserRegistry()
    reg.enqueue("u1", "r1")
    reg.enqueue("u1", "r2")
    assert reg.dequeue("u1") == "r1"
    assert reg.dequeue("u1") == "r2"
    assert reg.dequeue("u1") is None


def test_requeue_front_keeps_the_head():
    reg = UserRegistry()
    reg.enqueue("u1", "r1")
    reg.enqueue("u1", "r2")
    reg.dequeue("u1")
    reg.requeue_front("u1", "r1")
    assert reg.dequeue("u1") == "r1"


def test_inflight_and_queued_are_counted_separately():
    reg = UserRegistry()
    reg.admit("u1", "r1")
    reg.enqueue("u1", "r2")

    assert reg.inflight_count("u1") == 1
    assert reg.queued_count("u1") == 1
    assert reg.total_inflight() == 1
    assert reg.total_queued() == 1


def test_finish_frees_the_slot():
    reg = UserRegistry()
    reg.admit("u1", "r1")
    reg.finish("u1", "r1")
    assert reg.inflight_count("u1") == 0
    # Finishing twice is harmless.
    reg.finish("u1", "r1")
    assert reg.inflight_count("u1") == 0


def test_cancel_queued_removes_a_held_request():
    reg = UserRegistry()
    reg.enqueue("u1", "r1")
    reg.enqueue("u1", "r2")
    assert reg.cancel_queued("u1", "r1") is True
    assert reg.cancel_queued("u1", "r1") is False
    assert reg.queued_count("u1") == 1
    assert reg.dequeue("u1") == "r2"


def test_snapshot_is_per_user_counts():
    reg = UserRegistry()
    reg.admit("u1", "r1")
    reg.enqueue("u2", "r2")
    assert reg.snapshot() == {
        "u1": {"inflight": 1, "queued": 0},
        "u2": {"inflight": 0, "queued": 1},
    }
