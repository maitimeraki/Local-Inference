"""The dispatcher: the only thing that decides what the engine runs.

The engine here is a fake — no GPU, no vLLM — but it honours the same
contract: `submit` takes work, `finish` reports it done. That is enough to
check the two things that matter: that a request which does not fit waits
rather than being dropped or preempted, and that one user cannot starve
another.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from inferbench.step_engine import StreamEvent
from inferobs import metric_names as N
from infersched.admission import ADMIT, HOLD, REJECT
from infersched.config import DEFAULT_MAX_IN_FLIGHT, SchedulerConfig, resolve_max_in_flight
from infersched.dispatcher import Dispatcher, ScheduledRequest
from prometheus_client import REGISTRY


class FakeEngine:
    """Just enough engine for the dispatcher."""

    def __init__(self, total_blocks: int = 1000, block_size: int = 16) -> None:
        self._total = total_blocks
        self.block_size = block_size
        self.max_model_len = 4096
        self.submitted: dict[str, dict[str, Any]] = {}
        self.blocks: dict[str, set[int]] = {}
        self.preempted = 0

    def make_params(self, **kwargs) -> dict:
        return kwargs

    def count_tokens(self, prompt: str) -> int:
        return len(prompt.split())

    def submit(self, request_id, prompt, params, on_event=None, priority: int = 0):
        self.submitted[request_id] = {
            "prompt": prompt, "params": params, "on_event": on_event, "priority": priority,
        }
        self.blocks.setdefault(request_id, set())

    def abort(self, request_id) -> bool:
        handle = self.submitted.pop(request_id, None)
        if handle is None:
            return False
        self.blocks.pop(request_id, None)
        handle["on_event"](StreamEvent(finished=True, finish_reason="abort"))
        return True

    def finish(self, request_id, text: str = "ok") -> None:
        handle = self.submitted.pop(request_id)
        self.blocks.pop(request_id, None)
        handle["on_event"](
            StreamEvent(delta=text, finished=True, finish_reason="stop", completion_tokens=1)
        )

    def hold(self, request_id, block_ids: set[int]) -> None:
        """Give a request some blocks, as the engine would."""
        self.blocks.setdefault(request_id, set()).update(block_ids)

    def free_blocks(self) -> int:
        used = set().union(*self.blocks.values()) if self.blocks else set()
        return self._total - len(used)

    def total_blocks(self) -> int:
        return self._total

    def request_block_ids(self, request_id) -> set[int]:
        return set(self.blocks.get(request_id, set()))

    def preemptions(self) -> int:
        return self.preempted


def make_dispatcher(**cfg) -> tuple[Dispatcher, FakeEngine]:
    dispatcher = Dispatcher(SchedulerConfig(**cfg))
    engine = FakeEngine()
    dispatcher.attach_engine(engine, engine.block_size, engine.max_model_len)
    return dispatcher, engine


def make_request(request_id: str, user_id: str, **kwargs) -> ScheduledRequest:
    return ScheduledRequest(
        request_id=request_id,
        user_id=user_id,
        prompt="hello",
        prompt_tokens=kwargs.pop("prompt_tokens", 100),
        max_tokens=kwargs.pop("max_tokens", 50),
        **kwargs,
    )


def test_submit_admits_and_reserves():
    dispatcher, engine = make_dispatcher()
    request = make_request("r1", "u1")

    decision = dispatcher.submit(request)

    assert decision.decision == ADMIT
    assert "r1" in engine.submitted
    assert dispatcher.ledger.reserved_blocks == decision.reserved_blocks
    assert dispatcher.inflight_count() == 1


def test_a_duplicate_request_id_is_rejected():
    """A duplicate submit must not start a second engine request."""
    dispatcher, engine = make_dispatcher()
    dispatcher.submit(make_request("r1", "u1"))

    decision = dispatcher.submit(make_request("r1", "u1"))

    assert decision.decision == REJECT
    assert decision.reason == "duplicate"
    assert len(engine.submitted) == 1


def test_over_capacity_is_held_then_dispatched():
    dispatcher, engine = make_dispatcher(max_in_flight=1)

    dispatcher.submit(make_request("r1", "u1"))
    decision = dispatcher.submit(make_request("r2", "u2"))

    assert decision.decision == HOLD
    assert "r2" not in engine.submitted
    assert dispatcher.held_count() == 1

    # r1 finishes: its slot is recycled and r2 goes out.
    engine.finish("r1")

    assert "r2" in engine.submitted
    assert dispatcher.held_count() == 0
    assert dispatcher.inflight_count() == 1


def test_the_reservation_is_released_when_a_request_finishes():
    dispatcher, engine = make_dispatcher()
    dispatcher.submit(make_request("r1", "u1"))
    reserved = dispatcher.ledger.reserved_blocks
    assert reserved > 0

    engine.finish("r1")

    assert dispatcher.ledger.reserved_blocks == 0


def test_round_robin_across_users():
    """One user with a deep queue must not take every slot."""
    dispatcher, engine = make_dispatcher(max_in_flight=1)

    dispatcher.submit(make_request("r0", "u0"))  # takes the only slot
    dispatcher.submit(make_request("a1", "ua"))
    dispatcher.submit(make_request("a2", "ua"))
    dispatcher.submit(make_request("b1", "ub"))

    engine.finish("r0")

    # One slot freed. It goes to the user at the head of the round robin.
    assert "a1" in engine.submitted
    assert "a2" not in engine.submitted
    assert "b1" not in engine.submitted

    # The next slot goes to the other user, not to ua again.
    engine.finish("a1")

    assert "b1" in engine.submitted
    assert "a2" not in engine.submitted


def test_a_user_keeps_its_own_fifo_order():
    dispatcher, engine = make_dispatcher(max_in_flight=1)
    dispatcher.submit(make_request("r0", "u0"))
    dispatcher.submit(make_request("a1", "ua"))
    dispatcher.submit(make_request("a2", "ua"))

    engine.finish("r0")
    assert "a1" in engine.submitted

    engine.finish("a1")
    assert "a2" in engine.submitted


def test_abort_releases_the_reservation_and_the_slot():
    dispatcher, engine = make_dispatcher()
    dispatcher.submit(make_request("r1", "u1"))

    released = dispatcher.abort("r1")

    assert released > 0
    assert dispatcher.ledger.reserved_blocks == 0
    assert dispatcher.inflight_count() == 0
    assert "r1" not in engine.submitted


def test_abort_of_a_held_request_takes_it_out_of_the_queue():
    dispatcher, engine = make_dispatcher(max_in_flight=1)
    dispatcher.submit(make_request("r1", "u1"))
    dispatcher.submit(make_request("r2", "u2"))

    dispatcher.abort("r2")

    assert dispatcher.held_count() == 0
    engine.finish("r1")
    assert "r2" not in engine.submitted


def test_the_engine_is_not_ready_until_it_is_attached():
    dispatcher = Dispatcher(SchedulerConfig())
    decision = dispatcher.submit(make_request("r1", "u1"))
    assert decision.decision == REJECT
    assert decision.reason == "engine_not_ready"


def test_reconcile_reports_no_drift_when_the_ledger_accounts_for_everything():
    dispatcher, engine = make_dispatcher()
    dispatcher.submit(make_request("r1", "u1"))

    # The engine holds two blocks for r1 and has 998 free.
    engine.hold("r1", {1, 2})

    assert dispatcher.reconcile() == 0
    assert dispatcher.ledger.drift == 0


def test_reconcile_reports_drift_when_the_engine_holds_unclaimed_blocks():
    dispatcher, engine = make_dispatcher()
    dispatcher.submit(make_request("r1", "u1"))

    # The engine holds two blocks under a request the ledger does not track.
    engine.hold("stray", {1, 2})

    assert dispatcher.reconcile() == 2


# --- priority ----------------------------------------------------------------


def test_the_request_priority_reaches_the_engine():
    """The priority is the scheduler's only say in who survives."""
    dispatcher, engine = make_dispatcher()
    dispatcher.submit(make_request("r1", "u1", priority=3))

    assert engine.submitted["r1"]["priority"] == 3


def test_a_preemption_sheds_the_least_important_request():
    """The engine ran out of blocks, so somebody has to go.

    It is the request that matters least: the client of the important one
    gets its answer, and the client of the other one gets a retryable
    failure instead of a silent truncation.
    """
    dispatcher, engine = make_dispatcher()
    dispatcher.submit(make_request("keep", "u1", priority=1))
    dispatcher.submit(make_request("drop", "u2", priority=99))

    engine.preempted = 1
    dispatcher.reconcile()

    assert "drop" not in engine.submitted
    assert "keep" in engine.submitted
    assert dispatcher.get("drop").finish_reason == "capacity_breach"
    assert dispatcher.get("drop").state == "aborted"


def test_a_preemption_is_counted_so_the_breach_is_not_silent():
    dispatcher, engine = make_dispatcher()
    dispatcher.submit(make_request("r1", "u1"))

    before = REGISTRY.get_sample_value(N.SCHED_CAPACITY_BREACH_TOTAL) or 0.0
    engine.preempted = 1
    dispatcher.reconcile()

    assert (REGISTRY.get_sample_value(N.SCHED_CAPACITY_BREACH_TOTAL) or 0.0) == before + 1


# --- the in-flight ceiling ---------------------------------------------------


def test_the_engines_limit_is_the_ceiling():
    """The engine knows how many sequences it can run; the scheduler does not."""
    engine = FakeEngine()
    engine.cfg = SimpleNamespace(max_num_seqs=8)
    dispatcher = Dispatcher(SchedulerConfig())
    dispatcher.attach_engine(engine, engine.block_size, engine.max_model_len)

    assert dispatcher.max_in_flight == 8


def test_a_configured_ceiling_can_only_lower_the_engines():
    engine = FakeEngine()
    engine.cfg = SimpleNamespace(max_num_seqs=8)

    lower = Dispatcher(SchedulerConfig(max_in_flight=4))
    lower.attach_engine(engine, engine.block_size, engine.max_model_len)
    assert lower.max_in_flight == 4

    higher = Dispatcher(SchedulerConfig(max_in_flight=64))
    higher.attach_engine(engine, engine.block_size, engine.max_model_len)
    assert higher.max_in_flight == 8


def test_the_ceiling_falls_back_when_the_engine_does_not_say():
    assert resolve_max_in_flight(None, object()) == DEFAULT_MAX_IN_FLIGHT
    assert resolve_max_in_flight(7, object()) == 7
