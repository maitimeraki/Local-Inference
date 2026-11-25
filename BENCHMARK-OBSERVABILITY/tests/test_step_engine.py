"""The engine's request id is not the one the caller gave it.

vLLM replaces the caller's request id with an internal one (`q1` becomes
`q1-a2abe44e`) and keys its KV-block pool by that internal id. Everything
else — `RequestOutput.request_id`, `abort_request()` — speaks the caller's
id. The block pool is the one place that only understands the internal one,
so the translation has to happen before the ledger looks a request up.
"""

from __future__ import annotations

from typing import Any

from inferbench.config import EngineConfig
from inferbench.step_engine import StepEngine


class StubHost:
    """Records what the engine host was asked about."""

    def __init__(self) -> None:
        self.asked: list[str] = []

    def request_blocks(self, request_id: str) -> int | None:
        self.asked.append(request_id)
        return 3

    def request_block_ids(self, request_id: str) -> set[int] | None:
        self.asked.append(request_id)
        return {1, 2, 3}


def _engine() -> StepEngine:
    cfg = EngineConfig(name="stub", model="stub-model")
    engine = StepEngine(cfg)
    engine.host = StubHost()  # type: ignore[assignment]
    return engine


def test_the_block_pool_is_asked_about_the_engines_own_request_id() -> None:
    engine = _engine()
    handle = engine.submit("q1", "hello", None)
    handle.engine_id = "q1-a2abe44e"

    assert engine.request_block_ids("q1") == {1, 2, 3}
    assert engine.host.asked == ["q1-a2abe44e"]  # type: ignore[attr-defined]


def test_a_request_the_engine_has_not_named_yet_is_asked_about_by_its_own_id() -> None:
    engine = _engine()
    engine.submit("q2", "hello", None)

    engine.request_blocks("q2")
    assert engine.host.asked == ["q2"]  # type: ignore[attr-defined]


def test_an_unknown_request_id_is_passed_through() -> None:
    engine = _engine()

    engine.request_block_ids("never-submitted")
    assert engine.host.asked == ["never-submitted"]  # type: ignore[attr-defined]


def test_the_translation_is_the_only_thing_between_the_two_ids() -> None:
    """`abort` and output routing speak the caller's id; only blocks do not."""
    engine = _engine()
    handle = engine.submit("q3", "hello", None)
    handle.engine_id = "q3-deadbeef"

    seen: list[Any] = []
    handle.on_event = seen.append  # type: ignore[assignment]

    # The handle is keyed by the caller's id, which is what the engine's
    # `RequestOutput.request_id` carries.
    assert "q3" in engine._handles
    assert engine._handles["q3"] is handle


class StubVLLMEngine:
    """Stands in for vLLM's `LLMEngine`."""

    def __init__(self) -> None:
        self.added: list[tuple[str, str, Any, int]] = []

    def add_request(self, request_id, prompt, params, priority: int = 0) -> str:
        self.added.append((request_id, prompt, params, priority))
        return f"{request_id}-engine-id"


def test_the_priority_reaches_the_engine() -> None:
    """The scheduler's priority is only useful if the engine gets it.

    vLLM serves the smallest priority number first and preempts the largest
    one last, so a priority that stops at the scheduler changes nothing.
    """
    engine = _engine()
    engine._engine = StubVLLMEngine()  # type: ignore[assignment]

    engine.submit("q1", "hello", None, priority=7)
    engine._drain()

    assert engine._engine.added == [("q1", "hello", None, 7)]  # type: ignore[attr-defined]
