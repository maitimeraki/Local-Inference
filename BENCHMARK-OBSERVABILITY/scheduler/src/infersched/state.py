"""Reconciliation: read the engine's truth into the ledger.

Runs on the engine's own thread after every step, not on a timer. A timer
samples the two views at different instants and manufactures drift that is
not there.
"""

from __future__ import annotations

from typing import Any

from infersched.ledger import ReservationLedger


def reconcile(engine: Any, ledger: ReservationLedger, request_ids: list[str]) -> int | None:
    """Refresh the ledger from the engine. Returns the drift, or None."""
    free = engine.free_blocks()
    total = engine.total_blocks()
    ledger.observe(free_blocks=free, total_blocks=total)
    if free is None or total is None:
        return None
    for request_id in request_ids:
        block_ids = engine.request_block_ids(request_id)
        if block_ids is not None:
            ledger.observe_blocks(request_id, block_ids)
    return ledger.drift
