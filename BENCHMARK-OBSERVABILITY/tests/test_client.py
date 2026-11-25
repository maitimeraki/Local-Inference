"""The benchmark's concurrency model.

The scheduler caps one user's share of the GPU. If the whole batch arrives as
one user, a concurrency sweep measures that cap instead of concurrency: every
point above the cap looks the same, and the curve is flat for a reason that
has nothing to do with the engine.
"""

from __future__ import annotations

import asyncio

from inferbench.client import run_concurrent


class _StubClient:
    """Records who each request was sent as."""

    user_id = "bench"

    def __init__(self) -> None:
        self.users: list[str] = []
        self.live = 0
        self.max_live = 0

    async def complete(self, prompt, max_tokens, request_id, input_tokens=None, user_id=None):
        self.users.append(user_id)
        self.live += 1
        self.max_live = max(self.max_live, self.live)
        await asyncio.sleep(0)
        self.live -= 1
        return request_id


def test_a_batch_runs_as_independent_clients_not_one_user():
    client = _StubClient()
    asyncio.run(run_concurrent(client, [("p", 8, None)] * 10, concurrency=4))

    # Four in flight at once, no more.
    assert client.max_live == 4
    # Four distinct users, reused across the batch — a user per request would
    # sidestep the per-user cap entirely and measure nothing about it.
    assert len(set(client.users)) == 4
    assert set(client.users) == {"bench-u0", "bench-u1", "bench-u2", "bench-u3"}


def test_one_slot_means_one_user():
    client = _StubClient()
    asyncio.run(run_concurrent(client, [("p", 8, None)] * 3, concurrency=1))

    assert client.max_live == 1
    assert set(client.users) == {"bench-u0"}
