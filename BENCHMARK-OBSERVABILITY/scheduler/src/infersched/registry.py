"""Per-user state: who is running, who is waiting.

Deliberately knows nothing about KV blocks — that is the ledger's job. Two
tiers keeping their own copy of the same fact is how they drift apart.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class UserState:
    inflight: set[str] = field(default_factory=set)
    queued: deque[str] = field(default_factory=deque)


class UserRegistry:
    """Per-user in-flight sets and FIFO wait queues."""

    def __init__(self) -> None:
        self._users: dict[str, UserState] = {}

    def state(self, user_id: str) -> UserState:
        st = self._users.get(user_id)
        if st is None:
            st = UserState()
            self._users[user_id] = st
        return st

    def inflight_count(self, user_id: str) -> int:
        return len(self.state(user_id).inflight)

    def queued_count(self, user_id: str) -> int:
        return len(self.state(user_id).queued)

    def total_inflight(self) -> int:
        return sum(len(s.inflight) for s in self._users.values())

    def total_queued(self) -> int:
        return sum(len(s.queued) for s in self._users.values())

    def enqueue(self, user_id: str, request_id: str) -> None:
        self.state(user_id).queued.append(request_id)

    def dequeue(self, user_id: str) -> str | None:
        q = self.state(user_id).queued
        return q.popleft() if q else None

    def requeue_front(self, user_id: str, request_id: str) -> None:
        """Put a request back at the head of its user's queue."""
        self.state(user_id).queued.appendleft(request_id)

    def admit(self, user_id: str, request_id: str) -> None:
        self.state(user_id).inflight.add(request_id)

    def finish(self, user_id: str, request_id: str) -> None:
        self.state(user_id).inflight.discard(request_id)

    def cancel_queued(self, user_id: str, request_id: str) -> bool:
        q = self.state(user_id).queued
        try:
            q.remove(request_id)
        except ValueError:
            return False
        return True

    def snapshot(self) -> dict[str, dict[str, int]]:
        return {
            user_id: {"inflight": len(s.inflight), "queued": len(s.queued)}
            for user_id, s in self._users.items()
        }
