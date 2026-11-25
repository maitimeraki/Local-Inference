"""Tier-1 per-user state: who is in flight, and which idempotency keys are live.

Deliberately knows nothing about KV blocks. That is tier 2's ledger's job, and
a second copy of the block accounting here is exactly how the two drift apart.

Two things live here:

  * the per-user in-flight count, so one client cannot take every slot;
  * the idempotency keys, so a retried submit does not start a second engine
    request.

The key is scoped to the user: two users may pick the same key without
colliding.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class UserState:
    inflight: set[str] = field(default_factory=set)
    keys: dict[str, str] = field(default_factory=dict)  # idempotency key -> request id
    key_of: dict[str, str] = field(default_factory=dict)  # request id -> idempotency key
    key_ts: dict[str, float] = field(default_factory=dict)  # idempotency key -> claimed at


class UserRegistry:
    """Per-user in-flight accounting and idempotency keys.

    Thread-safe: uvicorn may run handlers on a threadpool.
    """

    def __init__(self, key_ttl_s: float = 900.0) -> None:
        self.key_ttl_s = key_ttl_s
        self._users: dict[str, UserState] = {}
        self._lock = threading.Lock()

    def _state(self, user_id: str) -> UserState:
        st = self._users.get(user_id)
        if st is None:
            st = UserState()
            self._users[user_id] = st
        return st

    def in_flight(self, user_id: str) -> int:
        with self._lock:
            st = self._users.get(user_id)
            return 0 if st is None else len(st.inflight)

    def lookup(self, user_id: str, key: str | None) -> str | None:
        """The request already running under this idempotency key, if any."""
        if not key:
            return None
        with self._lock:
            st = self._users.get(user_id)
            if st is None:
                return None
            request_id = st.keys.get(key)
            if request_id is None:
                return None
            if time.time() - st.key_ts.get(key, 0.0) > self.key_ttl_s:
                # A key older than its TTL counts as absent. A leaked entry
                # must not lock a user out of their own endpoint forever.
                self._forget(st, request_id)
                return None
            return request_id

    def acquire(self, user_id: str, request_id: str, key: str | None = None) -> None:
        """Mark a request in flight, and claim its idempotency key."""
        with self._lock:
            st = self._state(user_id)
            st.inflight.add(request_id)
            if key:
                st.keys[key] = request_id
                st.key_of[request_id] = key
                st.key_ts[key] = time.time()

    def release(self, user_id: str, request_id: str) -> None:
        """Give the slot and the key back."""
        with self._lock:
            st = self._users.get(user_id)
            if st is None:
                return
            st.inflight.discard(request_id)
            self._forget(st, request_id)

    @staticmethod
    def _forget(st: UserState, request_id: str) -> None:
        key = st.key_of.pop(request_id, None)
        if key is not None:
            st.keys.pop(key, None)
            st.key_ts.pop(key, None)
