"""Client for the scheduler tier's internal API.

The gateway is the only caller of this protocol. The scheduler is not a public
service — it has no authentication and `/internal/v1/state` exposes an unbounded
per-user map — so it belongs on loopback or a private network and nothing else
should point at it.

The contract is the one in PRODUCTION-DESIGN.md §12.2.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class SubmitResult:
    """What the scheduler decided about a request."""

    decision: str  # admit | hold | reject
    reason: str
    reserved_blocks: int
    hold_seq: int
    # What the request asked for, before the ceiling was applied. Zero when
    # the scheduler could not measure it (an older scheduler, or a reject
    # before the ledger had a block size).
    demand: int = 0


class SchedulerUnavailable(RuntimeError):
    """The scheduler could not be reached, or answered something unusable."""


class SchedulerClient:
    def __init__(self, base_url: str, timeout_s: float = 300.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    async def submit(self, client: httpx.AsyncClient, **fields: object) -> SubmitResult:
        """Ask the scheduler what to do with a request."""
        try:
            resp = await client.post(
                f"{self.base_url}/internal/v1/submit", json=fields, timeout=self.timeout_s
            )
        except httpx.HTTPError as exc:
            raise SchedulerUnavailable(f"scheduler unreachable: {exc}") from exc
        if resp.status_code >= 500:
            raise SchedulerUnavailable(f"scheduler returned {resp.status_code}")
        try:
            data = resp.json()
        except ValueError as exc:
            raise SchedulerUnavailable("scheduler returned a non-JSON body") from exc
        return SubmitResult(
            decision=str(data.get("decision") or "reject"),
            reason=str(data.get("reason") or ""),
            reserved_blocks=int(data.get("reserved_blocks") or 0),
            hold_seq=int(data.get("hold_seq") or 0),
            demand=int(data.get("demand") or 0),
        )

    async def abort(
        self, client: httpx.AsyncClient, request_id: str, reason: str = "client_abort"
    ) -> int:
        """Drop a request and give its reservation back. Blocks released."""
        try:
            resp = await client.post(
                f"{self.base_url}/internal/v1/abort",
                json={"request_id": request_id, "reason": reason},
                timeout=self.timeout_s,
            )
        except httpx.HTTPError:
            return 0
        if resp.status_code >= 400:
            return 0
        try:
            return int(resp.json().get("released_blocks") or 0)
        except ValueError:
            return 0

    async def stream(self, client: httpx.AsyncClient, request_id: str) -> AsyncIterator[dict]:
        """The scheduler's event frames for one request, as they happen.

        The first frame says what was decided; the middle frames carry deltas;
        the last frame carries the finish. A request that is held waits here
        until a slot frees — that is the whole point of the hold queue.
        """
        try:
            async with client.stream(
                "GET", f"{self.base_url}/internal/v1/stream/{request_id}", timeout=self.timeout_s
            ) as resp:
                if resp.status_code >= 400:
                    raise SchedulerUnavailable(f"scheduler stream returned {resp.status_code}")
                async for frame in _frames(resp.aiter_lines()):
                    yield frame
        except httpx.HTTPError as exc:
            raise SchedulerUnavailable(f"scheduler stream failed: {exc}") from exc

    async def state(self, client: httpx.AsyncClient) -> dict:
        """The scheduler's ledger and registry, for readiness."""
        try:
            resp = await client.get(f"{self.base_url}/internal/v1/state", timeout=self.timeout_s)
        except httpx.HTTPError as exc:
            raise SchedulerUnavailable(f"scheduler unreachable: {exc}") from exc
        if resp.status_code >= 400:
            raise SchedulerUnavailable(f"scheduler returned {resp.status_code}")
        try:
            return resp.json()
        except ValueError as exc:
            raise SchedulerUnavailable("scheduler returned a non-JSON body") from exc


async def _frames(lines: AsyncIterator[str]) -> AsyncIterator[dict]:
    """SSE `data:` frames, as dicts. Anything unparseable is skipped."""
    async for line in lines:
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            frame = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(frame, dict):
            yield frame
