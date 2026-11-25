"""Backend selection.

Two strategies, both needed:

  least_outstanding  sends to the backend with the fewest in-flight requests.
                     This is what keeps two differently-configured engines
                     comparable when you are *not* trying to split traffic.
  weighted           sends a fixed proportion to each backend. This is the A/B
                     mode: `vllm-baseline` gets 0.5, `vllm-optimized` gets 0.5,
                     and both see the same arrival pattern at the same time.
"""

from __future__ import annotations

import itertools

from infergw.config import Backend, GatewayConfig


class Router:
    def __init__(self, config: GatewayConfig) -> None:
        self.config = config
        self._outstanding: dict[str, int] = {b.name: 0 for b in config.backends}
        self._cycle_names = self._weighted_names(config)
        self._cycle = itertools.cycle(self._cycle_names)

    @staticmethod
    def _weighted_names(config: GatewayConfig) -> list[str]:
        """Expand weights into a repeated name list.

        Integer-proportioned and served round-robin rather than at random: a
        benchmark whose traffic split wobbles run to run cannot be compared to
        the run before it.
        """
        names: list[str] = []
        for b in config.backends:
            if not b.healthy:
                continue
            names.extend([b.name] * max(1, round(b.weight * 10)))
        return names or [b.name for b in config.backends if b.healthy]

    def healthy(self) -> list[Backend]:
        return [b for b in self.config.backends if b.healthy]

    def pick(self) -> Backend:
        """Choose a backend, or raise if none is healthy."""
        candidates = self.healthy()
        if not candidates:
            raise NoHealthyBackend("no healthy backends configured")

        if self.config.strategy == "weighted":
            names = self._weighted_names(self.config)
            if names != self._cycle_names:
                # Health changed; rebuild so the split still matches config.
                self._cycle_names = names
                self._cycle = itertools.cycle(names)
            chosen = next(self._cycle)
            return next(b for b in candidates if b.name == chosen)

        # least_outstanding — ties broken by declaration order, so a run is
        # reproducible rather than depending on dict iteration.
        return min(candidates, key=lambda b: self._outstanding.get(b.name, 0))

    def acquire(self, name: str) -> None:
        self._outstanding[name] = self._outstanding.get(name, 0) + 1

    def release(self, name: str) -> None:
        self._outstanding[name] = max(0, self._outstanding.get(name, 0) - 1)

    def outstanding(self, name: str) -> int:
        return self._outstanding.get(name, 0)

    def mark_unhealthy(self, name: str) -> None:
        """Take a backend out of rotation after repeated failures."""
        for b in self.config.backends:
            if b.name == name:
                b.healthy = False


class NoHealthyBackend(RuntimeError):
    pass
