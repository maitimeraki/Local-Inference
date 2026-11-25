"""infergw — the request router in front of the vLLM backends.

Exists for three reasons the direct-to-engine path cannot provide:

1. Arrival rate, queueing delay and admission decisions happen here. Nothing
   downstream can measure them.
2. It can split traffic across two engine configurations *simultaneously*, so
   an A/B comparison sees one identical arrival pattern instead of two runs
   fighting thermal drift.
3. It is where an exact-match cache turns a repeated prompt into a metadata
   lookup, which is the cheapest latency win available.
"""

__version__ = "0.1.0"
