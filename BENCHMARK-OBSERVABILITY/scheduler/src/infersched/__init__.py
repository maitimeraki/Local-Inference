"""The scheduler tier.

Sits between the gateway and the engine. Decides whether a request fits in
the KV cache, holds the ones that do not, and dispatches them as capacity
frees up.

The engine is the authority on what exists; this tier is the authority on
what we have promised. Keeping those two apart — and measuring the gap — is
the whole job.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
