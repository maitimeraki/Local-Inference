"""Exact-match response cache.

Keyed on the *decision-relevant* part of the request — prompt, length caps,
sampling parameters — so two requests that must produce the same output share
an entry and two that must not, do not. Getting this key wrong is worse than
having no cache: it returns a confident wrong answer.

`ponytail: in-process OrderedDict, per replica. A second gateway replica has
its own cache, so the hit rate halves with HPA. Move to Redis when the gateway
scales past one — that is the only thing here that needs it.`
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

# Fields that change the produced tokens. Anything not listed is ignored so a
# cosmetic extra field cannot fragment the cache.
_KEY_FIELDS = (
    "model", "prompt", "max_tokens", "min_tokens",
    "temperature", "top_p", "seed", "ignore_eos",
)


@dataclass
class CachedCompletion:
    """A completed generation, enough to replay either as JSON or as SSE."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    observed_duration_s: float  # what the miss cost, for the "saved" metric
    expires_at: float


def cache_key(body: dict) -> str:
    relevant = {k: body.get(k) for k in _KEY_FIELDS}
    blob = json.dumps(relevant, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class ExactCache:
    """LRU with TTL. Thread-safe: uvicorn may run handlers on a threadpool."""

    def __init__(self, max_entries: int = 2048, ttl_s: float = 600.0) -> None:
        self.max_entries = max_entries
        self.ttl_s = ttl_s
        self._entries: OrderedDict[str, CachedCompletion] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expirations = 0

    def get(self, key: str) -> CachedCompletion | None:
        now = time.time()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            if entry.expires_at < now:
                del self._entries[key]
                self.expirations += 1
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return entry

    def put(self, key: str, entry: CachedCompletion) -> None:
        with self._lock:
            self._entries[key] = entry
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
                self.evictions += 1

    def stats(self) -> dict:
        with self._lock:
            total = self.hits + self.misses
            return {
                "entries": len(self._entries),
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "expirations": self.expirations,
                "hit_rate": (self.hits / total) if total else 0.0,
            }

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


def text_from_sse(chunks: list[bytes]) -> tuple[str, int, str]:
    """Concatenate the `text` fields of an SSE body.

    Returns (text, completion_tokens, finish_reason). Parsed locally rather
    than importing the harness's parser: the gateway must not depend on the
    benchmark package.
    """
    parts: list[str] = []
    tokens = 0
    finish = "stop"
    for raw in chunks:
        for line in raw.split(b"\n"):
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == b"[DONE]":
                continue
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            if not choices:
                continue
            text = choices[0].get("text") or ""
            if text:
                parts.append(text)
                tokens += 1
            if choices[0].get("finish_reason"):
                finish = choices[0]["finish_reason"]
    return "".join(parts), tokens, finish
