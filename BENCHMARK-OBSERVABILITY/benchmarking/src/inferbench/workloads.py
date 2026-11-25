"""Turn workload profiles into concrete prompts of a known token length.

Prompts are built to an approximate token target, but the *reported* input
length always comes from the server's `usage.prompt_tokens` — that is the only
number that is true. The estimate here exists to make the request, not to be
measured.

`prefix_mode="shared"` is the switch that makes prefix caching observable: all
requests in the run carry an identical leading block, so a cache-enabled server
skips prefill for it and a disabled one does not.

`PromptFactory(salt=...)` keeps every *other* run from replaying the same
prompts, because a server that already has the prefix cached is not being
measured.
"""

from __future__ import annotations

import random
import zlib
from dataclasses import dataclass
from typing import Protocol


class Tokenizer(Protocol):
    """The sliver of the HF tokenizer interface this module uses."""

    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...


# Vocabulary for synthetic filler. Deliberately plain: the point is token
# *count*, not content. Realistic distributions come later, from app traces.
_WORDS = (
    "system analysis request response context model inference latency throughput "
    "batch schedule memory cache token prompt generation sequence attention layer "
    "vector matrix gradient tensor pipeline runtime kernel stream socket buffer "
    "client server protocol header payload request payload chunk prefix suffix"
).split()


def _filler_text(seed: int, words: int) -> str:
    rng = random.Random(seed)
    return " ".join(rng.choice(_WORDS) for _ in range(words))


def text_for_tokens(tokenizer: Tokenizer, n_tokens: int, seed: int = 0) -> str:
    """Text that tokenizes to approximately `n_tokens`.

    Encodes a generous filler block, slices to the target, and decodes. A short
    repair loop handles the case where the decode/re-encode round trip lands a
    token or two off, which it does at BPE boundaries.
    """
    if n_tokens <= 0:
        return ""
    # ~2 words per token is a safe overshoot for the word list above.
    filler = _filler_text(seed, n_tokens * 2 + 32)
    ids = tokenizer.encode(filler)
    if len(ids) >= n_tokens:
        text = tokenizer.decode(ids[:n_tokens])
    else:
        text = filler

    for _ in range(4):
        actual = len(tokenizer.encode(text))
        if actual == n_tokens:
            break
        if actual > n_tokens:
            text = tokenizer.decode(tokenizer.encode(text)[:n_tokens])
        else:
            text += " " + text_for_tokens(tokenizer, n_tokens - actual, seed + 1)
    return text


@dataclass
class Prompt:
    index: int
    text: str
    approx_input_tokens: int
    shared_prefix_tokens: int


class PromptFactory:
    """Builds the prompts for one workload profile.

    The shared prefix (when enabled) is generated once and reused verbatim, so
    the bytes are identical across requests and the server's prefix cache can
    actually hit.

    `salt` makes the *unique* part of every prompt different. A server that
    has already seen a prompt answers it from its prefix cache, so replaying
    the same prompts measures that cache rather than the workload — every
    measured run gets its own salt. The shared prefix is never salted: its
    cache hit is the experiment.
    """

    def __init__(self, tokenizer: Tokenizer, profile, salt: int = 0) -> None:
        self.tokenizer = tokenizer
        self.profile = profile
        self.salt = salt
        self._shared = ""
        if profile.prefix_mode == "shared" and profile.shared_prefix_tokens > 0:
            self._shared = text_for_tokens(
                tokenizer,
                profile.shared_prefix_tokens,
                # A fixed seed, not `hash()`: Python randomises string hashes
                # per process, so a shared prefix that differs between runs is
                # a different experiment every time.
                seed=zlib.crc32(profile.name.encode("utf-8")) & 0xFFFF,
            )

    def build(self, index: int) -> Prompt:
        unique_tokens = self.profile.input_tokens - (
            self.profile.shared_prefix_tokens if self._shared else 0
        )
        suffix = text_for_tokens(
            self.tokenizer, max(unique_tokens, 1), seed=self._seed(index)
        )
        text = f"{self._shared}{suffix}" if self._shared else suffix
        return Prompt(
            index=index,
            text=text,
            approx_input_tokens=self.profile.input_tokens,
            shared_prefix_tokens=self.profile.shared_prefix_tokens if self._shared else 0,
        )

    def build_many(self, count: int) -> list[Prompt]:
        return [self.build(i) for i in range(count)]

    def _seed(self, index: int) -> int:
        # Index stays well under a million, so no two (index, salt) pairs
        # collide, and salt=0 keeps the unsalted seed this always used.
        return index + self.salt * 1_000_000
