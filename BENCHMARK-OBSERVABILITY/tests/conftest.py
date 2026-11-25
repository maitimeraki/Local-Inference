"""Shared fixtures.

The whole suite is GPU-free and vLLM-free by design: the parts of this platform
that can be wrong in a way nobody notices — the latency math, the workload
token counts, the cache key, the experiment matrix — are exactly the parts that
do not need a GPU to test.
"""

from __future__ import annotations

import pytest


class CharTokenizer:
    """A tokenizer where one token is one character.

    Round-trips exactly, which makes `text_for_tokens` assertions exact rather
    than approximate. A real BPE tokenizer does not round-trip cleanly at
    boundaries, which is exactly why the platform reports the server's
    `usage.prompt_tokens` rather than its own estimate.
    """

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)


@pytest.fixture
def tokenizer() -> CharTokenizer:
    return CharTokenizer()


@pytest.fixture
def no_config_dir(monkeypatch, tmp_path):
    """Point config loading at an empty directory."""
    monkeypatch.setenv("INFERBENCH_CONFIG_DIR", str(tmp_path))
    return tmp_path
