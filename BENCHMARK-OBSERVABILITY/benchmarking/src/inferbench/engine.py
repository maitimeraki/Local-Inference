"""Build vLLM engines from config — as a library, in this process.

This is the module that replaces `docker run vllm/vllm-openai`. The engine
lives in our process, so its scheduler and KV cache are objects we can read
rather than a container we can only POST to.

One environment variable is load-bearing and is set here, before vLLM is
imported, because vLLM reads it at import time:

  VLLM_ENABLE_V1_MULTIPROCESSING=0   V1 otherwise runs EngineCore in a child
                                     process, which hides exactly the internals
                                     this project exists to see.

There is no V0 engine to fall back to: vLLM deleted it in 0.11.0, and
`VLLM_USE_V1` was removed from `envs.py` in 0.11.2.
"""

from __future__ import annotations

import os
from typing import Any

from inferbench.config import EngineConfig


class VllmNotInstalled(RuntimeError):
    """Raised with an actionable message rather than a bare ImportError."""

    def __init__(self) -> None:
        super().__init__(
            "vLLM is not importable in this interpreter. On Linux/WSL2 run:\n"
            "    uv pip install --python /home/anupam/.venv -e 'benchmarking[engine]'\n"
            "vLLM has no Windows build; use the WSL2 environment (see scripts/bootstrap_wsl.sh)."
        )


def configure_env(*, force: bool = False) -> None:
    """Set the environment vLLM reads at import. Idempotent.

    `force=False` (the default) leaves an explicitly-set value alone, so an
    operator can override from the shell without editing code. `force=True`
    is for the step-through inspector, where an override would leave the
    scheduler in a child process and the mode with nothing to inspect.
    """
    env = {
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "VLLM_LOGGING_LEVEL": os.environ.get("VLLM_LOGGING_LEVEL", "INFO"),
    }
    for key, value in env.items():
        if force or key not in os.environ:
            os.environ[key] = value


def require_vllm() -> Any:
    """Import vLLM or explain how to install it."""
    try:
        import vllm
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise VllmNotInstalled() from exc
    return vllm


def vllm_version() -> str | None:
    try:
        return require_vllm().__version__
    except Exception:
        return None


def build_llm(cfg: EngineConfig) -> Any:
    """Offline engine (`vllm.LLM`) — batch generate, deepest stats access."""
    configure_env()
    vllm = require_vllm()
    return vllm.LLM(**cfg.engine_kwargs())


def build_async_engine(cfg: EngineConfig) -> Any:
    """Serving engine (`AsyncLLMEngine`) for the OpenAI-compatible server."""
    configure_env()
    vllm = require_vllm()
    args = vllm.AsyncEngineArgs(**cfg.engine_kwargs())
    return vllm.AsyncLLMEngine.from_engine_args(args)


def build_sampling_params(vllm: Any, max_tokens: int, sampling: Any) -> Any:
    """SamplingParams pinned to an exact output length.

    `min_tokens == max_tokens` with `ignore_eos` is what makes two runs
    comparable: without it the engine stops at its own convenience and the
    difference between runs is answer length, not engine behaviour.
    """
    return vllm.SamplingParams(
        max_tokens=max_tokens,
        min_tokens=max_tokens,
        ignore_eos=sampling.ignore_eos,
        temperature=sampling.temperature,
        top_p=sampling.top_p,
        seed=sampling.seed,
    )
