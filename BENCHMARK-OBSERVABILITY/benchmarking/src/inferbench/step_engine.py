"""The V1 engine in this process, driven one step at a time.

`AsyncLLMEngine` cannot be used by the scheduler tier. In vLLM 0.31.x
`EngineCoreClient.make_client` raises `NotImplementedError` for an async
engine core without multiprocessing, and `AsyncLLM` calls
`make_async_mp_client(...)` unconditionally — so the async path *always*
runs the engine core in a child process, where the block pool is invisible.
`collective_rpc` is not a way around it: it reaches the executor workers,
not the scheduler.

The sync `LLMEngine` with `VLLM_ENABLE_V1_MULTIPROCESSING=0` uses an
`InprocClient` whose `engine_core` owns the scheduler, in this process.
That is the only configuration in which `block_pool.get_num_free_blocks()`
and `req_to_blocks` are reachable, and exact block accounting is the whole
premise of the reservation ledger.

The cost is that somebody has to call `step()`. This module runs the loop on
a background thread. V1's `EngineCore.step()` returns `({}, False)` when
there is nothing to do rather than spinning, so the idle loop is cheap.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from inferbench.config import EngineConfig
from inferbench.engine_host import EngineHost

__all__ = ["RequestHandle", "StepEngine", "StreamEvent"]

log = logging.getLogger(__name__)


@dataclass
class StreamEvent:
    """One thing that happened to a request.

    `delta` is the text produced since the previous event — the engine's
    cumulative output sliced, so a consumer can append it blindly.
    """

    delta: str = ""
    finished: bool = False
    finish_reason: str = "stop"
    prompt_tokens: int | None = None
    completion_tokens: int = 0
    error: str | None = None


class RequestHandle:
    """The caller's side of one in-flight request.

    Events go to a callback rather than onto a queue: the consumer is an
    asyncio loop in another thread, and a queue would mean a thread-pool hop
    per token for nothing.
    """

    def __init__(
        self, request_id: str, on_event: Callable[[StreamEvent], None] | None = None
    ) -> None:
        self.request_id = request_id
        self.on_event = on_event
        self.text = ""
        self.prompt_tokens: int | None = None
        self.completion_tokens = 0
        # The id the engine gave the request, once it has one.
        self.engine_id: str | None = None

    def push(self, event: StreamEvent) -> None:
        if self.on_event is None:
            return
        try:
            self.on_event(event)
        except Exception:
            # A consumer that raises must not take the engine's step loop with
            # it; the engine keeps running whether or not anyone is listening.
            log.exception("event callback failed for request %s", self.request_id)


class StepEngine:
    """The engine, plus the thread that drives it.

    Only the engine thread touches the engine. Callers submit work and read
    events, so there is no lock over the model forward pass.
    """

    def __init__(
        self,
        cfg: EngineConfig,
        *,
        on_step: Callable[[StepEngine], None] | None = None,
    ) -> None:
        self.cfg = cfg
        self.host: EngineHost | None = None
        self._on_step = on_step
        self._vllm: Any = None
        self._tokenizer: Any = None
        self._engine: Any = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._submissions: queue.Queue[tuple[str, str, Any, int]] = queue.Queue()
        self._aborts: queue.Queue[str] = queue.Queue()
        self._handles: dict[str, RequestHandle] = {}
        self._lock = threading.Lock()

    # --- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        """Build the engine and start the step loop. Loads the model."""
        from inferbench import engine as engine_mod

        # Forced, not merely defaulted: an in-process engine core is the
        # whole point here, and an inherited override would silently move it
        # to a child process where the ledger cannot see it.
        engine_mod.configure_env(force=True)
        vllm = engine_mod.require_vllm()

        self._vllm = vllm
        self._engine = vllm.LLMEngine.from_engine_args(  # type: ignore[attr-defined]
            vllm.EngineArgs(**self.cfg.engine_kwargs())
        )
        self.host = EngineHost(self._engine, model_name=self.cfg.model, vllm=vllm)
        self._tokenizer = self._engine.get_tokenizer()

        self._thread = threading.Thread(
            target=self._run, name="inferbench-step-engine", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the loop and shut the engine core down.

        The shutdown matters: the engine core holds the GPU, and leaving it
        behind is how a previous run ended up pinning 5 GB of VRAM.
        """
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None
        if self._engine is not None:
            self._engine.engine_core.shutdown()
            self._engine = None

    # --- the step loop --------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            self._drain()
            if not self._engine.has_unfinished_requests():
                self._wake.wait(timeout=0.05)
                self._wake.clear()
                # Heartbeat. The ledger has to stay honest even when the
                # engine is idle, because that is exactly when a held
                # request might fit.
                self._notify_step()
                continue
            outputs = self._engine.step()
            for out in outputs:
                self._route(out)
            self._notify_step()

    def _notify_step(self) -> None:
        if self._on_step is not None:
            self._on_step(self)

    def _drain(self) -> None:
        """Move queued submissions and aborts into the engine."""
        while True:
            try:
                request_id = self._aborts.get_nowait()
            except queue.Empty:
                break
            try:
                self._engine.abort_request([request_id])
            except Exception:  # a request that already finished is not an error
                pass

        while True:
            try:
                request_id, prompt, params, priority = self._submissions.get_nowait()
            except queue.Empty:
                return
            try:
                engine_id = self._engine.add_request(
                    request_id, prompt, params, priority=priority
                )
            except Exception as exc:
                handle = self._handles.get(request_id)
                if handle is not None:
                    handle.push(StreamEvent(error=f"{exc}", finished=True))
                    self._handles.pop(request_id, None)
            else:
                handle = self._handles.get(request_id)
                if handle is not None:
                    handle.engine_id = engine_id

    def _route(self, out: Any) -> None:
        request_id = getattr(out, "request_id", None)
        if request_id is None:
            return
        with self._lock:
            handle = self._handles.get(request_id)
        if handle is None:
            return

        text = ""
        finish_reason = "stop"
        completion_tokens = 0
        if out.outputs:
            seq = out.outputs[0]
            text = getattr(seq, "text", "") or ""
            finish_reason = getattr(seq, "finish_reason", None) or "stop"
            completion_tokens = len(getattr(seq, "token_ids", None) or ())

        prompt_tokens = getattr(out, "prompt_token_ids", None)
        if prompt_tokens is not None:
            handle.prompt_tokens = len(prompt_tokens)
        handle.completion_tokens = completion_tokens

        delta = text[len(handle.text) :]
        handle.text = text
        finished = bool(getattr(out, "finished", False))

        if delta or finished:
            handle.push(
                StreamEvent(
                    delta=delta,
                    finished=finished,
                    finish_reason=finish_reason,
                    prompt_tokens=handle.prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            )
        if finished:
            with self._lock:
                self._handles.pop(request_id, None)

    # --- requests -------------------------------------------------------------

    def submit(
        self,
        request_id: str,
        prompt: str,
        params: Any,
        on_event: Callable[[StreamEvent], None] | None = None,
        priority: int = 0,
    ) -> RequestHandle:
        """Queue a request for the engine. Returns its event handle."""
        handle = RequestHandle(request_id, on_event)
        with self._lock:
            self._handles[request_id] = handle
        self._submissions.put((request_id, prompt, params, priority))
        self._wake.set()
        return handle

    def abort(self, request_id: str) -> bool:
        """Cancel a request. False if it was not in flight."""
        with self._lock:
            handle = self._handles.pop(request_id, None)
        if handle is None:
            return False
        self._aborts.put(request_id)
        self._wake.set()
        handle.push(StreamEvent(finished=True, finish_reason="abort"))
        return True

    def in_flight(self) -> int:
        with self._lock:
            return len(self._handles)

    def make_params(
        self,
        *,
        max_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int | None = None,
        min_tokens: int = 0,
        ignore_eos: bool = False,
    ) -> Any:
        """Sampling parameters for one request.

        Deliberately not the benchmark's pinned-length params: production
        callers want the model to stop when it is done.
        """
        return self._vllm.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            # vLLM rejects min_tokens above max_tokens, and a caller asking
            # for both is asking for a length, not a floor.
            min_tokens=min(min_tokens, max_tokens),
            ignore_eos=ignore_eos,
        )

    # --- engine truth ---------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        """Token count for a prompt, tokenized the way the engine will.

        `add_special_tokens=True` matches vLLM's default `TokenizeParams`,
        so the block demand computed from this matches what the engine
        allocates.
        """
        tok = self._tokenizer
        if tok is None:
            return 0
        try:
            return len(tok.encode(text, add_special_tokens=True))
        except TypeError:
            return len(tok(text).input_ids)

    def free_blocks(self) -> int | None:
        return None if self.host is None else self.host.free_blocks()

    def total_blocks(self) -> int | None:
        return None if self.host is None else self.host.total_blocks()

    def request_blocks(self, request_id: str) -> int | None:
        if self.host is None:
            return None
        return self.host.request_blocks(self._engine_id(request_id))

    def request_block_ids(self, request_id: str) -> set[int] | None:
        if self.host is None:
            return None
        return self.host.request_block_ids(self._engine_id(request_id))

    def _engine_id(self, request_id: str) -> str:
        """The id the engine itself knows this request by.

        vLLM replaces the caller's request id with an internal one (`q1`
        becomes `q1-a2abe44e`) and keys its block pool by that internal id.
        `RequestOutput.request_id` and `abort_request()` both speak the
        caller's id, so the two meet here — and the block pool is the one
        place that only understands the internal one.
        """
        with self._lock:
            handle = self._handles.get(request_id)
        if handle is None or handle.engine_id is None:
            return request_id
        return handle.engine_id

    def preemptions(self) -> int:
        return 0 if self.host is None else self.host.preemptions()

    def block_size(self) -> int | None:
        return None if self.host is None else self.host.block_size()

    def max_model_len(self) -> int | None:
        return None if self.host is None else self.host.max_model_len()

    def request_counts(self) -> tuple[int | None, int | None]:
        return (None, None) if self.host is None else self.host.request_counts()
