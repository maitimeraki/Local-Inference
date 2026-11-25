"""GPU sampling via NVML.

Samples on a background thread for the duration of a run and reports the peak.
Peak, not mean, is the number that explains an OOM or a throttled run — a mean
hides the moment the card was full.

Degrades to "no GPU telemetry" rather than failing: a machine without NVML can
still measure latency.
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass


@dataclass
class GpuSample:
    ts: float
    index: int
    name: str | None = None
    util_ratio: float | None = None  # 0..1
    mem_used_bytes: int | None = None
    mem_total_bytes: int | None = None
    power_w: float | None = None
    temp_c: float | None = None


def nvml_available() -> bool:
    try:
        import pynvml

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        pynvml.nvmlShutdown()
        return count > 0
    except Exception:
        return False


def gpu_names() -> list[str]:
    """Names of the visible GPUs, for the run metadata."""
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            return [
                pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(i))
                for i in range(pynvml.nvmlDeviceGetCount())
            ]
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return []


def sample_once(index: int = 0) -> GpuSample | None:
    """One reading, or None when NVML is not usable."""
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(index)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            s = GpuSample(
                ts=time.time(),
                index=index,
                name=pynvml.nvmlDeviceGetName(h),
                mem_used_bytes=mem.used,
                mem_total_bytes=mem.total,
            )
            try:
                s.util_ratio = pynvml.nvmlDeviceGetUtilizationRates(h).gpu / 100.0
            except Exception:
                pass
            try:
                # NVML reports milliwatts.
                s.power_w = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
            except Exception:
                pass
            try:
                s.temp_c = float(pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))
            except Exception:
                pass
            return s
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return None


class GpuCollector:
    """Background sampler. Use as a context manager around a run."""

    def __init__(self, interval_s: float = 0.5, index: int = 0) -> None:
        self.interval_s = interval_s
        self.index = index
        self.samples: list[GpuSample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> GpuCollector:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def start(self) -> None:
        if self._thread is not None:
            return
        # Probe once up front: if NVML is unusable, do not spawn a thread that
        # will silently collect nothing.
        if sample_once(self.index) is None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            s = sample_once(self.index)
            if s is not None:
                self.samples.append(s)
            self._stop.wait(self.interval_s)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # -- derived -----------------------------------------------------------

    def peak_memory_bytes(self) -> int | None:
        values = [s.mem_used_bytes for s in self.samples if s.mem_used_bytes is not None]
        return max(values) if values else None

    def peak_util_ratio(self) -> float | None:
        values = [s.util_ratio for s in self.samples if s.util_ratio is not None]
        return max(values) if values else None

    def max_temp_c(self) -> float | None:
        values = [s.temp_c for s in self.samples if s.temp_c is not None]
        return max(values) if values else None

    def as_rows(self) -> list[dict]:
        return [asdict(s) for s in self.samples]
