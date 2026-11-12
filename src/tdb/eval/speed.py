"""End-to-end latency benchmark: one image in, detections out (batch size 1)."""

import platform
import statistics
import time
from collections.abc import Callable, Sequence

import numpy as np


def _sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def device_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return f"CPU ({platform.processor() or platform.machine()})"


def benchmark(
    predict: Callable[[np.ndarray], object],
    images: Sequence[np.ndarray],
    warmup: int = 20,
    runs: int = 200,
) -> dict:
    """Time `predict` per image, cycling through `images`; includes pre/post-processing."""
    if not images:
        raise ValueError("need at least one image")
    for i in range(warmup):
        predict(images[i % len(images)])
    _sync()

    times_ms = []
    for i in range(runs):
        start = time.perf_counter()
        predict(images[i % len(images)])
        _sync()
        times_ms.append((time.perf_counter() - start) * 1000)

    times_ms.sort()
    mean = statistics.fmean(times_ms)
    return {
        "device": device_name(),
        "runs": runs,
        "latency_ms_mean": round(mean, 2),
        "latency_ms_p50": round(times_ms[len(times_ms) // 2], 2),
        "latency_ms_p95": round(times_ms[int(len(times_ms) * 0.95) - 1], 2),
        "fps": round(1000 / mean, 1),
    }
