"""Peak-memory probe (CUDA or host RSS).

Sums `torch.cuda.max_memory_allocated` over visible devices; `PAL_MEM_PROBE=rss`
switches to host max RSS. Returns 0 bytes when disabled or without CUDA.
"""

from __future__ import annotations

import os
import resource
import sys
from collections.abc import Iterator
from contextlib import contextmanager

import torch


def _rss_bytes() -> int:
    """Process max RSS (high-water mark) in bytes; normalizes Linux KB -> bytes."""
    ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(ru_maxrss)
    return int(ru_maxrss) * 1024


@contextmanager
def probe_peak_memory(enabled: bool = True) -> Iterator[dict]:
    """Context manager that records peak memory across the bracket.

    Yields a mutable dict the caller reads after the with-block:
        {"enabled": bool, "peak_bytes": int, "n_devices": int, "mode": str}

    Modes: "cuda" (default), "rss" (env `PAL_MEM_PROBE=rss`, process max RSS
    in bytes), and off.
    """
    mode = os.environ.get("PAL_MEM_PROBE", "cuda").lower()

    if not enabled:
        result: dict = {"enabled": False, "peak_bytes": 0, "n_devices": 0, "mode": "off"}
        yield result
        return

    if mode == "rss":
        result = {"enabled": True, "peak_bytes": 0, "n_devices": 0, "mode": "rss"}
        try:
            yield result
        finally:
            # ru_maxrss is a process-wide high-water mark, read at exit.
            result["peak_bytes"] = _rss_bytes()
        return

    use_cuda = torch.cuda.is_available()
    n_devices = torch.cuda.device_count() if use_cuda else 0
    result = {
        "enabled": use_cuda,
        "peak_bytes": 0,
        "n_devices": n_devices,
        "mode": "cuda" if use_cuda else "off",
    }

    if use_cuda:
        for i in range(n_devices):
            torch.cuda.synchronize(i)
            torch.cuda.reset_peak_memory_stats(i)

    try:
        yield result
    finally:
        if use_cuda:
            total = 0
            for i in range(n_devices):
                torch.cuda.synchronize(i)
                total += int(torch.cuda.max_memory_allocated(i))
            result["peak_bytes"] = total
