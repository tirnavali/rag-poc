"""Retrieval and generation latency benchmarks."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class LatencyReport:
    label: str
    times_ms: list[float] = field(default_factory=list)

    def record(self, elapsed_s: float) -> None:
        self.times_ms.append(elapsed_s * 1000)

    def percentile(self, p: float) -> float:
        if not self.times_ms:
            return 0.0
        sorted_t = sorted(self.times_ms)
        idx = int(len(sorted_t) * p / 100)
        return sorted_t[min(idx, len(sorted_t) - 1)]

    def summary(self) -> dict:
        return {
            "label": self.label,
            "count": len(self.times_ms),
            "mean_ms": sum(self.times_ms) / len(self.times_ms) if self.times_ms else 0,
            "p50_ms": self.percentile(50),
            "p95_ms": self.percentile(95),
            "p99_ms": self.percentile(99),
        }


def time_call(fn: Callable, *args, **kwargs) -> tuple[float, any]:
    """Call fn(*args, **kwargs) and return (elapsed_seconds, result)."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return time.perf_counter() - start, result
