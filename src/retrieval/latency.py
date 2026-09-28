"""Per request timing, reported as percentiles of individual calls.

Each stage is timed as a single call with perf_counter_ns around it, repeated
over many requests, and summarised by p50, p95 and p99. Medians are the
headline because a laptop has background load and a mean absorbs every
scheduler hiccup. Thread counts are set explicitly by the caller and recorded,
because a per request number at 12 threads and one at 1 thread answer
different questions.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, Iterable, List

import numpy as np


def summarise(ns: List[int]) -> Dict[str, float]:
    a = np.asarray(ns, dtype=np.float64) / 1e3  # microseconds
    if len(a) == 0:
        return {"n": 0}
    return {
        "n": int(len(a)),
        "p50_us": float(np.percentile(a, 50)),
        "p95_us": float(np.percentile(a, 95)),
        "p99_us": float(np.percentile(a, 99)),
        "mean_us": float(a.mean()),
        "min_us": float(a.min()),
    }


def time_calls(fn: Callable, args_iter: Iterable, warmup: int = 5) -> Dict[str, float]:
    """Call fn(*args) for each args tuple, timing each call on its own."""
    args_list = list(args_iter)
    for a in args_list[:warmup]:
        fn(*a)
    ns = []
    for a in args_list:
        t = time.perf_counter_ns()
        fn(*a)
        ns.append(time.perf_counter_ns() - t)
    return summarise(ns)


class StageTimer:
    """Collects named stage durations across many requests."""

    def __init__(self) -> None:
        self.ns: Dict[str, List[int]] = {}

    def add(self, stage: str, ns: int) -> None:
        self.ns.setdefault(stage, []).append(ns)

    def summary(self) -> Dict[str, Dict[str, float]]:
        return {k: summarise(v) for k, v in self.ns.items()}
