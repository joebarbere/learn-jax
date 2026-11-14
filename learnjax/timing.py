"""Timing JAX code correctly: JAX dispatches asynchronously, so always block on the result."""
from __future__ import annotations

import statistics
import time
from collections.abc import Callable

import jax


def timeit(fn: Callable, *args, warmup: int = 2, iters: int = 10) -> dict:
    """Run fn(*args) `warmup` times (compilation), then time `iters` runs. Returns seconds."""
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - t0)
    return {"median": statistics.median(times), "min": min(times), "max": max(times), "iters": iters}
