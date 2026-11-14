"""Module 06 reference solutions. Try the exercises first — `make test N=06` grades yours."""
import statistics
import time
from collections import Counter

import jax
import jax.numpy as jnp


def count_primitives(f, *args) -> dict[str, int]:
    jaxpr = jax.make_jaxpr(f)(*args)
    return dict(Counter(eqn.primitive.name for eqn in jaxpr.jaxpr.eqns))


def jit_safe_clip(x, lo, hi):
    return jnp.where(x < lo, lo, jnp.where(x > hi, hi, x))


def power_series(x, n: int):
    total = jnp.zeros_like(x)
    term = jnp.ones_like(x)
    for _ in range(n + 1):
        total = total + term
        term = term * x
    return total


power_series_jit = jax.jit(power_series, static_argnames="n")


def counting_jit(f):
    counter = {"traces": 0}

    def traced(*args, **kwargs):
        counter["traces"] += 1  # a Python side effect: runs only while JAX traces
        return f(*args, **kwargs)

    return jax.jit(traced), counter


def pad_to_bucket(x, buckets=(8, 16, 32, 64)):
    n = x.shape[0]
    for size in sorted(buckets):
        if size >= n:
            return jnp.pad(x, (0, size - n)), n
    raise ValueError(f"length {n} exceeds the largest bucket {max(buckets)}")


def masked_mean(padded, length):
    mask = jnp.arange(padded.shape[0]) < length
    return jnp.sum(jnp.where(mask, padded, 0)) / length


def time_jitted(f, *args, iters: int = 20) -> dict:
    t0 = time.perf_counter()
    compiled = jax.jit(f).lower(*args).compile()
    compile_seconds = time.perf_counter() - t0
    jax.block_until_ready(compiled(*args))  # warm-up
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(compiled(*args))
        times.append(time.perf_counter() - t0)
    return {"compile_seconds": compile_seconds, "run_seconds": statistics.median(times)}
