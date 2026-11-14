"""Module 06 exercises: working *with* the compiler instead of against it.

Replace each `raise NotImplementedError` with your code, then run `make test N=06`.

Everything here is about what happens when `jax.jit` traces your function: which operations it
records, when it has to trace again, and how long compiling takes compared with running.
"""
import time  # noqa: F401  (you will need these)

import jax
import jax.numpy as jnp  # noqa: F401


def count_primitives(f, *args) -> dict[str, int]:
    """Trace f(*args) with `jax.make_jaxpr` and count the primitives in the top-level jaxpr.

    `jax.make_jaxpr(f)(*args)` returns a ClosedJaxpr; its `.jaxpr.eqns` is a list of equations, and
    each equation's `.primitive.name` is a string like "sin", "mul" or "dot_general".
    Return a dict {primitive name: how many times it appears}. Don't descend into nested jaxprs.
    """
    raise NotImplementedError


def jit_safe_clip(x, lo, hi):
    """Clip x elementwise into [lo, hi] — without Python `if`, `min()` or `max()`.

    It must work on arrays of any shape and *inside jax.jit*, where `if x < lo:` fails because the
    comparison is a traced value with no concrete True/False. Use `jnp.where` (or jnp.minimum/maximum).
    """
    raise NotImplementedError


def power_series(x, n: int):
    """Return 1 + x + x**2 + ... + x**n, built with a plain Python `for` loop over range(n + 1).

    The loop length depends on n, so n must be a *static* argument when this is jitted — see
    `power_series_jit` below, which is what the tests call.
    """
    raise NotImplementedError


# Make this work: jit power_series so that `power_series_jit(x, n=3)` compiles once per value of n.
# Hint: jax.jit(..., static_argnames=...). As written (no static args) calling it with n fails.
power_series_jit = jax.jit(power_series)


def counting_jit(f):
    """Wrap f with jax.jit and count how many times JAX *traces* it.

    Return (jitted_f, counter) where counter is a dict {"traces": 0}. Every time jitted_f is traced
    (first call, or a call with new shapes/dtypes/static values) counter["traces"] increases by 1;
    calls that reuse a compiled version don't change it.
    Hint: Python side effects inside a jitted function run only while tracing.
    """
    raise NotImplementedError


def pad_to_bucket(x, buckets=(8, 16, 32, 64)):
    """Pad a 1-D array with zeros up to the smallest bucket size >= len(x).

    Return (padded, length) where `length` is the original length as a Python int.
    Raise ValueError if len(x) is larger than every bucket.
    Why: a jitted function compiles once per input *shape*. Padding variable-length inputs to a few
    bucket sizes means a few compiles instead of one per length.
    """
    raise NotImplementedError


def masked_mean(padded, length):
    """Mean of the first `length` entries of `padded`, ignoring the zero padding.

    Must work under jax.jit with `length` as a *traced* integer (so no slicing `padded[:length]`,
    which needs a concrete length). Build a mask with jnp.arange and compare it to length.
    """
    raise NotImplementedError


def time_jitted(f, *args, iters: int = 20) -> dict:
    """Measure compile time and run time of jax.jit(f) separately.

    compile_seconds: wall time of `jax.jit(f).lower(*args).compile()`.
    run_seconds:     median wall time of calling the compiled function on args, with
                     `jax.block_until_ready` around each call (JAX dispatch is asynchronous!).
    Return {"compile_seconds": float, "run_seconds": float}.
    """
    raise NotImplementedError
