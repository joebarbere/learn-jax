"""Module 08 exercises: structured control flow and numerical precision.

Replace each `raise NotImplementedError` with your code, then run `make test N=08`.

Every function here must work under `jax.jit` — the tests call them jitted. That means no Python
`if`/`for`/`while` on *values* that depend on the inputs: use `lax.scan`, `lax.while_loop`,
`lax.switch` and `jnp.where` instead. (Python loops over a fixed, static number of steps are fine but
won't be needed.)
"""
import jax  # noqa: F401  (you will need these)
import jax.numpy as jnp  # noqa: F401
from jax import lax  # noqa: F401


def cumulative_sum(xs):
    """Running total of a 1-D array using `lax.scan` (not jnp.cumsum).

    cumulative_sum([1, 2, 3, 4]) -> [1, 3, 6, 10]. Output has the same shape and dtype as xs.
    Hint: the carry is the total so far; each step emits the new total.
    """
    raise NotImplementedError


def ema(xs, alpha: float):
    """Exponential moving average of a 1-D array using `lax.scan`.

    out[0] = xs[0]
    out[t] = alpha * xs[t] + (1 - alpha) * out[t-1]      for t >= 1

    `alpha` in (0, 1]: bigger alpha follows the data more closely. Return an array shaped like xs.
    """
    raise NotImplementedError


def newton_sqrt(a, tol: float = 1e-6, max_iter: int = 50):
    """Square root of a positive scalar `a` by Newton's method, using `lax.while_loop`.

    Start at x = a (or 1.0 if a < 1), iterate x <- (x + a / x) / 2, and stop when
    |x_new - x| < tol * x_new or after max_iter iterations. Return the float32 estimate.
    The loop state must be a tuple of arrays, e.g. (x, previous_x, iteration).
    """
    raise NotImplementedError


def piecewise(index, x):
    """Pick one of three functions of x with `lax.switch` — `index` may be a traced integer.

    index 0 -> x ** 2
    index 1 -> jnp.sin(x)
    index 2 -> -x
    Out-of-range indices are clamped by lax.switch (that's its documented behaviour).
    """
    raise NotImplementedError


def bf16_relative_error(x):
    """Relative error from rounding a float32 array to bfloat16 and back: |bf16(x) - x| / |x|.

    Return a float32 array shaped like x. Assume x has no zeros.

    Careful: under `jax.jit`, XLA is allowed to *delete* a float32 → bfloat16 → float32 round trip
    ("excess precision"), which would make the error 0. The lesson shows this happening. Use
    `lax.reduce_precision(x, exponent_bits=8, mantissa_bits=7)` — rounding XLA must keep.
    """
    raise NotImplementedError


def mixed_precision_matmul(a, b):
    """Matrix multiply with bfloat16 inputs and float32 accumulation and output.

    Cast a and b to bfloat16, multiply with `jnp.matmul(..., preferred_element_type=jnp.float32)`,
    and return the float32 result. This is how the matrix units on your GPU are meant to be used.
    """
    raise NotImplementedError


def stable_sum(x):
    """Sum an array that may be stored in bfloat16, accumulating in float32. Return a float32 scalar."""
    raise NotImplementedError
