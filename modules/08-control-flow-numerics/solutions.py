"""Module 08 reference solutions. Try the exercises first — `make test N=08` grades yours."""
import jax.numpy as jnp
from jax import lax


def cumulative_sum(xs):
    def step(total, x):
        total = total + x
        return total, total

    _, out = lax.scan(step, jnp.zeros((), xs.dtype), xs)
    return out


def ema(xs, alpha: float):
    def step(prev, x):
        new = alpha * x + (1 - alpha) * prev
        return new, new

    _, rest = lax.scan(step, xs[0], xs[1:])
    return jnp.concatenate([xs[:1], rest])


def newton_sqrt(a, tol: float = 1e-6, max_iter: int = 50):
    a = jnp.asarray(a, jnp.float32)
    x0 = jnp.where(a < 1, 1.0, a).astype(jnp.float32)

    def cond(state):
        x, prev, i = state
        return (jnp.abs(x - prev) >= tol * x) & (i < max_iter)

    def body(state):
        x, _, i = state
        return (x + a / x) / 2, x, i + 1

    first = (x0 + a / x0) / 2
    x, _, _ = lax.while_loop(cond, body, (first, x0, jnp.array(1)))
    return x


def piecewise(index, x):
    return lax.switch(index, [lambda v: v**2, jnp.sin, lambda v: -v], x)


def bf16_relative_error(x):
    x = jnp.asarray(x, jnp.float32)
    rounded = lax.reduce_precision(x, exponent_bits=8, mantissa_bits=7)  # bf16 rounding XLA can't elide
    return jnp.abs(rounded - x) / jnp.abs(x)


def mixed_precision_matmul(a, b):
    return jnp.matmul(a.astype(jnp.bfloat16), b.astype(jnp.bfloat16), preferred_element_type=jnp.float32)


def stable_sum(x):
    return jnp.sum(x.astype(jnp.float32))
