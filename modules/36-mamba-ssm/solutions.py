"""Module 36 reference solutions. Try the exercises first — `make test N=36` grades yours."""
import jax
import jax.numpy as jnp


def discretize_zoh(A, B, dt):
    A_bar = jnp.exp(dt * A)
    B_bar = (A_bar - 1.0) / A * B
    return A_bar, B_bar


def ssm_scan(A_bar, B_bar, C, xs):
    def step(h, x):
        h = A_bar * h + B_bar * x
        return h, jnp.sum(C * h)

    _, ys = jax.lax.scan(step, jnp.zeros_like(A_bar), xs)
    return ys


def ssm_kernel(A_bar, B_bar, C, L):
    powers = A_bar[None, :] ** jnp.arange(L)[:, None]          # (L, N)
    return powers @ (C * B_bar)


def ssm_conv(kernel, xs):
    L = xs.shape[0]
    n = 2 * L
    y = jnp.fft.irfft(jnp.fft.rfft(kernel, n) * jnp.fft.rfft(xs, n), n)
    return y[:L]


def linear_recurrence_assoc(a, b):
    def combine(left, right):
        a1, b1 = left
        a2, b2 = right
        return a1 * a2, a2 * b1 + b2

    _, h = jax.lax.associative_scan(combine, (a, b), axis=0)
    return h


def selective_scan(x, delta, A, B, C):
    def step(h, inp):
        x_t, d_t, B_t, C_t = inp
        A_bar = jnp.exp(d_t[:, None] * A)
        h = A_bar * h + d_t[:, None] * B_t[None, :] * x_t[:, None]
        return h, h @ C_t

    h0 = jnp.zeros((x.shape[1], A.shape[1]), x.dtype)
    _, ys = jax.lax.scan(step, h0, (x, delta, B, C))
    return ys
