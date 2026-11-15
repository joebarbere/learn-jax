import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _sys(key, N=6):
    k1, k2, k3 = jax.random.split(key, 3)
    A = -jnp.exp(jax.random.normal(k1, (N,)))            # negative: stable
    B = jax.random.normal(k2, (N,))
    C = jax.random.normal(k3, (N,))
    return A, B, C


def _loop(A_bar, B_bar, C, xs):
    A_bar, B_bar, C, xs = map(np.asarray, (A_bar, B_bar, C, xs))
    h = np.zeros_like(A_bar)
    out = []
    for x in xs:
        h = A_bar * h + B_bar * x
        out.append((C * h).sum())
    return np.array(out)


def test_discretize_zoh(impl):
    A, B, _ = _sys(jax.random.key(0))
    A_bar, B_bar = impl.discretize_zoh(A, B, 0.1)
    np.testing.assert_allclose(np.asarray(A_bar), np.exp(0.1 * np.asarray(A)), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(B_bar), (np.exp(0.1 * np.asarray(A)) - 1) / np.asarray(A) * np.asarray(B),
                               rtol=1e-5)
    # tiny steps: ZOH approaches Euler (B_bar ≈ dt * B)
    _, Bs = impl.discretize_zoh(A, B, 1e-4)
    np.testing.assert_allclose(np.asarray(Bs), 1e-4 * np.asarray(B), rtol=1e-3)


def test_ssm_scan(impl):
    A, B, C = _sys(jax.random.key(1))
    A_bar, B_bar = jnp.exp(0.3 * A), 0.3 * B
    xs = jax.random.normal(jax.random.key(2), (40,))
    np.testing.assert_allclose(np.asarray(impl.ssm_scan(A_bar, B_bar, C, xs)), _loop(A_bar, B_bar, C, xs),
                               rtol=1e-4, atol=1e-5)


def test_ssm_kernel(impl):
    A, B, C = _sys(jax.random.key(3))
    A_bar, B_bar = jnp.exp(0.2 * A), 0.2 * B
    impulse = np.zeros(25)
    impulse[0] = 1.0
    np.testing.assert_allclose(np.asarray(impl.ssm_kernel(A_bar, B_bar, C, 25)), _loop(A_bar, B_bar, C, impulse),
                               rtol=1e-4, atol=1e-6)


def test_ssm_conv(impl):
    k = jax.random.normal(jax.random.key(4), (33,))
    xs = jax.random.normal(jax.random.key(5), (33,))
    want = np.convolve(np.asarray(k), np.asarray(xs))[:33]          # causal, no wrap-around
    np.testing.assert_allclose(np.asarray(impl.ssm_conv(k, xs)), want, rtol=1e-4, atol=1e-4)


def test_linear_recurrence_assoc(impl):
    a = jax.random.uniform(jax.random.key(6), (37, 3, 2), minval=0.5, maxval=1.0)
    b = jax.random.normal(jax.random.key(7), (37, 3, 2))
    an, bn = np.asarray(a), np.asarray(b)
    h, want = np.zeros((3, 2)), []
    for t in range(37):
        h = an[t] * h + bn[t]
        want.append(h)
    np.testing.assert_allclose(np.asarray(impl.linear_recurrence_assoc(a, b)), np.array(want), rtol=1e-4, atol=1e-5)
    # a = 0 means "forget everything": h_t = b_t
    np.testing.assert_allclose(np.asarray(impl.linear_recurrence_assoc(jnp.zeros_like(a), b)), bn, atol=1e-6)


def test_three_ways_agree(impl):
    A, B, C = _sys(jax.random.key(8), N=8)
    A_bar, B_bar = impl.discretize_zoh(A, B, 0.05)
    xs = jax.random.normal(jax.random.key(9), (128,))
    y_scan = impl.ssm_scan(A_bar, B_bar, C, xs)
    y_conv = impl.ssm_conv(impl.ssm_kernel(A_bar, B_bar, C, 128), xs)
    h = impl.linear_recurrence_assoc(jnp.broadcast_to(A_bar, (128, 8)), B_bar[None, :] * xs[:, None])
    y_assoc = h @ C
    np.testing.assert_allclose(np.asarray(y_conv), np.asarray(y_scan), atol=1e-3)
    np.testing.assert_allclose(np.asarray(y_assoc), np.asarray(y_scan), atol=1e-4)


@pytest.mark.parametrize("D,N,L", [(3, 4, 20), (1, 2, 5)])
def test_selective_scan(impl, D, N, L):
    ks = jax.random.split(jax.random.key(10 + D), 5)
    x = jax.random.normal(ks[0], (L, D))
    delta = jax.nn.softplus(jax.random.normal(ks[1], (L, D)))
    A = -jnp.exp(jax.random.normal(ks[2], (D, N)))
    B = jax.random.normal(ks[3], (L, N))
    C = jax.random.normal(ks[4], (L, N))
    xn, dn, An, Bn, Cn = map(np.asarray, (x, delta, A, B, C))
    h, want = np.zeros((D, N)), []
    for t in range(L):
        h = np.exp(dn[t][:, None] * An) * h + dn[t][:, None] * Bn[t][None, :] * xn[t][:, None]
        want.append(h @ Cn[t])
    np.testing.assert_allclose(np.asarray(impl.selective_scan(x, delta, A, B, C)), np.array(want),
                               rtol=1e-4, atol=1e-5)
