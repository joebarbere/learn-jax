import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_cumulative_sum(impl):
    xs = jnp.arange(1.0, 6.0)
    out = jax.jit(impl.cumulative_sum)(xs)
    np.testing.assert_allclose(out, [1, 3, 6, 10, 15])
    assert out.shape == xs.shape and out.dtype == xs.dtype
    r = jax.random.normal(jax.random.key(0), (257,))
    np.testing.assert_allclose(impl.cumulative_sum(r), jnp.cumsum(r), rtol=1e-5, atol=1e-5)


def test_cumulative_sum_single(impl):
    np.testing.assert_allclose(impl.cumulative_sum(jnp.array([7.0])), [7.0])


def test_ema(impl):
    xs = jnp.array([0.0, 10.0, 10.0, 10.0])
    out = jax.jit(impl.ema, static_argnums=1)(xs, 0.5)
    np.testing.assert_allclose(out, [0.0, 5.0, 7.5, 8.75])
    # alpha = 1 reproduces the input exactly
    r = jax.random.normal(jax.random.key(1), (50,))
    np.testing.assert_allclose(impl.ema(r, 1.0), r, rtol=1e-6)


@pytest.mark.parametrize("a", [0.25, 2.0, 9.0, 12345.0])
def test_newton_sqrt(impl, a):
    est = jax.jit(impl.newton_sqrt)(a)
    assert est.dtype == jnp.float32
    assert float(est) == pytest.approx(np.sqrt(a), rel=1e-5)


def test_piecewise(impl):
    f = jax.jit(impl.piecewise)
    x = jnp.array([0.5, -2.0])
    np.testing.assert_allclose(f(0, x), x**2)
    np.testing.assert_allclose(f(1, x), jnp.sin(x), rtol=1e-6)
    np.testing.assert_allclose(f(2, x), -x)
    np.testing.assert_allclose(f(7, x), -x)  # clamped to the last branch


def test_bf16_relative_error(impl):
    x = jnp.array([1.0, 1.0 + 2.0**-9, 3.14159265, 1e-30, 6.5e4], jnp.float32)
    err = jax.jit(impl.bf16_relative_error)(x)
    assert err.dtype == jnp.float32 and err.shape == x.shape
    assert float(err[0]) == 0.0  # 1.0 is exactly representable
    assert float(jnp.max(err)) <= 2.0**-8 + 1e-7  # bf16 has 8 significant bits → half an ulp ≤ 2^-8
    assert float(err[2]) > 1e-4  # pi is not


def test_mixed_precision_matmul(impl):
    ka, kb = jax.random.split(jax.random.key(2))
    a = jax.random.normal(ka, (64, 128))
    b = jax.random.normal(kb, (128, 32))
    out = jax.jit(impl.mixed_precision_matmul)(a, b)
    assert out.dtype == jnp.float32 and out.shape == (64, 32)
    ref = a.astype(jnp.bfloat16).astype(jnp.float32) @ b.astype(jnp.bfloat16).astype(jnp.float32)
    np.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-3)


def test_stable_sum(impl):
    x = jnp.full((100_000,), 0.01, jnp.bfloat16)
    s = jax.jit(impl.stable_sum)(x)
    assert s.dtype == jnp.float32 and s.shape == ()
    exact = 100_000 * float(jnp.bfloat16(0.01))
    assert float(s) == pytest.approx(exact, rel=1e-4)
