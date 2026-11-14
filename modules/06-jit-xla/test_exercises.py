import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_count_primitives(impl):
    counts = impl.count_primitives(lambda x: jnp.sin(x) * 2 + jnp.sin(x), 1.0)
    assert counts == {"sin": 2, "mul": 1, "add": 1}
    counts = impl.count_primitives(lambda a, b: a @ b, jnp.ones((2, 3)), jnp.ones((3, 4)))
    assert counts == {"dot_general": 1}


def test_jit_safe_clip(impl):
    x = jnp.array([-3.0, -0.5, 0.0, 0.7, 5.0])
    expected = np.clip(np.asarray(x), -1.0, 1.0)
    np.testing.assert_allclose(impl.jit_safe_clip(x, -1.0, 1.0), expected)
    jitted = jax.jit(impl.jit_safe_clip)
    np.testing.assert_allclose(jitted(x, -1.0, 1.0), expected)
    np.testing.assert_allclose(jitted(jnp.full((2, 2), 9.0), 0.0, 2.0), jnp.full((2, 2), 2.0))


def test_power_series_jit(impl):
    x = jnp.array([0.0, 0.5, 2.0])
    np.testing.assert_allclose(impl.power_series_jit(x, n=3), 1 + x + x**2 + x**3, rtol=1e-6)
    np.testing.assert_allclose(impl.power_series_jit(x, n=0), jnp.ones(3), rtol=1e-6)
    np.testing.assert_allclose(impl.power_series(x, 2), 1 + x + x**2, rtol=1e-6)


def test_counting_jit(impl):
    f, counter = impl.counting_jit(lambda x: x * 2)
    assert counter == {"traces": 0}
    f(jnp.ones(3))
    f(jnp.ones(3) * 5)  # same shape and dtype: reuses the compiled version
    assert counter["traces"] == 1
    f(jnp.ones(4))  # new shape: traces again
    assert counter["traces"] == 2
    np.testing.assert_allclose(f(jnp.arange(3.0)), [0.0, 2.0, 4.0])


def test_pad_to_bucket(impl):
    padded, n = impl.pad_to_bucket(jnp.arange(1.0, 6.0))
    assert padded.shape == (8,) and n == 5 and isinstance(n, int)
    np.testing.assert_allclose(padded, [1, 2, 3, 4, 5, 0, 0, 0])
    assert impl.pad_to_bucket(jnp.ones(16))[0].shape == (16,)  # exactly a bucket size: no padding
    assert impl.pad_to_bucket(jnp.ones(3), buckets=(4, 2))[0].shape == (4,)  # unsorted buckets
    with pytest.raises(ValueError):
        impl.pad_to_bucket(jnp.ones(65))


def test_masked_mean_compiles_once_per_bucket(impl):
    traces = []

    @jax.jit
    def mm(padded, length):
        traces.append(padded.shape)
        return impl.masked_mean(padded, length)

    for n in range(1, 21):
        x = jnp.arange(1.0, n + 1.0)
        padded, length = impl.pad_to_bucket(x)
        assert float(mm(padded, length)) == pytest.approx(float(x.mean()), rel=1e-5)
    assert len(traces) == 3  # buckets 8, 16, 32 — not 20 compilations


def test_time_jitted(impl):
    a = jnp.ones((256, 256))
    out = impl.time_jitted(lambda x: jnp.tanh(x @ x).sum(), a, iters=5)
    assert set(out) == {"compile_seconds", "run_seconds"}
    assert out["compile_seconds"] > 0 and out["run_seconds"] > 0
    assert out["compile_seconds"] > out["run_seconds"]  # compiling costs far more than one run
